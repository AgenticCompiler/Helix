from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

from helix.batch.discovery import (
    NO_CANDIDATE_OPERATOR_FILE,
    PrefixedTextStream,
    discover_batch_workspaces,
    is_batch_operator_candidate,
    resolve_batch_operator_file,
)
from helix.generation.models import GenerationOptions
from helix.generation.orchestration import build_generation_request, run_generation_request
from helix.eval.mcp import managed_mcp_scope, managed_mcp_server_names_for_request
from helix.models import AgentResult, CommandKind
from helix.batch.affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_devices,
    configured_batch_npu_slots,
    parse_batch_npu_devices,
    validate_batch_affinity_capacity,
)
from helix.skills.selection import resolve_staged_skills

_BATCH_GEN_EVAL_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_GEN_EVAL_EXCLUDED_NAMES = {"__init__.py"}


@dataclass(frozen=True)
class BatchGenEvalWorkspace:
    workspace: Path
    operator_file: Path


@dataclass(frozen=True)
class BatchGenEvalResult:
    workspace: Path
    succeeded: bool
    message: str


def run_gen_eval_batch(
    root: Path,
    options: GenerationOptions,
    *,
    max_concurrency: int,
    operator_filter: str | None = None,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    generation_request_runner = run_request or run_generation_request
    discovered, failures = discover_batch_workspaces(
        root,
        resolve_operator_file=lambda workspace: resolve_batch_gen_eval_operator_file(
            workspace,
            operator_filter=operator_filter,
        ),
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
    runnable = [
        BatchGenEvalWorkspace(workspace=workspace, operator_file=operator_file)
        for workspace, operator_file in discovered
    ]
    results = [
        BatchGenEvalResult(workspace=workspace, succeeded=False, message=message)
        for workspace, message in failures
    ]
    if not runnable and not results:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    output_lock = threading.Lock()
    stream = stdout or sys.stdout
    devices = (
        configured_batch_npu_devices()
        if options.npu_devices is None
        else parse_batch_npu_devices(options.npu_devices)
    )
    validate_batch_affinity_capacity(
        devices,
        max_concurrency=max_concurrency,
        workers_per_npu_raw=options.workers_per_npu,
        ignore_workers_per_npu=options.enable_mcp,
    )
    affinity_pool = (
        BatchNpuAffinityPool(slots)
        if not options.enable_mcp
        and (slots := configured_batch_npu_slots(options.npu_devices, options.workers_per_npu)) is not None
        else None
    )

    def _run_item(
        item: BatchGenEvalWorkspace,
        forwarded_stdout: TextIO | None = None,
        forwarded_stderr: TextIO | None = None,
    ) -> AgentResult:
        request = build_generation_request(
            CommandKind.GEN_EVAL,
            item.operator_file,
            item.operator_file,
            item.workspace,
            options,
        )
        if affinity_pool is not None:
            with affinity_pool.acquire() as device:
                request = replace(
                    request,
                    extra_env={
                        **(request.extra_env or {}),
                        **affinity_env_for_device(device),
                    },
                )
                if forwarded_stdout is not None or forwarded_stderr is not None:
                    return generation_request_runner(request, forwarded_stdout, forwarded_stderr)
                return generation_request_runner(request)
        if forwarded_stdout is not None or forwarded_stderr is not None:
            return generation_request_runner(request, forwarded_stdout, forwarded_stderr)
        return generation_request_runner(request)

    staged_skill_names, _ = resolve_staged_skills(CommandKind.GEN_EVAL, enable_mcp=options.enable_mcp)
    scope = (
        managed_mcp_scope(
            npu_devices=options.npu_devices,
            workers_per_npu=options.workers_per_npu,
        )
        if managed_mcp_server_names_for_request(staged_skill_names, enable_mcp=options.enable_mcp)
        else nullcontext()
    )
    with scope, ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[AgentResult], BatchGenEvalWorkspace] = {}
        for item in runnable:
            if options.stream_output:
                prefix = f"[{item.workspace.name}] "
                prefixed_stream = PrefixedTextStream(stream, prefix, output_lock)
                forwarded_stream = cast(TextIO, prefixed_stream)
                futures[executor.submit(_run_item, item, forwarded_stream, forwarded_stream)] = item
            else:
                futures[executor.submit(_run_item, item)] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                results.append(
                    BatchGenEvalResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=f"unexpected gen-eval failure: {exc}",
                    )
                )
                continue
            if result.succeeded:
                results.append(
                    BatchGenEvalResult(
                        workspace=item.workspace,
                        succeeded=True,
                        message=f"generated eval artifacts for {item.operator_file.name}",
                    )
                )
            else:
                results.append(
                    BatchGenEvalResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=summarize_batch_gen_eval_failure(result),
                    )
                )

    return render_batch_gen_eval_results(results, stdout=stream)


def resolve_batch_gen_eval_operator_file(
    workspace: Path,
    *,
    operator_filter: str | None = None,
) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_gen_eval_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
        operator_filter=operator_filter,
    )


def is_batch_gen_eval_operator_candidate(path: Path) -> bool:
    return is_batch_operator_candidate(
        path,
        excluded_names=_BATCH_GEN_EVAL_EXCLUDED_NAMES,
        excluded_prefixes=_BATCH_GEN_EVAL_EXCLUDED_PREFIXES,
    )


def summarize_batch_gen_eval_failure(result: AgentResult) -> str:
    for output in (result.stderr, result.stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"gen-eval exited with return code {result.return_code}"


def render_batch_gen_eval_results(
    results: list[BatchGenEvalResult],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    ordered_results = sorted(results, key=lambda item: item.workspace.name)
    succeeded = sum(1 for item in ordered_results if item.succeeded)
    failed = len(ordered_results) - succeeded
    for item in ordered_results:
        status = "OK" if item.succeeded else "FAIL"
        print(f"[{status}] {item.workspace.name}: {item.message}", file=stream)
    print(f"Summary: {succeeded} succeeded, {failed} failed", file=stream)
    return 0 if failed == 0 and ordered_results else 1
