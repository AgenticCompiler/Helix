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

from triton_agent.batch_utils import (
    NO_CANDIDATE_OPERATOR_FILE,
    PrefixedTextStream,
    discover_batch_workspaces,
    is_batch_operator_candidate,
    resolve_batch_operator_file,
)
from triton_agent.convert.models import ConvertOptions
from triton_agent.convert.orchestration import build_convert_request, run_convert_request
from triton_agent.mcp import managed_mcp_scope, managed_mcp_server_names_for_request
from triton_agent.models import AgentResult, CommandKind
from triton_agent.npu_affinity import (
    BatchNpuAffinityPool,
    affinity_env_for_device,
    configured_batch_npu_devices,
    configured_batch_npu_slots,
    validate_batch_affinity_capacity,
)
from triton_agent.skill_staging import resolve_staged_skills

_BATCH_CONVERT_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_", "triton_", "tilelang_")
_BATCH_CONVERT_EXCLUDED_NAMES = {"__init__.py"}


@dataclass(frozen=True)
class BatchConvertWorkspace:
    workspace: Path
    operator_file: Path


@dataclass(frozen=True)
class BatchConvertResult:
    workspace: Path
    succeeded: bool
    message: str


def run_convert_batch(
    root: Path,
    options: ConvertOptions,
    *,
    max_concurrency: int,
    operator_filter: str | None = None,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    convert_request_runner = run_request or run_convert_request
    discovered, failures = discover_batch_workspaces(
        root,
        resolve_operator_file=lambda workspace: resolve_batch_convert_operator_file(
            workspace,
            operator_filter=operator_filter,
        ),
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
    runnable = [
        BatchConvertWorkspace(workspace=workspace, operator_file=operator_file)
        for workspace, operator_file in discovered
    ]
    results = [
        BatchConvertResult(workspace=workspace, succeeded=False, message=message)
        for workspace, message in failures
    ]
    if not runnable and not results:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    output_lock = threading.Lock()
    stream = stdout or sys.stdout
    devices = configured_batch_npu_devices()
    if not options.enable_mcp:
        validate_batch_affinity_capacity(devices, max_concurrency=max_concurrency)
    affinity_pool = (
        BatchNpuAffinityPool(slots)
        if not options.enable_mcp and (slots := configured_batch_npu_slots()) is not None
        else None
    )

    def _run_item(
        item: BatchConvertWorkspace,
        forwarded_stdout: TextIO | None = None,
        forwarded_stderr: TextIO | None = None,
    ) -> AgentResult:
        request = build_convert_request(
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
                    return convert_request_runner(request, forwarded_stdout, forwarded_stderr)
                return convert_request_runner(request)
        if forwarded_stdout is not None or forwarded_stderr is not None:
            return convert_request_runner(request, forwarded_stdout, forwarded_stderr)
        return convert_request_runner(request)

    staged_skill_names, _ = resolve_staged_skills(CommandKind.CONVERT, enable_mcp=options.enable_mcp)
    scope = (
        managed_mcp_scope()
        if managed_mcp_server_names_for_request(staged_skill_names, enable_mcp=options.enable_mcp)
        else nullcontext()
    )
    with scope, ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[AgentResult], BatchConvertWorkspace] = {}
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
                    BatchConvertResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=f"unexpected convert failure: {exc}",
                    )
                )
                continue
            if result.succeeded:
                results.append(
                    BatchConvertResult(
                        workspace=item.workspace,
                        succeeded=True,
                        message=f"converted {item.operator_file.name}",
                    )
                )
            else:
                results.append(
                    BatchConvertResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=summarize_batch_convert_failure(result),
                    )
                )

    return render_batch_convert_results(results, stdout=stream)


def resolve_batch_convert_operator_file(
    workspace: Path,
    *,
    operator_filter: str | None = None,
) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_convert_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
        operator_filter=operator_filter,
    )


def is_batch_convert_operator_candidate(path: Path) -> bool:
    return is_batch_operator_candidate(
        path,
        excluded_names=_BATCH_CONVERT_EXCLUDED_NAMES,
        excluded_prefixes=_BATCH_CONVERT_EXCLUDED_PREFIXES,
    )


def summarize_batch_convert_failure(result: AgentResult) -> str:
    for output in (result.stderr, result.stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"convert exited with return code {result.return_code}"


def render_batch_convert_results(
    results: list[BatchConvertResult],
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
