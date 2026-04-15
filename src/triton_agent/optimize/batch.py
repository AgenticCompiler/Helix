from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TextIO, cast

from triton_agent.batch_utils import (
    NO_CANDIDATE_OPERATOR_FILE,
    PrefixedTextStream,
    discover_batch_workspaces,
    is_batch_operator_candidate,
    resolve_batch_operator_file,
)
from triton_agent.models import AgentResult
from triton_agent.optimize.models import BatchOptimizeResult, BatchOptimizeWorkspace, OptimizeRunOptions
from triton_agent.optimize.render import render_batch_optimize_results
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request

_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}


def run_optimize_batch(
    root: Path,
    options: OptimizeRunOptions,
    *,
    max_concurrency: int,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    optimize_request_runner = run_request or run_optimize_request
    discovered, failures = discover_batch_workspaces(
        root,
        resolve_operator_file=resolve_batch_optimize_operator_file,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
    runnable = [
        BatchOptimizeWorkspace(workspace=workspace, operator_file=operator_file)
        for workspace, operator_file in discovered
    ]
    results = [
        BatchOptimizeResult(workspace=workspace, succeeded=False, message=message)
        for workspace, message in failures
    ]
    if not runnable and not results:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    output_lock = threading.Lock()
    stream = stdout or sys.stdout

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[AgentResult], BatchOptimizeWorkspace] = {}
        for item in runnable:
            try:
                request = build_optimize_request(item.operator_file, item.workspace, options)
            except ValueError as exc:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=str(exc),
                    )
                )
                continue
            if options.show_output:
                prefix = f"[{item.workspace.name}] "
                prefixed_stream = PrefixedTextStream(stream, prefix, output_lock)
                forwarded_stream = cast(TextIO, prefixed_stream)
                futures[
                    executor.submit(optimize_request_runner, request, forwarded_stream, forwarded_stream)
                ] = item
            else:
                futures[executor.submit(optimize_request_runner, request)] = item

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive boundary
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=f"unexpected optimize failure: {exc}",
                    )
                )
                continue
            if result.succeeded:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=True,
                        message=f"optimized {item.operator_file.name}",
                    )
                )
            else:
                results.append(
                    BatchOptimizeResult(
                        workspace=item.workspace,
                        succeeded=False,
                        message=summarize_batch_optimize_failure(result),
                    )
                )

    return render_batch_optimize_results(results, stdout=stream)


def resolve_batch_optimize_operator_file(workspace: Path) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_optimize_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )


def is_batch_optimize_operator_candidate(path: Path) -> bool:
    return is_batch_operator_candidate(
        path,
        excluded_names=_BATCH_OPTIMIZE_EXCLUDED_NAMES,
        excluded_prefixes=_BATCH_OPTIMIZE_EXCLUDED_PREFIXES,
    )


def summarize_batch_optimize_failure(result: AgentResult) -> str:
    for output in (result.stderr, result.stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"optimize exited with return code {result.return_code}"
