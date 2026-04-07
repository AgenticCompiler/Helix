from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from io import TextIOBase
from pathlib import Path
from typing import TextIO, cast

from triton_agent.models import AgentResult
from triton_agent.optimize.models import BatchOptimizeResult, BatchOptimizeWorkspace, OptimizeRunOptions
from triton_agent.optimize.render import render_batch_optimize_results
from triton_agent.optimize.runtime import build_optimize_request, run_optimize_request

_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}


class PrefixedTextStream(TextIOBase):
    def __init__(self, stream: TextIO, prefix: str, lock: threading.Lock) -> None:
        self._stream = stream
        self._prefix = prefix
        self._lock = lock
        self._at_line_start = True

    def write(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            for chunk in text.splitlines(keepends=True):
                if self._at_line_start:
                    self._stream.write(self._prefix)
                self._stream.write(chunk)
                self._at_line_start = chunk.endswith("\n")
            if text and not text.endswith(("\n", "\r")):
                self._at_line_start = False
            return len(text)

    def flush(self) -> None:
        with self._lock:
            self._stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._stream, "isatty", None)
        return bool(callable(isatty) and isatty())


def run_optimize_batch(
    root: Path,
    options: OptimizeRunOptions,
    *,
    max_concurrency: int,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    optimize_request_runner = run_request or run_optimize_request
    workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspace_candidates:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    results: list[BatchOptimizeResult] = []
    runnable: list[BatchOptimizeWorkspace] = []
    for workspace in workspace_candidates:
        try:
            operator_file = resolve_batch_optimize_operator_file(workspace)
        except ValueError as exc:
            results.append(BatchOptimizeResult(workspace=workspace, succeeded=False, message=str(exc)))
            continue
        runnable.append(BatchOptimizeWorkspace(workspace=workspace, operator_file=operator_file))

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
    candidates = [
        path
        for path in sorted(workspace.iterdir())
        if path.is_file() and is_batch_optimize_operator_candidate(path)
    ]
    if not candidates:
        raise ValueError("found no candidate operator file after excluding generated artifacts")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"found multiple candidate operator files: {names}")
    return candidates[0]


def is_batch_optimize_operator_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_OPTIMIZE_EXCLUDED_NAMES:
        return False
    return not any(path.name.startswith(prefix) for prefix in _BATCH_OPTIMIZE_EXCLUDED_PREFIXES)


def summarize_batch_optimize_failure(result: AgentResult) -> str:
    for output in (result.stderr, result.stdout):
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        if lines:
            return lines[-1]
    return f"optimize exited with return code {result.return_code}"
