from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import TextIOBase
from pathlib import Path
from typing import TextIO, cast

from triton_agent.generation import GenerationOptions, build_generation_request, run_generation_request
from triton_agent.models import AgentResult, CommandKind

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


def run_gen_eval_batch(
    root: Path,
    options: GenerationOptions,
    *,
    max_concurrency: int,
    stdout: TextIO | None = None,
    run_request: Callable[..., AgentResult] | None = None,
) -> int:
    generation_request_runner = run_request or run_generation_request
    workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspace_candidates:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1

    results: list[BatchGenEvalResult] = []
    runnable: list[BatchGenEvalWorkspace] = []
    for workspace in workspace_candidates:
        try:
            operator_file = resolve_batch_gen_eval_operator_file(workspace)
        except ValueError as exc:
            results.append(BatchGenEvalResult(workspace=workspace, succeeded=False, message=str(exc)))
            continue
        runnable.append(BatchGenEvalWorkspace(workspace=workspace, operator_file=operator_file))

    output_lock = threading.Lock()
    stream = stdout or sys.stdout

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: dict[Future[AgentResult], BatchGenEvalWorkspace] = {}
        for item in runnable:
            request = build_generation_request(
                CommandKind.GEN_EVAL,
                item.operator_file,
                item.operator_file,
                item.workspace,
                options,
            )
            if options.show_output:
                prefix = f"[{item.workspace.name}] "
                prefixed_stream = PrefixedTextStream(stream, prefix, output_lock)
                forwarded_stream = cast(TextIO, prefixed_stream)
                futures[
                    executor.submit(generation_request_runner, request, forwarded_stream, forwarded_stream)
                ] = item
            else:
                futures[executor.submit(generation_request_runner, request)] = item

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


def resolve_batch_gen_eval_operator_file(workspace: Path) -> Path:
    candidates = [
        path
        for path in sorted(workspace.iterdir())
        if path.is_file() and is_batch_gen_eval_operator_candidate(path)
    ]
    if not candidates:
        raise ValueError("found no candidate operator file after excluding generated artifacts")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"found multiple candidate operator files: {names}")
    return candidates[0]


def is_batch_gen_eval_operator_candidate(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_GEN_EVAL_EXCLUDED_NAMES:
        return False
    return not any(path.name.startswith(prefix) for prefix in _BATCH_GEN_EVAL_EXCLUDED_PREFIXES)


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
