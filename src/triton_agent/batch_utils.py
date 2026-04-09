from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from io import TextIOBase
from pathlib import Path
from typing import TextIO

NO_CANDIDATE_OPERATOR_FILE = "found no candidate operator file after excluding generated artifacts"


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


def discover_batch_workspaces(
    root: Path,
    *,
    resolve_operator_file: Callable[[Path], Path],
    no_candidate_message: str = NO_CANDIDATE_OPERATOR_FILE,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, str]]]:
    workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    child_results: list[tuple[Path, str]] = []
    child_runnable: list[tuple[Path, Path]] = []

    for workspace in workspace_candidates:
        try:
            operator_file = resolve_operator_file(workspace)
        except ValueError as exc:
            child_results.append((workspace, str(exc)))
            continue
        child_runnable.append((workspace, operator_file))

    has_real_child_workspace = bool(child_runnable) or any(
        message != no_candidate_message for _, message in child_results
    )
    if has_real_child_workspace:
        return child_runnable, child_results

    try:
        operator_file = resolve_operator_file(root)
    except ValueError:
        return child_runnable, child_results
    return [(root, operator_file)], []


def resolve_batch_operator_file(
    workspace: Path,
    *,
    is_operator_candidate: Callable[[Path], bool],
    no_candidate_message: str = NO_CANDIDATE_OPERATOR_FILE,
) -> Path:
    candidates = [
        path for path in sorted(workspace.iterdir()) if path.is_file() and is_operator_candidate(path)
    ]
    if not candidates:
        raise ValueError(no_candidate_message)
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(f"found multiple candidate operator files: {names}")
    return candidates[0]


def is_batch_operator_candidate(
    path: Path,
    *,
    excluded_names: Iterable[str],
    excluded_prefixes: Iterable[str],
) -> bool:
    if path.suffix != ".py":
        return False
    if path.name in excluded_names:
        return False
    return not any(path.name.startswith(prefix) for prefix in excluded_prefixes)
