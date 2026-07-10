from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Callable


@dataclass(frozen=True)
class ProgressSnapshot:
    latest_mtime: float | None
    file_fingerprints: tuple[tuple[str, int, float], ...]


def is_optimize_progress_path(path: Path, workspace: Path) -> bool:
    if not path.is_file():
        return False
    relative = path.relative_to(workspace)
    if relative == Path("opt-note.md") or relative == Path("learned_lessons.md"):
        return True
    parts = relative.parts
    if not parts:
        return False
    if parts[0] == "baseline":
        return True
    if re.fullmatch(r"opt-round-\d+", parts[0]) is not None:
        return True
    return False


def scan_optimize_progress(workspace: Path) -> ProgressSnapshot:
    entries: list[tuple[str, int, float]] = []
    _append_progress_file_fingerprint(workspace / "opt-note.md", workspace, entries)
    _append_progress_file_fingerprint(workspace / "learned_lessons.md", workspace, entries)
    _append_progress_tree_fingerprints(workspace / "baseline", workspace, entries)
    for round_dir in sorted(path for path in workspace.glob("opt-round-*") if path.is_dir()):
        _append_progress_tree_fingerprints(round_dir, workspace, entries)
    entries.sort(key=lambda item: item[0])
    latest = max((item[2] for item in entries), default=None)
    return ProgressSnapshot(latest_mtime=latest, file_fingerprints=tuple(entries))


def _append_progress_tree_fingerprints(
    root: Path,
    workspace: Path,
    entries: list[tuple[str, int, float]],
) -> None:
    if not root.is_dir():
        return

    for current_root, _dirnames, filenames in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=lambda _error: None,
    ):
        current_path = Path(current_root)
        for filename in sorted(filenames):
            _append_progress_file_fingerprint(current_path / filename, workspace, entries)


def _append_progress_file_fingerprint(
    path: Path,
    workspace: Path,
    entries: list[tuple[str, int, float]],
) -> None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    if not path.is_file():
        return
    rel = path.relative_to(workspace).as_posix()
    entries.append((rel, stat.st_size, stat.st_mtime))


def build_optimize_progress_probe(workspace: Path) -> Callable[[], float | None]:
    state = {"snapshot": scan_optimize_progress(workspace)}

    def probe() -> float | None:
        snapshot = scan_optimize_progress(workspace)
        if snapshot != state["snapshot"]:
            state["snapshot"] = snapshot
            return snapshot.latest_mtime
        return None

    return probe
