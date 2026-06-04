from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Residual ground-truth files that must not live inside operator workspaces.
_OFFLINE_EVAL_FILENAMES = ("validation-meta.json", "manifest.json")
_HELD_SUBDIR = ".triton-agent/offline-eval-held"


@contextmanager
def isolate_workspace_for_simulate(workspace: Path) -> Iterator[None]:
    """Move offline-evaluation JSON out of the workspace while a simulate agent runs."""
    workspace = workspace.expanduser().resolve()
    held_dir = workspace / _HELD_SUBDIR
    moved: list[tuple[Path, Path]] = []
    try:
        for name in _OFFLINE_EVAL_FILENAMES:
            source = workspace / name
            if not source.is_file():
                continue
            held_dir.mkdir(parents=True, exist_ok=True)
            destination = held_dir / name
            if destination.exists():
                destination.unlink()
            source.rename(destination)
            moved.append((destination, source))
        yield
    finally:
        for held_path, original_path in reversed(moved):
            if held_path.is_file():
                held_path.rename(original_path)
        if held_dir.is_dir() and not any(held_dir.iterdir()):
            held_dir.rmdir()
            parent = held_dir.parent
            if parent.is_dir() and parent.name == ".triton-agent" and not any(parent.iterdir()):
                parent.rmdir()
