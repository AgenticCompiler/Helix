from __future__ import annotations

from pathlib import Path


def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"
