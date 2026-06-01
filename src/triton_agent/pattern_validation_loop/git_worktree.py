from __future__ import annotations

import subprocess
from pathlib import Path


def resolve_git_worktree(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    cwd = candidate if candidate.is_dir() else candidate.parent
    if not cwd.exists():
        raise ValueError(f"Input path does not exist: {candidate}")
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "not a Git work tree"
        raise ValueError(f"Input path is not inside a Git work tree: {candidate} ({detail})")
    return Path(result.stdout.strip()).resolve()
