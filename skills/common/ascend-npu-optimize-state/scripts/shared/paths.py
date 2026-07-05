from __future__ import annotations

from pathlib import Path


def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"


def existing_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def declared_state_file(state_dir: Path, workspace: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    declared_path = Path(relative_path)
    state_relative = existing_file(state_dir / declared_path)
    if state_relative is not None:
        return state_relative
    return existing_file(workspace / declared_path)


def missing_issue(relative_path: str | None, *, default_path: str) -> str:
    if relative_path is None:
        return f"missing {default_path}"
    return f"missing {relative_path}"
