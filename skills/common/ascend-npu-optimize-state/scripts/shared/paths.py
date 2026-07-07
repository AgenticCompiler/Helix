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


def missing_path_issue(
    field_name: str,
    relative_path: str | None,
    *,
    expected_path: str | None = None,
) -> str:
    if relative_path is None:
        if expected_path is None:
            return f"missing required path field: {field_name}"
        return f"{field_name} is missing (expected {expected_path})"
    if expected_path is None:
        return f"{field_name} points to a missing file: {relative_path}"
    return f"{field_name} points to a missing file: {relative_path} (expected {expected_path})"


def unexpected_path_name_issue(field_name: str, relative_path: str, *, expected_name: str) -> str:
    return f"{field_name} must use {expected_name} (got {relative_path})"


def noncanonical_path_issue(field_name: str, relative_path: str, *, expected_path: str) -> str:
    return (
        f"{field_name} must point to the canonical baseline perf artifact "
        f"{expected_path} (got {relative_path})"
    )


def invalid_dependency_issue(field_name: str, dependency_label: str, reason: str) -> str:
    return f"cannot validate {field_name} because {dependency_label} is invalid: {reason}"
