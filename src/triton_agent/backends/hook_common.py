from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class HookStageState:
    created_paths: list[Path]


@dataclass(frozen=True)
class HookStageOptions:
    trace_enabled: bool = False
    guard_enabled: bool = False
    trace_path: Path | None = None
    run_id: str | None = None


def cleanup_hook_stage(state: HookStageState) -> list[str]:
    warnings: list[str] = []
    for path in reversed(state.created_paths):
        try:
            if not path.exists() and not path.is_symlink():
                continue
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError as exc:
            warnings.append(f"Failed to clean up staged agent hook path {path}: {exc}")
    return warnings


def describe_prepare(state: HookStageState) -> list[str]:
    if not state.created_paths:
        return ["No backend-specific hooks staged."]
    return [f"Staged agent hooks: {', '.join(str(path) for path in state.created_paths)}"]


def describe_cleanup(state: HookStageState) -> list[str]:
    if not state.created_paths:
        return ["No backend-specific hooks to clean up."]
    return [f"Cleaning up staged agent hooks: {', '.join(str(path) for path in state.created_paths)}"]


def replace_string_placeholder(value: Any, placeholder: str, replacement: str) -> Any:
    if isinstance(value, str):
        return value.replace(placeholder, replacement)
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [replace_string_placeholder(item, placeholder, replacement) for item in items]
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {
            key: replace_string_placeholder(item, placeholder, replacement)
            for key, item in mapping.items()
        }
    return value
