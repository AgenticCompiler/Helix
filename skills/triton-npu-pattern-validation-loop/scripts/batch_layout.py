#!/usr/bin/env python3
"""Shared batch-root layout helpers for pattern validation workspaces."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from batch_evaluation import (
    mark_workspace_completed,
    resolve_workspace_meta,
    workspace_has_operator,
)

COMPLETED_DIR_NAME = "_completed"
_LEGACY_META_FILENAME = "validation-meta.json"


class BatchLayoutError(RuntimeError):
    pass


def is_reserved_batch_subdir(name: str) -> bool:
    return name.startswith(".") or name == COMPLETED_DIR_NAME


def completed_root(batch_root: Path) -> Path:
    return batch_root / COMPLETED_DIR_NAME


def _workspace_dir(batch_root: Path, name: str) -> Path:
    return batch_root / name


def list_active_validation_workspaces(batch_root: Path) -> list[Path]:
    root = batch_root.expanduser().resolve()
    active: list[Path] = []
    for name in _active_workspace_names(root):
        workspace = _workspace_dir(root, name)
        if not workspace.is_dir():
            continue
        try:
            meta = resolve_workspace_meta(workspace, batch_root=root)
        except RuntimeError:
            continue
        if str(meta.get("validation_status", "")).strip().lower() == "completed":
            continue
        if workspace_has_operator(workspace, meta):
            active.append(workspace)
    return sorted(active)


def list_completed_validation_workspaces(batch_root: Path) -> list[Path]:
    root = completed_root(batch_root.expanduser().resolve())
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _active_workspace_names(batch_root: Path) -> list[str]:
    from batch_evaluation import list_registered_workspace_names

    return list_registered_workspace_names(batch_root, include_completed=False)


def active_workspace_count(batch_root: Path) -> int:
    return len(list_active_validation_workspaces(batch_root))


def archive_passed_workspace(workspace: Path, *, batch_root: Path) -> Path:
    batch_path = batch_root.expanduser().resolve()
    source = workspace.expanduser().resolve()
    if not source.is_dir():
        raise BatchLayoutError(f"workspace not found: {source}")
    if completed_root(batch_path) in source.parents or source.parent == completed_root(batch_path):
        raise BatchLayoutError(f"workspace already under completed: {source}")
    try:
        resolve_workspace_meta(source, batch_root=batch_path)
    except RuntimeError as exc:
        raise BatchLayoutError(str(exc)) from exc

    destination_root = completed_root(batch_path)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise BatchLayoutError(f"completed workspace already exists: {destination}")

    mark_workspace_completed(batch_path, source.name)

    legacy_meta = source / _LEGACY_META_FILENAME
    if legacy_meta.is_file():
        legacy_meta.unlink()

    shutil.move(source.as_posix(), destination.as_posix())
    return destination


def archive_passed_workspaces(
    reports: list[dict[str, Any]],
    *,
    batch_root: Path,
) -> list[Path]:
    archived: list[Path] = []
    active_by_name = {path.name: path for path in list_active_validation_workspaces(batch_root)}
    for report in reports:
        if not report.get("passed"):
            continue
        name = str(report.get("workspace", ""))
        workspace = active_by_name.get(name)
        if workspace is None:
            continue
        archived.append(archive_passed_workspace(workspace, batch_root=batch_root))
    return archived
