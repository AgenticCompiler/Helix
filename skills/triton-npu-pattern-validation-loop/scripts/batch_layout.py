#!/usr/bin/env python3
"""Shared batch-root layout helpers for pattern validation workspaces."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPLETED_DIR_NAME = "_completed"


class BatchLayoutError(RuntimeError):
    pass


def is_reserved_batch_subdir(name: str) -> bool:
    return name.startswith(".") or name == COMPLETED_DIR_NAME


def completed_root(batch_root: Path) -> Path:
    return batch_root / COMPLETED_DIR_NAME


def list_active_validation_workspaces(batch_root: Path) -> list[Path]:
    root = batch_root.expanduser().resolve()
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not is_reserved_batch_subdir(path.name)
        and (path / "validation-meta.json").is_file()
    )


def list_completed_validation_workspaces(batch_root: Path) -> list[Path]:
    root = completed_root(batch_root.expanduser().resolve())
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "validation-meta.json").is_file()
    )


def active_workspace_count(batch_root: Path) -> int:
    return len(list_active_validation_workspaces(batch_root))


def archive_passed_workspace(workspace: Path, *, batch_root: Path) -> Path:
    batch_path = batch_root.expanduser().resolve()
    source = workspace.expanduser().resolve()
    if not source.is_dir():
        raise BatchLayoutError(f"workspace not found: {source}")
    if completed_root(batch_path) in source.parents or source.parent == completed_root(batch_path):
        raise BatchLayoutError(f"workspace already under completed: {source}")
    meta_path = source / "validation-meta.json"
    if not meta_path.is_file():
        raise BatchLayoutError(f"missing validation-meta.json: {source}")

    destination_root = completed_root(batch_path)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise BatchLayoutError(f"completed workspace already exists: {destination}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise BatchLayoutError(f"invalid validation-meta.json: {meta_path}")
    meta["validation_status"] = "completed"
    meta["archived_at"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
