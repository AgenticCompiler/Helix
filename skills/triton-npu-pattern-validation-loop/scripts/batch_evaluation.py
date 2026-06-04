#!/usr/bin/env python3
"""Batch-level evaluation registry (ground truth outside operator workspaces)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BATCH_EVALUATION_FILENAME = "batch-evaluation.json"
_LEGACY_META_FILENAME = "validation-meta.json"
_SCHEMA_VERSION = 1


class BatchEvaluationError(RuntimeError):
    pass


def batch_evaluation_path(batch_root: Path) -> Path:
    return batch_root.expanduser().resolve() / BATCH_EVALUATION_FILENAME


def empty_batch_evaluation(*, batch_root: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "workspaces": {},
    }
    if batch_root is not None:
        payload["batch_root"] = batch_root.expanduser().resolve().as_posix()
    return payload


def load_batch_evaluation(batch_root: Path) -> dict[str, Any]:
    path = batch_evaluation_path(batch_root)
    if not path.is_file():
        return empty_batch_evaluation(batch_root=batch_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BatchEvaluationError(f"invalid {BATCH_EVALUATION_FILENAME}: {path}")
    workspaces = payload.setdefault("workspaces", {})
    if not isinstance(workspaces, dict):
        raise BatchEvaluationError(f"invalid workspaces map in {path}")
    return payload


def write_batch_evaluation(batch_root: Path, payload: dict[str, Any]) -> Path:
    path = batch_evaluation_path(batch_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = dict(payload)
    normalized["schema_version"] = _SCHEMA_VERSION
    workspaces = normalized.get("workspaces")
    if not isinstance(workspaces, dict):
        raise BatchEvaluationError("workspaces must be an object")
    path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def upsert_workspace_entry(
    batch_root: Path,
    workspace_name: str,
    entry: dict[str, Any],
) -> None:
    name = workspace_name.strip()
    if not name:
        raise BatchEvaluationError("workspace name is required")
    payload = load_batch_evaluation(batch_root)
    workspaces = payload["workspaces"]
    existing = workspaces.get(name)
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(entry)
        merged["workspace"] = name
        workspaces[name] = merged
    else:
        merged = dict(entry)
        merged["workspace"] = name
        workspaces[name] = merged
    write_batch_evaluation(batch_root, payload)


def resolve_workspace_meta(
    workspace: Path,
    *,
    batch_root: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace.expanduser().resolve()
    root = (batch_root or workspace.parent).expanduser().resolve()
    name = workspace.name

    payload = load_batch_evaluation(root)
    entry = payload.get("workspaces", {}).get(name)
    if isinstance(entry, dict):
        meta = dict(entry)
        meta.setdefault("workspace", name)
        return meta

    legacy_path = workspace / _LEGACY_META_FILENAME
    if legacy_path.is_file():
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        if isinstance(legacy, dict):
            legacy.setdefault("workspace", name)
            return legacy

    raise BatchEvaluationError(
        f"no evaluation entry for workspace {name!r} in {batch_evaluation_path(root).as_posix()} "
        f"and no legacy {_LEGACY_META_FILENAME}",
    )


def list_registered_workspace_names(batch_root: Path, *, include_completed: bool = False) -> list[str]:
    root = batch_root.expanduser().resolve()
    names: set[str] = set()
    payload = load_batch_evaluation(root)
    for name, entry in payload.get("workspaces", {}).items():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("validation_status", "")).strip().lower()
        if status == "completed" and not include_completed:
            continue
        names.add(str(name))

    for path in root.iterdir():
        if not path.is_dir():
            continue
        if (path / _LEGACY_META_FILENAME).is_file():
            names.add(path.name)

    return sorted(names)


def workspace_has_operator(workspace: Path, meta: dict[str, Any]) -> bool:
    operator_name = str(meta.get("operator_filename", "")).strip()
    if operator_name:
        return (workspace / operator_name).is_file()
    py_files = [
        path
        for path in workspace.iterdir()
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("test_")
    ]
    return len(py_files) == 1


def mark_workspace_completed(batch_root: Path, workspace_name: str) -> None:
    upsert_workspace_entry(
        batch_root,
        workspace_name,
        {
            "validation_status": "completed",
            "archived_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def migrate_legacy_workspace_meta(batch_root: Path) -> list[str]:
    """Import per-workspace validation-meta.json into batch-evaluation.json."""
    root = batch_root.expanduser().resolve()
    migrated: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith(".") or path.name == "_completed":
            continue
        legacy = path / _LEGACY_META_FILENAME
        if not legacy.is_file():
            continue
        meta = json.loads(legacy.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            continue
        upsert_workspace_entry(root, path.name, meta)
        migrated.append(path.name)
    return migrated
