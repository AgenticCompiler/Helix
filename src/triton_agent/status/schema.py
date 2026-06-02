"""Status schema collection: produce status-schema.json from status results."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_agent.optimize.models import OptimizeStatusWorkspace

_STATUS_SCHEMA_FILENAME = "status-schema.json"
_SCHEMA_VERSION = 1

_SOURCE_FILES = [
    "optimize-batch-status.json",
    "opt-note.md",
    "opt-round-*/*_perf.txt",
    "opt-round-*/round-state.json",
    "baseline/perf.txt",
    "opt-verify/verify-*/verify-state.json",
    "log_check_result.json",
    "pattern_analysis.json",
]


def collect_status_schema(
    root: Path,
    results: list[OptimizeStatusWorkspace],
) -> dict[str, Any]:
    root = root.resolve()
    now_iso = datetime.now(timezone.utc).isoformat()

    input_sources = _build_input_sources(root)

    summary = _build_summary(results)

    workspace_entries: list[dict[str, Any]] = []
    for item in sorted(results, key=lambda r: r.workspace.name):
        workspace_entries.append(_build_workspace_entry(item))

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": now_iso,
        "root": root.as_posix(),
        "collector": {
            "name": "status",
            "input_sources": input_sources,
        },
        "summary": summary,
        "workspaces": workspace_entries,
    }


def write_status_schema(
    root: Path,
    results: list[OptimizeStatusWorkspace],
    output_path: Path | None = None,
) -> Path:
    state = collect_status_schema(root, results)
    target = output_path or (root / _STATUS_SCHEMA_FILENAME)
    target.write_text(
        json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def warn_missing_sources(root: Path) -> None:
    sources = _build_input_sources(root)
    missing = [s["file"] for s in sources if s["status"] == "missing"]
    if not missing:
        return
    for source_name in missing:
        print(
            f"[status-schema] warning: missing source file: {source_name}",
            file=sys.stderr,
            flush=True,
        )


# --- input source checking ---


def _build_input_sources(root: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    workspaces = sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    for source in _SOURCE_FILES:
        present = _check_source_present(root, source, workspaces)
        result.append({
            "file": source,
            "status": "present" if present else "missing",
        })
    return result


def _check_source_present(root: Path, source: str, workspaces: list[Path]) -> bool:
    if not any(c in source for c in "*?["):
        if (root / source).exists():
            return True
    for ws in workspaces:
        if list(ws.glob(source)):
            return True
    return False


# --- summary aggregation ---


def _build_summary(results: list[OptimizeStatusWorkspace]) -> dict[str, Any]:
    total = len(results)
    health_ok = 0
    health_warning = 0
    health_no_session = 0

    verified_count = 0
    unverified_count = 0
    no_verify = 0

    for item in results:
        if item.state == "ok":
            health_ok += 1
        elif item.state == "warning":
            health_warning += 1
        else:
            health_no_session += 1

        if item.latest_verify_state is not None:
            if item.verified:
                verified_count += 1
            else:
                unverified_count += 1
        else:
            no_verify += 1

    return {
        "total_workspaces": total,
        "optimize": {
            "health": {
                "ok": health_ok,
                "warning": health_warning,
                "no_session": health_no_session,
            },
        },
        "verify": {
            "verified": verified_count,
            "unverified": unverified_count,
            "no_verify": no_verify,
        },
    }


# --- per-workspace entry ---


def _build_workspace_entry(item: OptimizeStatusWorkspace) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "workspace": item.workspace.name,
        "state": item.state,
    }

    if item.state != "no-session":
        entry["avg_improvement"] = item.avg_improvement
        entry["geomean_speedup"] = item.geomean_speedup
        entry["best_round"] = item.best_round
        entry["logged_best"] = item.logged_best

    if item.warnings:
        entry["warnings"] = list(item.warnings)

    entry["verified"] = item.verified
    entry["verified_geomean_speedup"] = item.verified_geomean_speedup
    if item.latest_verify_state is not None:
        entry["latest_verify_state"] = item.latest_verify_state.as_posix()

    return entry


__all__ = [
    "collect_status_schema",
    "write_status_schema",
    "warn_missing_sources",
]
