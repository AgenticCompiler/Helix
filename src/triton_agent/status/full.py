"""Status schema collection: produce status-schema.json from status results."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

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

    workspace_entries: list[dict[str, Any]] = []
    for item in sorted(results, key=lambda r: r.workspace.name):
        workspace_entries.append(_build_workspace_entry(item))

    summary = _build_summary(workspace_entries)

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


# --- helper ---


def _as_json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, Any], value)


def _as_json_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast(list[object], value)


def _int_list(value: object) -> list[int]:
    items = _as_json_list(value)
    if items is None:
        return []
    result: list[int] = []
    for item in items:
        if isinstance(item, int):
            result.append(item)
        elif isinstance(item, (str, float)):
            try:
                result.append(int(item))
            except (ValueError, TypeError):
                pass
    return result


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


def _build_summary(workspace_entries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(workspace_entries)
    health_ok = 0
    health_warning = 0
    health_no_session = 0

    verified_count = 0
    unverified_count = 0
    no_verify = 0

    check_passed = 0
    check_failed = 0
    check_skipped = 0

    for entry in workspace_entries:
        state = entry.get("state", "no-session")
        if state == "ok":
            health_ok += 1
        elif state == "warning":
            health_warning += 1
        else:
            health_no_session += 1

        if entry.get("latest_verify_state") is not None:
            if entry.get("verified"):
                verified_count += 1
            else:
                unverified_count += 1
        else:
            no_verify += 1

        chk = _as_json_object(entry.get("check"))
        if chk is not None:
            cs = chk.get("status", "skipped")
            if cs == "passed":
                check_passed += 1
            elif cs == "failed":
                check_failed += 1
            else:
                check_skipped += 1
        else:
            check_skipped += 1

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
        "check": {
            "passed": check_passed,
            "failed": check_failed,
            "skipped": check_skipped,
        },
    }


# --- per-workspace entry ---


def _build_workspace_entry(item: OptimizeStatusWorkspace) -> dict[str, Any]:
    ws_path = item.workspace
    entry: dict[str, Any] = {
        "workspace": ws_path.name,
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

    entry["check"] = _collect_check(ws_path)
    entry["pattern"] = _collect_pattern(ws_path)

    return entry


def _collect_check(ws_path: Path) -> dict[str, Any]:
    json_path = ws_path / "log_check_result.json"
    if not json_path.is_file():
        return {
            "status": "skipped",
            "checks": [],
        }
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "skipped",
            "checks": [],
        }
    data = _as_json_object(payload)
    if data is None:
        return {
            "status": "skipped",
            "checks": [],
        }

    overall = data.get("overall", "UNKNOWN")
    if overall == "PASS":
        check_status = "passed"
    elif overall == "FAIL":
        check_status = "failed"
    else:
        check_status = "skipped"

    raw_check_items = _as_json_list(data.get("checks"))
    if raw_check_items is None:
        raw_check_items = []

    checks: list[dict[str, Any]] = []
    for check_value in raw_check_items:
        c = _as_json_object(check_value)
        if c is None:
            continue
        cid = c.get("id")
        if not isinstance(cid, str):
            continue
        result_value = c.get("result")
        result = result_value if result_value in ("pass", "fail") else "fail"
        detail_value = c.get("detail")
        if result == "pass":
            detail: str | None = None
        elif detail_value is None:
            detail = ""
        else:
            detail = str(detail_value)
        name = c.get("name")
        checks.append({
            "id": cid,
            "name": name if isinstance(name, str) else "",
            "result": result,
            "detail": detail,
        })

    return {
        "status": check_status,
        "checks": checks,
    }


def _collect_pattern(ws_path: Path) -> dict[str, Any]:
    empty: dict[str, list[dict[str, Any]]] = {
        "given": [],
        "new": [],
        "extended": [],
    }
    json_path = ws_path / "pattern_analysis.json"
    if not json_path.is_file():
        return empty
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    data = _as_json_object(payload)
    if data is None:
        return empty

    summary = _as_json_object(data.get("summary"))
    if summary is None:
        return empty

    given: list[dict[str, Any]] = []
    raw_given = _as_json_list(summary.get("given"))
    if raw_given is not None:
        for item_value in raw_given:
            item = _as_json_object(item_value)
            if item is not None and isinstance(item.get("name"), str):
                evidence = item.get("evidence")
                given.append({
                    "name": item["name"],
                    "rounds": _int_list(item.get("rounds")),
                    "evidence": evidence if evidence in ("explicit", "inferred") else "inferred",
                })

    new: list[dict[str, Any]] = []
    raw_new = _as_json_list(summary.get("new"))
    if raw_new is not None:
        for item_value in raw_new:
            item = _as_json_object(item_value)
            if item is not None and isinstance(item.get("name"), str):
                new.append({
                    "name": item["name"],
                    "rounds": _int_list(item.get("rounds")),
                })

    extended: list[dict[str, Any]] = []
    raw_ext = _as_json_list(summary.get("extended"))
    if raw_ext is not None:
        for item_value in raw_ext:
            item = _as_json_object(item_value)
            if item is not None and isinstance(item.get("name"), str):
                from_value = item.get("from")
                extended.append({
                    "name": item["name"],
                    "rounds": _int_list(item.get("rounds")),
                    "from": from_value if isinstance(from_value, str) else "",
                })

    return {
        "given": given,
        "new": new,
        "extended": extended,
    }


__all__ = [
    "collect_status_schema",
    "write_status_schema",
    "warn_missing_sources",
]
