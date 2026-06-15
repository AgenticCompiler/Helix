"""Report-batch-state collector: scan batch-root and produce report-batch-state.json."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from triton_agent.optimize.batch import load_optimize_batch_status, optimize_batch_workspace_key
from triton_agent.status.core import inspect_optimize_status_workspace
from triton_agent.status.core import find_latest_verify_state, inspect_verify_state_summary

_BATCH_REPORT_STATE_FILENAME = "report-batch-state.json"
_SCHEMA_VERSION = 1

_SOURCE_FILES = [
    "optimize-batch-status.json",
    "opt-note.md",
    "opt-round-*/*_perf.txt",
    "opt-round-*/round-state.json",
    "opt-verify/verify-*/verify-state.json",
    "log_check_result.json",
    "pattern_analysis.json",
]


def _as_json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, Any], value)


def _as_json_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast(list[object], value)


def collect_report_batch_state(batch_root: Path) -> dict[str, Any]:
    """Scan batch-root and return the normalized report-batch-state dict."""
    batch_root = batch_root.resolve()
    now_iso = datetime.now(timezone.utc).isoformat()
    batch_status = load_optimize_batch_status(batch_root)

    workspaces = _discover_workspaces(batch_root)
    workspace_entries: list[dict[str, Any]] = []
    for ws_path in sorted(workspaces, key=lambda p: p.name):
        entry = _collect_workspace(batch_root, ws_path, batch_status)
        workspace_entries.append(entry)

    summary = _build_summary(workspace_entries)
    input_sources = _build_input_sources(batch_root, workspaces)

    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": now_iso,
        "batch_root": batch_root.as_posix(),
        "collector": {
            "name": "report-batch",
            "input_sources": input_sources,
        },
        "summary": summary,
        "workspaces": workspace_entries,
    }


def write_report_batch_state(batch_root: Path, output_path: Path | None = None) -> Path:
    """Collect and write report-batch-state.json. Returns the path written."""
    state = collect_report_batch_state(batch_root)
    target = output_path or (batch_root / _BATCH_REPORT_STATE_FILENAME)
    target.write_text(
        json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


# --- workspace discovery ---

def _discover_workspaces(root: Path) -> list[Path]:
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


# --- per-workspace collection ---

def _collect_workspace(
    batch_root: Path,
    ws_path: Path,
    batch_status: dict[str, dict[str, str]],
) -> dict[str, Any]:
    ws_key = optimize_batch_workspace_key(batch_root, ws_path)
    status_record = batch_status.get(ws_key)

    # status & operator_file
    if status_record is not None:
        raw_status = status_record.get("status", "incomplete")
        if raw_status in ("completed", "skipped"):
            status = raw_status
        else:
            status = "incomplete"
        operator_file = status_record.get("operator_file")
    else:
        status = "incomplete"
        operator_file = None

    # optimize.*
    optimize = _collect_optimize(ws_path)

    # verify.*
    verify = _collect_verify(ws_path)

    # check.*
    check = _collect_check(ws_path)

    # pattern.*
    pattern = _collect_pattern(ws_path)

    return {
        "workspace": ws_key if ws_key != "." else ws_path.name,
        "operator_file": operator_file,
        "status": status,
        "optimize": optimize,
        "verify": verify,
        "check": check,
        "pattern": pattern,
    }


def _collect_optimize(ws_path: Path) -> dict[str, Any]:
    try:
        ws_status = inspect_optimize_status_workspace(ws_path)
    except Exception:
        return {
            "status": "no-session",
            "round_count": 0,
            "best_round": None,
            "best_geomean_speedup": None,
        }
    round_dirs = sorted(
        p for p in ws_path.iterdir()
        if p.is_dir() and p.name.startswith("opt-round-")
    )
    return {
        "status": ws_status.state,
        "round_count": len(round_dirs),
        "best_round": ws_status.best_round,
        "best_geomean_speedup": ws_status.geomean_speedup,
    }


def _collect_verify(ws_path: Path) -> dict[str, Any]:
    state_path = find_latest_verify_state(ws_path)
    if state_path is None:
        return {
            "status": "skipped",
            "geomean_speedup": None,
        }
    try:
        passed, geomean = inspect_verify_state_summary(state_path)
    except Exception:
        return {
            "status": "skipped",
            "geomean_speedup": None,
        }
    return {
        "status": "passed" if passed else "failed",
        "geomean_speedup": geomean,
    }


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
        # Normalize: null detail for pass, keep detail for fail
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


def _build_input_sources(batch_root: Path, workspaces: list[Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for source in _SOURCE_FILES:
        present = _check_source_present(batch_root, source, workspaces)
        result.append({
            "file": source,
            "status": "present" if present else "missing",
        })
    return result


def _check_source_present(batch_root: Path, source: str, workspaces: list[Path]) -> bool:
    if not any(c in source for c in "*?["):
        if (batch_root / source).exists():
            return True
    for ws in workspaces:
        if list(ws.glob(source)):
            return True
    return False


# --- summary aggregation ---

def _build_summary(workspaces: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(workspaces)

    process_completed = 0
    process_incomplete = 0
    process_skipped = 0

    health_ok = 0
    health_warning = 0
    health_no_session = 0

    verify_passed = 0
    verify_failed = 0
    verify_skipped = 0

    check_passed = 0
    check_failed = 0
    check_skipped = 0

    for ws in workspaces:
        # process
        s = ws.get("status", "incomplete")
        if s == "completed":
            process_completed += 1
        elif s == "skipped":
            process_skipped += 1
        else:
            process_incomplete += 1

        # health
        opt = ws.get("optimize", {})
        opt_obj = _as_json_object(opt)
        if opt_obj is not None:
            hs = opt_obj.get("status", "no-session")
            if hs == "ok":
                health_ok += 1
            elif hs == "warning":
                health_warning += 1
            else:
                health_no_session += 1
        else:
            health_no_session += 1

        # verify
        vfy = ws.get("verify", {})
        vfy_obj = _as_json_object(vfy)
        if vfy_obj is not None:
            vs = vfy_obj.get("status", "skipped")
            if vs == "passed":
                verify_passed += 1
            elif vs == "failed":
                verify_failed += 1
            else:
                verify_skipped += 1
        else:
            verify_skipped += 1

        # check
        chk = ws.get("check", {})
        chk_obj = _as_json_object(chk)
        if chk_obj is not None:
            cs = chk_obj.get("status", "skipped")
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
            "process": {
                "completed": process_completed,
                "incomplete": process_incomplete,
                "skipped": process_skipped,
            },
            "health": {
                "ok": health_ok,
                "warning": health_warning,
                "no_session": health_no_session,
            },
        },
        "verify": {
            "passed": verify_passed,
            "failed": verify_failed,
            "skipped": verify_skipped,
        },
        "check": {
            "passed": check_passed,
            "failed": check_failed,
            "skipped": check_skipped,
        },
    }

