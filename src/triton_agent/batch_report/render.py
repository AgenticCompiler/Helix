"""Render report-batch.md from report-batch-state.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


def _as_json_object(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, Any], value)


def _as_json_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return cast(list[object], value)


def _as_json_objects(value: object) -> list[dict[str, Any]]:
    items = _as_json_list(value)
    if items is None:
        return []
    result: list[dict[str, Any]] = []
    for item in items:
        obj = _as_json_object(item)
        if obj is not None:
            result.append(obj)
    return result


def _string_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _int_value(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _rounds_string(value: object) -> str:
    items = _as_json_list(value)
    if items is None:
        return ""
    labels = [f"round-{item}" for item in items if isinstance(item, (int, str))]
    return ", ".join(labels)


def render_batch_report(state: dict[str, Any]) -> str:
    """Render report-batch.md from a report-batch-state dict."""
    lines: list[str] = []
    summary = _as_json_object(state.get("summary"))
    workspaces = _as_json_objects(state.get("workspaces"))

    lines.append("# Batch Report")
    lines.append("")
    lines.append(f"**Generated**: {_string_value(state.get('generated_at'), 'unknown')}")
    lines.append(f"**Batch root**: {_string_value(state.get('batch_root'), 'unknown')}")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    total = _int_value(summary.get("total_workspaces") if summary is not None else None)
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total workspaces | {total} |")

    _append_process_summary(lines, summary)
    _append_health_summary(lines, summary)
    _append_verify_summary(lines, summary)
    _append_check_summary(lines, summary)
    lines.append("")

    # --- Workspace table ---
    lines.append("## Workspaces")
    lines.append("")
    headers = [
        "Workspace",
        "Status",
        "Optimize",
        "Best Round",
        "Best Speedup",
        "Verify",
        "Check",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")

    for ws in workspaces:
        name = _string_value(ws.get("workspace"))
        status = _string_value(ws.get("status"))
        opt = _as_json_object(ws.get("optimize"))
        opt_status = _string_value(opt.get("status") if opt is not None else None)
        best_round_value = opt.get("best_round") if opt is not None else None
        best_round = _string_value(best_round_value, "-") if best_round_value is not None else "-"
        best_speedup = _format_speedup(opt.get("best_geomean_speedup") if opt is not None else None)
        verify = _as_json_object(ws.get("verify"))
        verify_status = _string_value(verify.get("status") if verify is not None else None)
        check = _as_json_object(ws.get("check"))
        check_status = _string_value(check.get("status") if check is not None else None)

        lines.append(
            f"| {name} | {status} | {opt_status} | {best_round} | {best_speedup} | {verify_status} | {check_status} |"
        )
    lines.append("")

    # --- Per-workspace details ---
    for ws in workspaces:
        _append_workspace_detail(lines, ws)

    return "\n".join(lines).rstrip() + "\n"


def _append_process_summary(lines: list[str], summary: dict[str, Any] | None) -> None:
    optimize = _as_json_object(summary.get("optimize")) if summary is not None else None
    process = _as_json_object(optimize.get("process")) if optimize is not None else None
    completed = _int_value(process.get("completed") if process is not None else None)
    incomplete = _int_value(process.get("incomplete") if process is not None else None)
    skipped = _int_value(process.get("skipped") if process is not None else None)
    lines.append(f"| Process — completed | {completed} |")
    lines.append(f"| Process — incomplete | {incomplete} |")
    lines.append(f"| Process — skipped | {skipped} |")


def _append_health_summary(lines: list[str], summary: dict[str, Any] | None) -> None:
    optimize = _as_json_object(summary.get("optimize")) if summary is not None else None
    health = _as_json_object(optimize.get("health")) if optimize is not None else None
    ok_val = _int_value(health.get("ok") if health is not None else None)
    warning_val = _int_value(health.get("warning") if health is not None else None)
    no_session = _int_value(health.get("no_session") if health is not None else None)
    lines.append(f"| Optimize health — ok | {ok_val} |")
    lines.append(f"| Optimize health — warning | {warning_val} |")
    lines.append(f"| Optimize health — no-session | {no_session} |")


def _append_verify_summary(lines: list[str], summary: dict[str, Any] | None) -> None:
    verify = _as_json_object(summary.get("verify")) if summary is not None else None
    passed = _int_value(verify.get("passed") if verify is not None else None)
    failed = _int_value(verify.get("failed") if verify is not None else None)
    skipped = _int_value(verify.get("skipped") if verify is not None else None)
    lines.append(f"| Verify — passed | {passed} |")
    lines.append(f"| Verify — failed | {failed} |")
    lines.append(f"| Verify — skipped | {skipped} |")


def _append_check_summary(lines: list[str], summary: dict[str, Any] | None) -> None:
    check = _as_json_object(summary.get("check")) if summary is not None else None
    passed = _int_value(check.get("passed") if check is not None else None)
    failed = _int_value(check.get("failed") if check is not None else None)
    skipped = _int_value(check.get("skipped") if check is not None else None)
    lines.append(f"| Check — passed | {passed} |")
    lines.append(f"| Check — failed | {failed} |")
    lines.append(f"| Check — skipped | {skipped} |")


def _append_workspace_detail(lines: list[str], ws: dict[str, Any]) -> None:
    name = _string_value(ws.get("workspace"))
    lines.append(f"### {name}")
    lines.append("")

    # check details
    check = _as_json_object(ws.get("check"))
    checks_list = _as_json_objects(check.get("checks") if check is not None else None)
    if checks_list:
        lines.append("**Check results:**")
        lines.append("")
        for c in checks_list:
            cid = _string_value(c.get("id"))
            cname = _string_value(c.get("name"))
            result = _string_value(c.get("result"))
            detail = c.get("detail")
            status_icon = "PASS" if result == "pass" else "FAIL"
            lines.append(f"- **{cid}** ({cname}): {status_icon}")
            if detail:
                lines.append(f"  - {detail}")
        lines.append("")

    # pattern summary
    pattern = _as_json_object(ws.get("pattern"))
    given = _as_json_objects(pattern.get("given") if pattern is not None else None)
    new = _as_json_objects(pattern.get("new") if pattern is not None else None)
    extended = _as_json_objects(pattern.get("extended") if pattern is not None else None)
    if given or new or extended:
        lines.append("**Patterns used:**")
        lines.append("")
        if given:
            lines.append("- Given:")
            for k in given:
                kname = _string_value(k.get("name"))
                kev = _string_value(k.get("evidence"))
                krnds_str = _rounds_string(k.get("rounds"))
                lines.append(f"  - {kname} [{krnds_str}] ({kev})")
        if new:
            lines.append("- New:")
            for n in new:
                nname = _string_value(n.get("name"))
                nrnds_str = _rounds_string(n.get("rounds"))
                lines.append(f"  - {nname} [{nrnds_str}]")
        if extended:
            lines.append("- Extended:")
            for e in extended:
                ename = _string_value(e.get("name"))
                ernds_str = _rounds_string(e.get("rounds"))
                efrom = _string_value(e.get("from"))
                lines.append(f"  - {ename} [{ernds_str}] (from: {efrom})")
        lines.append("")


def _format_speedup(value: object) -> str:
    if value is None:
        return "-"
    if not isinstance(value, (int, float, str)):
        return "-"
    try:
        return f"{float(value):.2f}x"
    except (ValueError, TypeError):
        return "-"


def render_batch_report_file(
    state_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Read report-batch-state.json and write report-batch.md."""
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    data = _as_json_object(payload)
    if data is None:
        raise ValueError(f"{state_path.name} did not contain a JSON object")
    target = output_path or (state_path.parent / "report-batch.md")
    md = render_batch_report(data)
    target.write_text(md, encoding="utf-8")
    return target


__all__ = [
    "render_batch_report",
    "render_batch_report_file",
]
