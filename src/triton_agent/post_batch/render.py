"""Render post-batch-report.md from post-batch-state.json."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_post_batch_report(state: dict[str, Any]) -> str:
    """Render post-batch-report.md from a post-batch-state dict."""
    lines: list[str] = []
    summary = state.get("summary", {})
    workspaces: list[dict[str, Any]] = state.get("workspaces", [])

    lines.append("# Post-Batch Report")
    lines.append("")
    lines.append(f"**Generated**: {state.get('generated_at', 'unknown')}")
    lines.append(f"**Batch root**: {state.get('batch_root', 'unknown')}")
    lines.append("")

    # --- Summary ---
    lines.append("## Summary")
    lines.append("")
    total = summary.get("total_workspaces", 0)
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
        name = ws.get("workspace", "")
        status = ws.get("status", "")
        opt = ws.get("optimize", {}) or {}
        opt_status = opt.get("status", "")
        best_round = opt.get("best_round") or "-"
        best_speedup = _format_speedup(opt.get("best_geomean_speedup"))
        verify = ws.get("verify", {}) or {}
        verify_status = verify.get("status", "")
        check = ws.get("check", {}) or {}
        check_status = check.get("status", "")

        lines.append(
            f"| {name} | {status} | {opt_status} | {best_round} | {best_speedup} | {verify_status} | {check_status} |"
        )
    lines.append("")

    # --- Per-workspace details ---
    for ws in workspaces:
        _append_workspace_detail(lines, ws)

    return "\n".join(lines).rstrip() + "\n"


def _append_process_summary(lines: list[str], summary: dict[str, Any]) -> None:
    optimize = summary.get("optimize", {}) or {}
    process = optimize.get("process", {}) or {}
    completed = process.get("completed", 0)
    incomplete = process.get("incomplete", 0)
    skipped = process.get("skipped", 0)
    lines.append(f"| Process — completed | {completed} |")
    lines.append(f"| Process — incomplete | {incomplete} |")
    lines.append(f"| Process — skipped | {skipped} |")


def _append_health_summary(lines: list[str], summary: dict[str, Any]) -> None:
    optimize = summary.get("optimize", {}) or {}
    health = optimize.get("health", {}) or {}
    ok_val = health.get("ok", 0)
    warning_val = health.get("warning", 0)
    no_session = health.get("no_session", 0)
    lines.append(f"| Optimize health — ok | {ok_val} |")
    lines.append(f"| Optimize health — warning | {warning_val} |")
    lines.append(f"| Optimize health — no-session | {no_session} |")


def _append_verify_summary(lines: list[str], summary: dict[str, Any]) -> None:
    verify = summary.get("verify", {}) or {}
    passed = verify.get("passed", 0)
    failed = verify.get("failed", 0)
    skipped = verify.get("skipped", 0)
    lines.append(f"| Verify — passed | {passed} |")
    lines.append(f"| Verify — failed | {failed} |")
    lines.append(f"| Verify — skipped | {skipped} |")


def _append_check_summary(lines: list[str], summary: dict[str, Any]) -> None:
    check = summary.get("check", {}) or {}
    passed = check.get("passed", 0)
    failed = check.get("failed", 0)
    skipped = check.get("skipped", 0)
    lines.append(f"| Check — passed | {passed} |")
    lines.append(f"| Check — failed | {failed} |")
    lines.append(f"| Check — skipped | {skipped} |")


def _append_workspace_detail(lines: list[str], ws: dict[str, Any]) -> None:
    name = ws.get("workspace", "")
    lines.append(f"### {name}")
    lines.append("")

    # check details
    check = ws.get("check", {}) or {}
    checks_list: list[dict[str, Any]] = check.get("checks", []) or []
    if checks_list:
        lines.append("**Check results:**")
        lines.append("")
        for c in checks_list:
            cid = c.get("id", "")
            cname = c.get("name", "")
            result = c.get("result", "")
            detail = c.get("detail")
            status_icon = "PASS" if result == "pass" else "FAIL"
            lines.append(f"- **{cid}** ({cname}): {status_icon}")
            if detail:
                lines.append(f"  - {detail}")
        lines.append("")

    # pattern summary
    pattern = ws.get("pattern", {}) or {}
    given = pattern.get("given", []) or []
    new = pattern.get("new", []) or []
    extended = pattern.get("extended", []) or []
    if given or new or extended:
        lines.append("**Patterns used:**")
        lines.append("")
        if given:
            lines.append("- Given:")
            for k in given:
                kname = k.get("name", "")
                kev = k.get("evidence", "")
                krnds = k.get("rounds", [])
                krnds_str = ", ".join(f"round-{n}" for n in krnds)
                lines.append(f"  - {kname} [{krnds_str}] ({kev})")
        if new:
            lines.append("- New:")
            for n in new:
                nname = n.get("name", "")
                nrnds = n.get("rounds", [])
                nrnds_str = ", ".join(f"round-{n}" for n in nrnds)
                lines.append(f"  - {nname} [{nrnds_str}]")
        if extended:
            lines.append("- Extended:")
            for e in extended:
                ename = e.get("name", "")
                ernds = e.get("rounds", [])
                efrom = e.get("from", "")
                ernds_str = ", ".join(f"round-{n}" for n in ernds)
                lines.append(f"  - {ename} [{ernds_str}] (from: {efrom})")
        lines.append("")


def _format_speedup(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}x"
    except (ValueError, TypeError):
        return "-"


def render_post_batch_report_file(
    state_path: Path,
    output_path: Path | None = None,
) -> Path:
    """Read post-batch-state.json and write post-batch-report.md."""
    import json
    data = json.loads(state_path.read_text(encoding="utf-8"))
    target = output_path or (state_path.parent / "post-batch-report.md")
    md = render_post_batch_report(data)
    target.write_text(md, encoding="utf-8")
    return target


__all__ = [
    "render_post_batch_report",
    "render_post_batch_report_file",
]
