"""Render log_check_result.md and pattern_analysis.md from structured JSON."""

from __future__ import annotations

from typing import Any


def render_log_check_markdown(data: dict[str, Any]) -> str:
    """Render log_check_result.md from log_check_result.json data."""
    overall = data.get("overall", "FAIL")
    failed_checks = data.get("failed_checks", "")
    overview_detail = data.get("overview_detail", "")
    checks: list[dict[str, Any]] = data.get("checks", [])

    lines: list[str] = []
    lines.append("summary:")
    lines.append(f"overall: {overall}")
    if overall == "PASS":
        lines.append("failed_checks: none")
    else:
        lines.append(f"failed_checks: {failed_checks or 'none'}")
    lines.append(f"overview_detail: {overview_detail}")
    lines.append("")

    for check in checks:
        cid = check.get("id", "")
        name = check.get("name", "")
        result = check.get("result", "fail").upper()
        detail = check.get("detail") or ""
        lines.append(f"{cid}: {name}")
        lines.append(f"result: {result}")
        lines.append(f"detail: {detail}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_pattern_analysis_markdown(data: dict[str, Any]) -> str:
    """Render pattern_analysis.md from pattern_analysis.json data."""
    rounds: list[dict[str, Any]] = data.get("rounds", [])
    summary: dict[str, Any] = data.get("summary", {})

    lines: list[str] = []
    lines.append("# Pattern Analysis")
    lines.append("")

    lines.append("## Per-Round Breakdown")
    lines.append("")
    for r in rounds:
        round_name = r.get("round", "unknown")
        patterns: list[dict[str, Any]] = r.get("patterns", [])
        lines.append(f"### {round_name}")
        lines.append("")
        if not patterns:
            lines.append("- No patterns detected")
            lines.append("")
            continue
        for p in patterns:
            name = p.get("name", "")
            evidence = p.get("evidence", "")
            source = p.get("source", "")
            lines.append(f"- **{name}** (evidence: {evidence})")
            if source:
                lines.append(f"  Source: {source}")
        lines.append("")

    lines.append("## Summary")
    lines.append("")

    given: list[dict[str, Any]] = summary.get("given", [])
    new: list[dict[str, Any]] = summary.get("new", [])
    extended: list[dict[str, Any]] = summary.get("extended", [])

    lines.append("### Given Patterns (matched staged references)")
    lines.append("")
    if given:
        for p in given:
            name = p.get("name", "")
            ev = p.get("evidence", "")
            rnds = p.get("rounds", [])
            rnds_str = ", ".join(f"round-{n}" for n in rnds)
            lines.append(f"- **{name}**: rounds [{rnds_str}], evidence: {ev}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("### New Strategies (not matching any staged reference)")
    lines.append("")
    if new:
        for n in new:
            name = n.get("name", "")
            rnds = n.get("rounds", [])
            rnds_str = ", ".join(f"round-{n}" for n in rnds)
            lines.append(f"- **{name}**: rounds [{rnds_str}]")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("### Extended Patterns (incremental enhancement of known patterns)")
    lines.append("")
    if extended:
        for e in extended:
            name = e.get("name", "")
            rnds = e.get("rounds", [])
            base = e.get("from", "")
            rnds_str = ", ".join(f"round-{n}" for n in rnds)
            lines.append(f"- **{name}**: rounds [{rnds_str}], extends: {base}")
    else:
        lines.append("- none")
    lines.append("")

    # Evidence distribution
    all_explicit = sum(
        1 for p in given if p.get("evidence") == "explicit"
    )
    all_inferred = sum(
        1 for p in given if p.get("evidence") == "inferred"
    )
    lines.append("### Evidence Distribution")
    lines.append("")
    lines.append(f"- explicit: {all_explicit}")
    lines.append(f"- inferred: {all_inferred}")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "render_log_check_markdown",
    "render_pattern_analysis_markdown",
]
