#!/usr/bin/env python3
"""Collect optimize-round evidence for pattern-validation review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from batch_evaluation import resolve_workspace_meta
from batch_layout import (
    archive_passed_workspaces,
    list_active_validation_workspaces,
    list_completed_validation_workspaces,
)

_ARTIFACT_NAMES = ("attempts.md", "summary.md", "opt-note.md")
_EXCERPT_LIMIT = 8000
_HEURISTIC_NOTE = (
    "Heuristic substring match on attempts.md and summary.md only. "
    "The orchestrating agent must confirm mechanism alignment with synthesis."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect per-workspace optimize evidence (round artifacts, pattern-id hints) "
            "for agent review. Exit code reflects collection errors only, not validation pass/fail."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workspace", help="Single operator workspace directory.")
    group.add_argument("--batch-root", help="Batch root containing child workspaces.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report.",
    )
    parser.add_argument(
        "--output",
        help="Write JSON report to this path (implies structured output).",
    )
    parser.add_argument(
        "--archive-passed",
        action="store_true",
        help=(
            "Move workspaces with heuristic_suggested_pass=true into batch-root/_completed/. "
            "Use only after agent review confirms pass."
        ),
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="With --batch-root, also include workspaces under _completed/.",
    )
    args = parser.parse_args(argv)

    if args.workspace:
        workspaces = [Path(args.workspace).expanduser().resolve()]
        batch_root: Path | None = None
    else:
        batch_root = Path(args.batch_root).expanduser().resolve()
        workspaces = list_active_validation_workspaces(batch_root)
        if args.include_completed:
            workspaces.extend(list_completed_validation_workspaces(batch_root))

    if not workspaces:
        print("audit_batch: no workspaces to collect", file=sys.stderr)
        return 2

    reports = [collect_workspace_evidence(path) for path in workspaces]
    archived: list[str] = []
    if args.archive_passed:
        if batch_root is None:
            print("audit_batch: --archive-passed requires --batch-root", file=sys.stderr)
            return 2
        moved = archive_passed_workspaces(_heuristic_reports_for_archive(reports), batch_root=batch_root)
        archived = [path.as_posix() for path in moved]

    payload: dict[str, Any] = {
        "schema_version": 1,
        "review_model": "agent",
        "heuristic_note": _HEURISTIC_NOTE,
        "reports": reports,
        "archived": archived,
        "active_remaining": [
            path.name for path in list_active_validation_workspaces(batch_root)
        ]
        if batch_root is not None
        else [],
        "completed_total": len(list_completed_validation_workspaces(batch_root))
        if batch_root is not None
        else 0,
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).expanduser().resolve().write_text(text, encoding="utf-8")
    if args.json or args.output:
        if not args.output:
            print(text, end="")
    else:
        for report in reports:
            print(render_evidence_report(report))
        for path in archived:
            print(f"archived: {path}")
        if batch_root is not None:
            remaining = payload["active_remaining"]
            print(
                f"active workspaces remaining: {len(remaining)}; "
                f"completed total: {payload['completed_total']}",
            )

    return 0


def collect_workspace_evidence(workspace: Path) -> dict[str, object]:
    meta = resolve_workspace_meta(workspace)
    expected = [str(item) for item in meta.get("expected_patterns", [])]
    round_dirs = sorted(workspace.glob("opt-round-*"))
    rounds: list[dict[str, object]] = []
    corpus_parts: list[str] = []
    for round_dir in round_dirs:
        artifacts: dict[str, object] = {}
        for name in _ARTIFACT_NAMES:
            path = round_dir / name
            if not path.is_file():
                artifacts[name] = {"path": path.as_posix(), "exists": False}
                continue
            text = path.read_text(encoding="utf-8")
            corpus_parts.append(text)
            artifacts[name] = {
                "path": path.as_posix(),
                "exists": True,
                "size_bytes": path.stat().st_size,
                "excerpt": _excerpt(text),
            }
        rounds.append({"round": round_dir.name, "artifacts": artifacts})

    corpus = "\n".join(corpus_parts).lower()
    matched = [pattern for pattern in expected if pattern.lower() in corpus]
    missing = [pattern for pattern in expected if pattern not in matched]
    location = "completed" if workspace.parent.name == "_completed" else "active"
    heuristic_suggested_pass = not missing and bool(round_dirs)
    return {
        "workspace": workspace.name,
        "location": location,
        "validation_meta": {
            "expected_patterns": expected,
            "validation_target": meta.get("validation_target"),
            "synthesis_refs": meta.get("synthesis_refs", []),
        },
        "round_count": len(round_dirs),
        "has_baseline": (workspace / "baseline").is_dir(),
        "rounds": rounds,
        "expected_patterns": expected,
        "heuristic_pattern_hits": matched,
        "heuristic_missing_patterns": missing,
        "heuristic_suggested_pass": heuristic_suggested_pass,
        "agent_review_required": True,
    }


def render_evidence_report(report: dict[str, object]) -> str:
    hint = "SUGGEST_PASS" if report["heuristic_suggested_pass"] else "NEEDS_REVIEW"
    location = report.get("location", "active")
    lines = [
        f"[{hint}] {report['workspace']} ({location})",
        f"  rounds: {report['round_count']}  baseline: {report['has_baseline']}",
        f"  expected: {', '.join(str(x) for x in report['expected_patterns']) or '(none)'}",
        f"  heuristic hits:  {', '.join(str(x) for x in report['heuristic_pattern_hits']) or '(none)'}",
    ]
    missing = report["heuristic_missing_patterns"]
    if missing:
        lines.append(f"  heuristic missing:  {', '.join(str(x) for x in missing)}")
    lines.append("  agent_review_required: yes")
    return "\n".join(lines)


def _heuristic_reports_for_archive(reports: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "workspace": report["workspace"],
            "location": report["location"],
            "passed": bool(report["heuristic_suggested_pass"]),
        }
        for report in reports
    ]


def _excerpt(text: str) -> str:
    if len(text) <= _EXCERPT_LIMIT:
        return text
    return text[:_EXCERPT_LIMIT] + "\n... [truncated]"


if __name__ == "__main__":
    raise SystemExit(main())
