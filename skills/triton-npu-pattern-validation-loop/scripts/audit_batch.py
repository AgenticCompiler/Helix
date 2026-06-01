#!/usr/bin/env python3
"""Audit optimize workspaces against validation-meta expected patterns."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from batch_layout import (
    archive_passed_workspaces,
    list_active_validation_workspaces,
    list_completed_validation_workspaces,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether optimize rounds cited expected pattern IDs.",
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
        "--archive-passed",
        action="store_true",
        help="Move passed active workspaces into batch-root/_completed/.",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="With --batch-root, also audit workspaces under _completed/.",
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

    reports = [audit_workspace(path) for path in workspaces]
    archived: list[str] = []
    if args.archive_passed:
        if batch_root is None:
            print("audit_batch: --archive-passed requires --batch-root", file=sys.stderr)
            return 2
        moved = archive_passed_workspaces(reports, batch_root=batch_root)
        archived = [path.as_posix() for path in moved]

    payload: dict[str, Any] = {
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

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(render_report(report))
        for path in archived:
            print(f"archived: {path}")
        if batch_root is not None:
            remaining = payload["active_remaining"]
            print(
                f"active workspaces remaining: {len(remaining)}; "
                f"completed total: {payload['completed_total']}",
            )

    active_reports = [
        report
        for report in reports
        if batch_root is None or report["location"] == "active"
    ]
    failed = any(not report["passed"] for report in active_reports)
    return 1 if failed else 0


def audit_workspace(workspace: Path) -> dict[str, object]:
    meta_path = workspace / "validation-meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = [str(item) for item in meta.get("expected_patterns", [])]
    round_dirs = sorted(workspace.glob("opt-round-*"))
    corpus_parts: list[str] = []
    for round_dir in round_dirs:
        for name in ("attempts.md", "summary.md"):
            path = round_dir / name
            if path.is_file():
                corpus_parts.append(path.read_text(encoding="utf-8"))
    corpus = "\n".join(corpus_parts).lower()

    matched = [pattern for pattern in expected if pattern.lower() in corpus]
    missing = [pattern for pattern in expected if pattern not in matched]
    location = "completed" if workspace.parent.name == "_completed" else "active"
    return {
        "workspace": workspace.name,
        "location": location,
        "expected_patterns": expected,
        "matched_patterns": matched,
        "missing_patterns": missing,
        "round_count": len(round_dirs),
        "has_baseline": (workspace / "baseline").is_dir(),
        "passed": not missing and bool(round_dirs),
    }


def render_report(report: dict[str, object]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    location = report.get("location", "active")
    lines = [
        f"[{status}] {report['workspace']} ({location})",
        f"  rounds: {report['round_count']}  baseline: {report['has_baseline']}",
        f"  expected: {', '.join(str(x) for x in report['expected_patterns']) or '(none)'}",
        f"  matched:  {', '.join(str(x) for x in report['matched_patterns']) or '(none)'}",
    ]
    missing = report["missing_patterns"]
    if missing:
        lines.append(f"  missing:  {', '.join(str(x) for x in missing)}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
