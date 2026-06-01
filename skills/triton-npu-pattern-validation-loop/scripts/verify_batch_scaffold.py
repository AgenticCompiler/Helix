#!/usr/bin/env python3
"""Verify pattern-validation workspaces before optimize-batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from batch_layout import list_active_validation_workspaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check that validation workspaces follow workspace-scaffold-contract Step 2b "
            "before running optimize-batch."
        ),
    )
    parser.add_argument("--batch-root", required=True, help="Batch root directory.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report.",
    )
    args = parser.parse_args(argv)

    batch_root = Path(args.batch_root).expanduser().resolve()
    workspaces = list_active_validation_workspaces(batch_root)
    if not workspaces:
        print(f"verify_batch_scaffold: no active workspaces under {batch_root}", file=sys.stderr)
        return 1

    metas = [_load_meta(path) for path in workspaces]
    shared_sources = _shared_source_paths(metas)
    reports = [
        verify_workspace(workspace, meta, shared_sources=shared_sources)
        for workspace, meta in zip(workspaces, metas, strict=True)
    ]
    failed = [report for report in reports if report["issues"]]

    payload = {
        "batch_root": batch_root.as_posix(),
        "workspace_count": len(reports),
        "failed_count": len(failed),
        "reports": reports,
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        for report in reports:
            print(_render_report(report))
        print(
            f"Summary: {len(reports) - len(failed)} ok, {len(failed)} failed scaffold checks",
        )

    return 1 if failed else 0


def verify_workspace(
    workspace: Path,
    meta: dict[str, Any],
    *,
    shared_sources: dict[str, list[str]],
) -> dict[str, Any]:
    issues: list[str] = []
    operator_name = str(meta.get("operator_filename", "")).strip()
    if not operator_name:
        issues.append("validation-meta.json missing operator_filename")
        operator_path = None
        operator_text = ""
    else:
        operator_path = workspace / operator_name
        if not operator_path.is_file():
            issues.append(f"operator file not found: {operator_name}")
            operator_text = ""
        else:
            operator_text = operator_path.read_text(encoding="utf-8")

    source_path = str(meta.get("source_path", "")).strip()
    validation_target = str(meta.get("validation_target", "")).strip()
    split_from = str(meta.get("split_from", "")).strip()
    shared_source = source_path and len(shared_sources.get(source_path, [])) > 1

    if shared_source or split_from:
        if not validation_target:
            issues.append(
                "validation_target is required when split_from is set or multiple workspaces "
                f"share source_path {source_path!r} (Step 2b manual split)"
            )
        if shared_source and not split_from:
            issues.append(
                f"split_from is required when multiple workspaces share source_path {source_path!r}"
            )
        if shared_source and not meta.get("included_symbols"):
            issues.append(
                "included_symbols is required when multiple workspaces share the same source_path"
            )

    if validation_target and operator_text and validation_target not in operator_text:
        issues.append(
            f"validation_target {validation_target!r} not found in operator file; "
            "operator may be an un-split whole-file copy"
        )

    for symbol in _string_list(meta.get("included_symbols")):
        if operator_text and symbol not in operator_text:
            issues.append(f"included_symbols entry {symbol!r} missing from operator file")

    for symbol in _string_list(meta.get("excluded_targets")):
        if operator_text and symbol in operator_text:
            issues.append(
                f"excluded_targets entry {symbol!r} still present in operator file; "
                "Step 2b extract likely incomplete"
            )

    return {
        "workspace": workspace.name,
        "source_path": source_path or None,
        "validation_target": validation_target or None,
        "split_from": split_from or None,
        "shared_source_path": shared_source,
        "issues": issues,
        "passed": not issues,
    }


def _load_meta(workspace: Path) -> dict[str, Any]:
    meta_path = workspace / "validation-meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"verify_batch_scaffold: missing validation-meta.json: {workspace}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"verify_batch_scaffold: invalid validation-meta.json: {meta_path}")
    return payload


def _shared_source_paths(metas: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for meta in metas:
        source_path = str(meta.get("source_path", "")).strip()
        workspace = str(meta.get("workspace", "")).strip()
        if not source_path or not workspace:
            continue
        grouped.setdefault(source_path, []).append(workspace)
    return grouped


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _render_report(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [f"[{status}] {report['workspace']}"]
    if report.get("source_path"):
        lines.append(f"  source_path: {report['source_path']}")
    if report.get("validation_target"):
        lines.append(f"  validation_target: {report['validation_target']}")
    if report.get("shared_source_path"):
        lines.append("  shared_source_path: yes")
    for issue in report["issues"]:
        lines.append(f"  issue: {issue}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
