from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO

from triton_agent.skill_loader import load_skill_script_module

_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"
_VERIFY_SCRIPT = "verify_batch_scaffold"


def run_pattern_validation_verify(
    batch_root: Path,
    *,
    json_output: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Run workspace scaffold verification for a pattern-validation batch root."""
    batch_root = batch_root.expanduser().resolve()
    if not batch_root.is_dir():
        print(
            f"[pattern-validation-verify] batch root is not a directory: {batch_root}",
            file=sys.stderr,
        )
        return 2

    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, _VERIFY_SCRIPT)
    argv = ["--batch-root", batch_root.as_posix()]
    if json_output:
        argv.append("--json")
        return int(module.main(argv))

    payload = _verify_batch_payload(module, batch_root)
    out = stream or sys.stdout
    for report in payload["reports"]:
        out.write(_render_verify_report(report))
        out.write("\n")
    out.write(
        f"Summary: {payload['workspace_count'] - payload['failed_count']} ok, "
        f"{payload['failed_count']} failed scaffold checks\n",
    )
    return 1 if payload["failed_count"] else 0


def _verify_batch_payload(module: Any, batch_root: Path) -> dict[str, Any]:
    workspaces = module.list_active_validation_workspaces(batch_root)
    if not workspaces:
        return {
            "batch_root": batch_root.as_posix(),
            "workspace_count": 0,
            "failed_count": 1,
            "reports": [],
        }

    metas = [module._load_meta(path) for path in workspaces]
    shared_sources = module._shared_source_paths(metas)
    reports = [
        module.verify_workspace(workspace, meta, shared_sources=shared_sources)
        for workspace, meta in zip(workspaces, metas, strict=True)
    ]
    failed = [report for report in reports if report["issues"]]
    return {
        "batch_root": batch_root.as_posix(),
        "workspace_count": len(reports),
        "failed_count": len(failed),
        "reports": reports,
    }


def _render_verify_report(report: dict[str, Any]) -> str:
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
