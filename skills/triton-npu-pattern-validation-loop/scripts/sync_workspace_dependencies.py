#!/usr/bin/env python3
"""Inject repo sys.path, run import smoke, optionally copy fla.* into deps/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from batch_layout import list_active_validation_workspaces
from workspace_deps import sync_workspace_dependencies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inject repo sys.path into the operator (default), run import smoke, "
            "and fall back to deps/ copy only when smoke fails or --copy-deps is set."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workspace", help="Single workspace directory.")
    group.add_argument("--batch-root", help="Batch root with active workspaces.")
    parser.add_argument("--repo", required=True, help="Target repository root.")
    parser.add_argument(
        "--copy-deps",
        action="store_true",
        help="Skip repo path injection and copy fla.* modules into deps/ (isolated snapshot).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON (single workspace or list for batch).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).expanduser().resolve()
    if not repo_root.is_dir():
        print(f"sync_workspace_dependencies: repo is not a directory: {repo_root}", file=sys.stderr)
        return 2

    if args.workspace:
        workspaces = [Path(args.workspace).expanduser().resolve()]
    else:
        batch_root = Path(args.batch_root).expanduser().resolve()
        workspaces = list_active_validation_workspaces(batch_root)

    if not workspaces:
        print("sync_workspace_dependencies: no workspaces to sync", file=sys.stderr)
        return 2

    reports: list[dict[str, object]] = []
    for workspace in workspaces:
        try:
            reports.append(
                sync_workspace_dependencies(
                    workspace,
                    repo_root,
                    force_deps_copy=args.copy_deps,
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"sync_workspace_dependencies: {workspace.name}: {exc}", file=sys.stderr)
            return 1

    if args.json:
        payload = reports[0] if len(reports) == 1 else {"workspaces": reports}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        exit_code = 0
        for report in reports:
            strategy = report.get("dependency_strategy", "repo_path")
            smoke = report.get("import_smoke_passed", False)
            copied = report.get("copied_dependencies", [])
            print(
                f"{report['workspace']}: strategy={strategy} import_smoke={smoke} "
                f"deps_files={len(copied)}",
            )
            for warning in report.get("warnings", []):
                print(f"  warning: {warning}", file=sys.stderr)
            if not smoke:
                print(
                    f"  import smoke error: {report.get('import_smoke_error')}",
                    file=sys.stderr,
                )
                exit_code = 1
        return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
