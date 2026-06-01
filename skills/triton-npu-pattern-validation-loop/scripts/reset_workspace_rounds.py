#!/usr/bin/env python3
"""Remove optimize round artifacts but keep baseline and operator harness files."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from batch_layout import list_active_validation_workspaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reset optimize rounds for validation workspaces.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--workspace")
    group.add_argument("--batch-root")
    args = parser.parse_args(argv)

    if args.workspace:
        workspaces = [Path(args.workspace).expanduser().resolve()]
    else:
        workspaces = list_active_validation_workspaces(Path(args.batch_root))

    for workspace in workspaces:
        reset_workspace(workspace)
        print(workspace.as_posix())
    return 0


def reset_workspace(workspace: Path) -> None:
    for path in workspace.glob("opt-round-*"):
        if path.is_dir():
            shutil.rmtree(path)
    for name in ("opt-note.md", "learned_lessons.md"):
        target = workspace / name
        if target.exists():
            target.unlink()
    for path in workspace.glob("opt_*.py"):
        if path.is_file():
            path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
