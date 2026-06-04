from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from triton_agent.skill_loader import load_skill_script_module

_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"
_SYNC_SCRIPT = "sync_workspace_dependencies"
_BATCH_LAYOUT_SCRIPT = "batch_layout"


def sync_batch_workspace_dependencies(
    batch_root: Path,
    repo_root: Path,
    *,
    stream: TextIO | None = None,
) -> int:
    """Inject repo paths and run import smoke for each active workspace (deps/ fallback on failure)."""
    batch_root = batch_root.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    if not batch_root.is_dir():
        print(
            f"[pattern-validation-sync-deps] batch root is not a directory: {batch_root}",
            file=sys.stderr,
        )
        return 2
    if not repo_root.is_dir():
        print(
            f"[pattern-validation-sync-deps] repo is not a directory: {repo_root}",
            file=sys.stderr,
        )
        return 2

    layout = load_skill_script_module(_PATTERN_VALIDATION_SKILL, _BATCH_LAYOUT_SCRIPT)
    sync = load_skill_script_module(_PATTERN_VALIDATION_SKILL, _SYNC_SCRIPT)
    workspaces = layout.list_active_validation_workspaces(batch_root)
    if not workspaces:
        return 0

    out = stream or sys.stderr
    failures = 0
    for workspace in workspaces:
        code = int(
            sync.main(
                [
                    "--workspace",
                    workspace.as_posix(),
                    "--repo",
                    repo_root.as_posix(),
                ],
            ),
        )
        if code != 0:
            failures += 1
            continue
        out.write(f"[pattern-validation-sync-deps] synced {workspace.name}\n")
    return 1 if failures else 0
