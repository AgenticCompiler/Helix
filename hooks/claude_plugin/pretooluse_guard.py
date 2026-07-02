#!/usr/bin/env python3
"""
Claude Code PreToolUse hook wrapper for the standalone optimize plugin.

This variant computes its workspace policy dynamically from the active cwd
instead of relying on a runner-generated policy.json file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from state_bootstrap import (
    compiler_source_read_root,
    missing_state_denial_reason,
    resolve_workspace,
    should_manage_payload,
)


def _bootstrap_support_import() -> None:
    current_dir = Path(__file__).resolve().parent
    candidates = (
        current_dir.parent.parent / "src",
        current_dir.parent,
        current_dir,
    )
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.is_dir() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


_bootstrap_support_import()

from hook_runtime.pretooluse_adapter import build_denial_output, run_with_policy  # noqa: E402


_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected optimize runtime files or `triton-agent-logs/` output. "
    "Use the skill instructions and documented command interface instead."
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"triton-agent claude plugin PreToolUse failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict) or not should_manage_payload(payload):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0

    tool_name = payload.get("tool_name")
    if tool_name in {"Edit", "MultiEdit", "Write"}:
        state_reason = missing_state_denial_reason(workspace)
        if state_reason is not None:
            json.dump(build_denial_output(state_reason), sys.stdout)
            return 0

    return run_with_policy(
        policy=_policy(workspace),
        payload=payload,
        failure_prefix="triton-agent claude plugin PreToolUse",
    )


def _policy(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    allow_read_roots = [str(root)]
    compiler_source = compiler_source_read_root()
    if compiler_source is not None:
        allow_read_roots.append(str(compiler_source.resolve()))
    return {
        "workspace_root": str(root),
        "allow_read_roots": allow_read_roots,
        "deny_read_globs": [
            str(root / ".triton-agent"),
            str(root / ".triton-agent" / "**"),
            str(root / "triton-agent-logs" / "**"),
        ],
        "deny_message": _DENY_MESSAGE,
    }


if __name__ == "__main__":
    raise SystemExit(main())
