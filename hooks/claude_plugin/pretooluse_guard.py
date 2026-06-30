#!/usr/bin/env python3
"""
Claude Code PreToolUse hook wrapper for the standalone optimize plugin.

This variant computes its workspace policy dynamically from the active cwd
instead of relying on a runner-generated policy.json file.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, cast

from state_bootstrap import missing_state_denial_reason, resolve_workspace, should_manage_payload


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
            json.dump(_build_denial_output(state_reason), sys.stdout)
            return 0

    try:
        reason = _deny_reason_for_tool_use(_policy(workspace), payload)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"triton-agent claude plugin PreToolUse failed open: {exc}", file=sys.stderr)
        return 0
    if reason is not None:
        json.dump(_build_denial_output(reason), sys.stdout)
    return 0


def _policy(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    return {
        "workspace_root": str(root),
        "allow_read_roots": [str(root)],
        "deny_read_globs": [
            str(root / ".triton-agent"),
            str(root / ".triton-agent" / "**"),
            str(root / "triton-agent-logs" / "**"),
        ],
        "deny_message": _DENY_MESSAGE,
    }


def _build_denial_output(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _deny_reason_for_tool_use(policy: dict[str, Any], payload: dict[str, Any]) -> str | None:
    module = _load_policy_module()
    deny_reason = cast(
        Callable[[dict[str, Any], dict[str, Any]], str | None] | None,
        getattr(module, "deny_reason_for_tool_use", None),
    )
    if not callable(deny_reason):
        raise RuntimeError("shared guard policy module does not export deny_reason_for_tool_use")
    return deny_reason(policy, payload)


def _load_policy_module() -> Any:
    current_dir = Path(__file__).resolve().parent
    candidates = [
        current_dir / "tool_use_guard_policy.py",
        current_dir.parent / "shared" / "tool_use_guard_policy.py",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        spec = importlib.util.spec_from_file_location("tool_use_guard_policy", candidate)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("unable to locate shared guard policy module")


if __name__ == "__main__":
    raise SystemExit(main())
