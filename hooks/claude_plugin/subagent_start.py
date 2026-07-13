#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import (
    bootstrap_runtime_state,
    is_optimize_subagent_payload,
    record_runtime_owner,
    resolve_agent_type,
    resolve_workspace,
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SubagentStart failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict) or not is_optimize_subagent_payload(payload):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    agent_id = payload.get("agent_id")
    agent_type = resolve_agent_type(payload)
    if not isinstance(agent_id, str) or not agent_id or agent_type is None:
        return 0
    try:
        result = bootstrap_runtime_state(workspace)
        record_runtime_owner(workspace / ".helix", agent_id=agent_id, agent_type=agent_type)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SubagentStart failed open: {exc}", file=sys.stderr)
        return 0
    if not result.additional_context:
        return 0
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SubagentStart",
                "additionalContext": result.additional_context,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
