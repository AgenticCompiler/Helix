#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import bootstrap_runtime_state, resolve_workspace


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"triton-agent claude plugin SessionStart failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    try:
        result = bootstrap_runtime_state(workspace)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"triton-agent claude plugin SessionStart failed open: {exc}", file=sys.stderr)
        return 0
    if not result.additional_context:
        return 0
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": result.additional_context,
            }
        },
        sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
