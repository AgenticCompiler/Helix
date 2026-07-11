#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import cleanup_runtime_tree, resolve_workspace, should_cleanup_for_subagent


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SubagentStop failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    runtime_dir = workspace / ".helix"
    try:
        if should_cleanup_for_subagent(payload, runtime_dir):
            cleanup_runtime_tree(runtime_dir)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SubagentStop failed open: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
