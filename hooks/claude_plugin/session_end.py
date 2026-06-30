#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import cleanup_runtime_tree, resolve_workspace, should_manage_payload


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"triton-agent claude plugin SessionEnd failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict) or not should_manage_payload(payload):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    try:
        cleanup_runtime_tree(workspace / ".triton-agent")
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"triton-agent claude plugin SessionEnd failed open: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
