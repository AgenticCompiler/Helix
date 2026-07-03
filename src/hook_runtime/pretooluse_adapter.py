from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from hook_runtime.tool_use_decision import deny_reason_for_tool_use


def build_denial_output(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], data)


def run_policy_file_wrapper(
    *,
    argv: list[str] | None,
    failure_prefix: str,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args(argv)

    try:
        policy = load_json_object(Path(args.policy))
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hooks must fail open
        _print_fail_open(failure_prefix, exc)
        return 0

    return run_with_policy(
        policy=policy,
        payload=payload,
        failure_prefix=failure_prefix,
    )


def run_with_policy(
    *,
    policy: dict[str, Any],
    payload: Any,
    failure_prefix: str,
) -> int:
    try:
        if not isinstance(payload, dict):
            raise ValueError("expected JSON object payload on stdin")
        reason = deny_reason_for_tool_use(policy, cast(dict[str, Any], payload))
    except Exception as exc:  # noqa: BLE001 - hooks must fail open
        _print_fail_open(failure_prefix, exc)
        return 0

    if reason is not None:
        json.dump(build_denial_output(reason), sys.stdout)
    return 0


def _print_fail_open(prefix: str, exc: Exception) -> None:
    print(f"{prefix} failed open: {exc}", file=sys.stderr)
