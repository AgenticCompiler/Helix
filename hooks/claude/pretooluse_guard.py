#!/usr/bin/env python3
"""
Claude Code PreToolUse hook wrapper for triton-agent optimize runs.

This wrapper adapts Claude Code hook stdin/stdout handling to the shared
backend-agnostic guard policy module.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, cast


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args(argv)

    try:
        policy = _load_json(Path(args.policy))
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - Hooks should fail open.
        print(f"triton-agent claude hook failed open: {exc}", file=sys.stderr)
        return 0

    try:
        reason = _deny_reason_for_tool_use(policy, payload)
    except Exception as exc:  # noqa: BLE001 - Hooks should fail open.
        print(f"triton-agent claude hook failed open: {exc}", file=sys.stderr)
        return 0

    if reason is None:
        return 0

    json.dump(_build_denial_output(reason), sys.stdout)
    return 0


def _build_denial_output(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


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
