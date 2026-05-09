#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


READ_COMMANDS = {
    "awk",
    "cat",
    "head",
    "less",
    "more",
    "python",
    "python3",
    "rg",
    "sed",
    "tail",
}

PATH_FRAGMENT_RE = re.compile(
    r"(?P<path>(?:/|\.\.?/|\.codex/)[A-Za-z0-9_./*?{}+@%:,=-]+)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args(argv)

    try:
        policy = _load_json(Path(args.policy))
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - Hooks should fail open.
        print(f"triton-agent codex hook failed open: {exc}", file=sys.stderr)
        return 0

    try:
        reason = evaluate_payload(policy, payload)
    except Exception as exc:  # noqa: BLE001 - Hooks should fail open.
        print(f"triton-agent codex hook failed open: {exc}", file=sys.stderr)
        return 0

    if reason is None:
        return 0

    json.dump(build_denial_output(reason), sys.stdout)
    return 0


def evaluate_payload(policy: dict[str, Any], payload: dict[str, Any]) -> str | None:
    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if tool_name != "Bash" or not isinstance(tool_input, dict):
        return None

    command = tool_input.get("command")
    if not isinstance(command, str):
        return None

    tokens = _split_command(command)
    if not _contains_read_command(tokens):
        return None

    workspace_root = _resolve_policy_path(policy.get("workspace_root"))
    if workspace_root is None:
        return None

    cwd = _resolve_cwd(tool_input.get("cwd"), workspace_root)
    allow_roots = _allow_roots(policy, workspace_root)
    deny_globs = [str(item) for item in policy.get("deny_read_globs", []) if isinstance(item, str)]
    deny_message = str(policy.get("deny_message") or "This read is blocked by workspace policy.")

    for candidate in _candidate_paths(command, tokens):
        resolved = _resolve_candidate(candidate, cwd, workspace_root)
        if resolved is None:
            continue
        if not _is_under_any_root(resolved, allow_roots):
            return deny_message
        if _matches_any_glob(resolved, deny_globs):
            return deny_message

    return None


def build_denial_output(reason: str) -> dict[str, Any]:
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


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _contains_read_command(tokens: list[str]) -> bool:
    return any(_is_read_command_token(token) for token in tokens)


def _is_read_command_token(token: str) -> bool:
    return Path(token).name in READ_COMMANDS


def _candidate_paths(command: str, tokens: list[str]) -> list[str]:
    candidates: list[str] = []
    for token in tokens:
        if _is_read_command_token(token):
            continue
        if _looks_like_path(token):
            candidates.append(token)

    for match in PATH_FRAGMENT_RE.finditer(command):
        path = match.group("path")
        if not _is_read_command_token(path):
            candidates.append(path)

    return candidates


def _looks_like_path(token: str) -> bool:
    return (
        token.startswith("/")
        or token.startswith("./")
        or token.startswith("../")
        or token.startswith(".codex/")
    )


def _resolve_policy_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    return Path(value).expanduser().resolve()


def _resolve_cwd(value: object, workspace_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        return workspace_root
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = workspace_root / path
    return path.resolve()


def _allow_roots(policy: dict[str, Any], workspace_root: Path) -> list[Path]:
    roots = [workspace_root]
    for raw_root in policy.get("allow_read_roots", []):
        root = _resolve_policy_path(raw_root)
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def _resolve_candidate(candidate: str, cwd: Path, workspace_root: Path) -> Path | None:
    if "*" in candidate or "?" in candidate or "{" in candidate or "}" in candidate:
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve()
    except OSError:
        return (workspace_root / candidate).resolve()


def _is_under_any_root(path: Path, roots: list[Path]) -> bool:
    return any(_is_relative_to(path, root) for root in roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _matches_any_glob(path: Path, patterns: list[str]) -> bool:
    raw_path = str(path)
    return any(fnmatch.fnmatch(raw_path, pattern) for pattern in patterns)


if __name__ == "__main__":
    raise SystemExit(main())
