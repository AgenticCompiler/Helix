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
READ_TOOL_PATH_KEYS = ("file_path", "filePath")
SHELL_WRAPPER_FLAGS = {"-c", "-lc"}
SHELL_WRAPPERS = {"bash", "sh", "zsh"}
PYTHON_COMMANDS = {"python", "python3"}
SHELL_CONTROL_OPERATORS = {"&&", "||", "|", ";", "&"}
PROTECTED_RELATIVE_PATH_PREFIXES = ("triton-agent-logs/",)

PATH_FRAGMENT_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_./-])(?P<path>(?:/|\.\.?/|\.codex/|triton-agent-logs/)[A-Za-z0-9_./*?{}+@%:,=-]+)"
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
    if not isinstance(tool_input, dict):
        return None

    workspace_root = _resolve_policy_path(policy.get("workspace_root"))
    if workspace_root is None:
        return None

    cwd = _resolve_cwd(tool_input.get("cwd") or payload.get("cwd"), workspace_root)
    allow_roots = _allow_roots(policy, workspace_root)
    deny_globs = [str(item) for item in policy.get("deny_read_globs", []) if isinstance(item, str)]
    deny_message = str(policy.get("deny_message") or "This read is blocked by workspace policy.")

    if tool_name == "Read":
        candidate = _read_tool_path(tool_input)
        if candidate is None:
            return None
        return _evaluate_candidate(candidate, cwd, workspace_root, allow_roots, deny_globs, deny_message)

    if tool_name != "Bash":
        return None

    command = tool_input.get("command")
    if not isinstance(command, str):
        return None

    for candidate, allow_protected_script_entrypoint in _candidate_paths(command):
        reason = _evaluate_candidate(
            candidate,
            cwd,
            workspace_root,
            allow_roots,
            deny_globs,
            deny_message,
            allow_protected_script_entrypoint=allow_protected_script_entrypoint,
        )
        if reason is not None:
            return reason

    return None


def _evaluate_candidate(
    candidate: str,
    cwd: Path,
    workspace_root: Path,
    allow_roots: list[Path],
    deny_globs: list[str],
    deny_message: str,
    *,
    allow_protected_script_entrypoint: bool = False,
) -> str | None:
    resolved = _resolve_candidate(candidate, cwd, workspace_root)
    if resolved is None:
        return None
    if not _is_under_any_root(resolved, allow_roots):
        return deny_message
    if allow_protected_script_entrypoint and _is_protected_script_path(resolved, workspace_root):
        return None
    if _matches_any_glob(resolved, deny_globs):
        return deny_message
    return None


def _read_tool_path(tool_input: dict[str, Any]) -> str | None:
    for key in READ_TOOL_PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _candidate_paths(command: str) -> list[tuple[str, bool]]:
    return _candidate_paths_inner(command, seen_commands=set())


def _candidate_paths_inner(command: str, *, seen_commands: set[str]) -> list[tuple[str, bool]]:
    if command in seen_commands:
        return []

    tokens = _split_command(command)
    candidates: list[tuple[str, bool]] = []
    next_seen_commands = seen_commands | {command}
    for nested_command in _shell_wrapper_commands(tokens):
        candidates.extend(_candidate_paths_inner(nested_command, seen_commands=next_seen_commands))

    if not _contains_read_command(tokens):
        return candidates

    python_entrypoint_indexes = _python_entrypoint_candidate_indexes(tokens)
    python_entrypoint_values = {tokens[index] for index in python_entrypoint_indexes}
    explicit_path_tokens = {token for token in tokens if _looks_like_path(token)}

    for index, token in enumerate(tokens):
        if _is_read_command_token(token):
            continue
        if _looks_like_path(token):
            candidates.append((token, index in python_entrypoint_indexes))

    for match in PATH_FRAGMENT_RE.finditer(command):
        path = match.group("path")
        if (
            not _is_read_command_token(path)
            and path not in python_entrypoint_values
            and not _is_nested_path_fragment(path, explicit_path_tokens)
        ):
            candidates.append((path, False))

    return candidates


def _shell_wrapper_commands(tokens: list[str]) -> list[str]:
    commands: list[str] = []
    for index, token in enumerate(tokens):
        if Path(token).name not in SHELL_WRAPPERS:
            continue
        if index + 2 >= len(tokens):
            continue
        if tokens[index + 1] not in SHELL_WRAPPER_FLAGS:
            continue
        commands.append(tokens[index + 2])
    return commands


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _contains_read_command(tokens: list[str]) -> bool:
    return any(_is_read_command_token(token) for token in tokens)


def _is_read_command_token(token: str) -> bool:
    return Path(token).name in READ_COMMANDS


def _is_python_command_token(token: str) -> bool:
    return Path(token).name in PYTHON_COMMANDS


def _python_entrypoint_candidate_indexes(tokens: list[str]) -> set[int]:
    indexes: set[int] = set()
    for index, token in enumerate(tokens):
        if not _is_python_command_token(token):
            continue
        entrypoint_index = _python_entrypoint_candidate_index(tokens, index + 1)
        if entrypoint_index is not None:
            indexes.add(entrypoint_index)
    return indexes


def _python_entrypoint_candidate_index(tokens: list[str], start: int) -> int | None:
    for index in range(start, len(tokens)):
        token = tokens[index]
        if token in SHELL_CONTROL_OPERATORS:
            return None
        if token in {"-c", "-m"}:
            return None
        if token == "--":
            continue
        if token.startswith("-"):
            continue
        if _looks_like_path(token):
            return index
        return None
    return None


def _looks_like_path(token: str) -> bool:
    return (
        token.startswith("/")
        or token.startswith("./")
        or token.startswith("../")
        or token.startswith(".codex/")
        or token.startswith(PROTECTED_RELATIVE_PATH_PREFIXES)
    )


def _is_nested_path_fragment(candidate: str, explicit_path_tokens: set[str]) -> bool:
    return any(candidate != token and candidate in token for token in explicit_path_tokens)


def _is_protected_script_path(path: Path, workspace_root: Path) -> bool:
    try:
        relative = path.relative_to(workspace_root)
    except ValueError:
        return False
    parts = relative.parts
    return len(parts) >= 5 and parts[0] == ".codex" and parts[1] == "skills" and parts[3] == "scripts"

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
