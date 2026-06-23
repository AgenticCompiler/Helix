#!/usr/bin/env python3
"""
Shared PreToolUse hook guard for triton-agent optimize runs.

This script is staged by the Codex, Claude backends when
optimize is launched with --enable-agent-hook.  It runs before every tool
invocation and enforces two layers of policy:

  Layer 1 — Read / Bash access control (policy.json deny_read_globs)
    Blocks the agent from inspecting staged skill implementation files
    and triton-agent-logs/ output.  Read-type shell commands (cat, head,
    rg, etc.) are also blocked.  Bash shell wrappers are unwrapped so
    that nested commands are inspected.

  Layer 2 — Native edit-tool phase enforcement (.triton-agent/state.json)
    Restricts built-in Write / Edit / MultiEdit tools based on the
    current optimize workflow phase:

      baseline              Only the source operator, root-level
                            test_*/bench_* harnesses, and baseline/
                            may be edited.

      awaiting_round_start  All native edits are blocked until
                            triton-npu-optimize-start-round opens a
                            new opt-round-N/.

      round_active          Only files inside the active opt-round-N/
                            directory may be edited.

    Each denial message includes a "first-version scope" disclaimer
    noting that Bash file writes are NOT intercepted — only the
    backend's native edit tools are guarded.

Design doc: docs/specs/2026-06-23-optimize-native-edit-hook-guard-design.md
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

# ── Shell commands that indicate a "read" intent ──────────────────────────
# When any of these appear in a Bash command, every path-like token in that
# command is treated as a candidate read path and checked against the policy.
READ_COMMANDS = {
    "awk",
    "cat",
    "head",
    "less",
    "more",
    "rg",
    "sed",
    "tail",
}
# Native edit tools that are subject to phase-based path restrictions (Layer 2).
# Bash-based file writes (e.g. `tee`, `dd`, redirections) are NOT intercepted.
EDIT_TOOLS = {"Edit", "MultiEdit", "Write"}
READ_TOOL_PATH_KEYS = ("file_path", "filePath")
EDIT_TOOL_PATH_KEYS = ("file_path", "path", "filePath", "notebook_path", "notebookPath")
SHELL_WRAPPER_FLAGS = {"-c", "-lc"}
SHELL_WRAPPERS = {"bash", "sh", "zsh"}
PROTECTED_RELATIVE_PATH_PREFIXES = ("triton-agent-logs/",)
WORKFLOW_STATE_RELATIVE_PATH = Path(".triton-agent") / "state.json"

PATH_FRAGMENT_RE = re.compile(
    r"(?:^|[^A-Za-z0-9_./-])(?P<path>(?:/|\.\.?/|\.codex/|\.claude/|\.opencode/|triton-agent-logs/)[A-Za-z0-9_./*?{}+@%:,=-]+)"
)
WINDOWS_PATH_FRAGMENT_RE = re.compile(
    r"(?P<path>[A-Za-z]:[\\/][A-Za-z0-9_ .\\/(){}+@%:,=-]+)"
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
    """Return a denial reason string, or None if the tool is allowed."""
    guard_policy = _guard_policy(policy)
    if guard_policy.get("enabled") is False:
        return None

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None

    workspace_root = _resolve_policy_path(policy.get("workspace_root"))
    if workspace_root is None:
        return None

    cwd = _resolve_cwd(tool_input.get("cwd") or payload.get("cwd"), workspace_root)
    allow_roots = _allow_roots(guard_policy, workspace_root)
    protected_script_roots = _protected_script_roots(guard_policy, workspace_root)
    deny_globs = [str(item) for item in guard_policy.get("deny_read_globs", []) if isinstance(item, str)]
    deny_message = str(guard_policy.get("deny_message") or "This read is blocked by workspace policy.")

    # Layer 2 (phase enforcement): built-in edit tools are checked against
    # .triton-agent/state.json before the read/Bash deny_globs layer.
    if tool_name in EDIT_TOOLS:
        candidate = _edit_tool_path(tool_input)
        if candidate is None:
            return None
        return _evaluate_built_in_edit_candidate(candidate, cwd, workspace_root)

    if tool_name == "Read":
        candidate = _read_tool_path(tool_input)
        if candidate is None:
            return None
        return _evaluate_candidate(
            candidate,
            cwd,
            workspace_root,
            allow_roots,
            protected_script_roots,
            deny_globs,
            deny_message,
        )

    if tool_name != "Bash":
        return None

    command = tool_input.get("command")
    if not isinstance(command, str):
        return None

    # Walk every path-like token in the command (including nested shell
    # wrappers and regex-detected fragments) and check each against the
    # same allow_roots / deny_globs / protected_script_roots policy.
    for candidate, allow_protected_script_entrypoint in _candidate_paths(command):
        reason = _evaluate_candidate(
            candidate,
            cwd,
            workspace_root,
            allow_roots,
            protected_script_roots,
            deny_globs,
            deny_message,
            allow_protected_script_entrypoint=allow_protected_script_entrypoint,
        )
        if reason is not None:
            return reason

    return None


def _guard_policy(policy: dict[str, Any]) -> dict[str, Any]:
    guard_policy = policy.get("guard")
    if isinstance(guard_policy, dict):
        return guard_policy
    return policy


def _evaluate_candidate(
    candidate: str,
    cwd: Path,
    workspace_root: Path,
    allow_roots: list[Path],
    protected_script_roots: list[Path],
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
    if allow_protected_script_entrypoint and _is_protected_script_path(resolved, protected_script_roots):
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


def _edit_tool_path(tool_input: dict[str, Any]) -> str | None:
    for key in EDIT_TOOL_PATH_KEYS:
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

    explicit_path_tokens = {token for token in tokens if _looks_like_path(token)}

    for index, token in enumerate(tokens):
        if _is_read_command_token(token):
            continue
        if _looks_like_path(token):
            candidates.append((token, False))

    for match in PATH_FRAGMENT_RE.finditer(command):
        path = match.group("path")
        if (
            not _is_read_command_token(path)
            and not _is_nested_path_fragment(path, explicit_path_tokens)
        ):
            candidates.append((path, False))
    for match in WINDOWS_PATH_FRAGMENT_RE.finditer(command):
        path = match.group("path").rstrip("'\"),")
        if (
            not _is_read_command_token(path)
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
        commands.append(tokens[index + 2].strip("\"'"))
    return commands


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return []


def _contains_read_command(tokens: list[str]) -> bool:
    return any(_is_read_command_token(token) for token in tokens)


def _is_read_command_token(token: str) -> bool:
    return Path(token).name in READ_COMMANDS


def _looks_like_path(token: str) -> bool:
    if token.startswith("-"):
        return False
    path = Path(token)
    return (
        path.is_absolute()
        or token.startswith("/")
        or token.startswith("./")
        or token.startswith("../")
        or token.startswith(".codex/")
        or token.startswith(".claude/")
        or token.startswith(".opencode/")
        or token.startswith(PROTECTED_RELATIVE_PATH_PREFIXES)
        or "\\" in token
        or path.suffix != ""
    )


def _is_nested_path_fragment(candidate: str, explicit_path_tokens: set[str]) -> bool:
    return any(candidate != token and candidate in token for token in explicit_path_tokens)


def _is_protected_script_path(path: Path, protected_script_roots: list[Path]) -> bool:
    for root in protected_script_roots:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        parts = relative.parts
        if len(parts) >= 3 and parts[1] == "scripts":
            return True
    return False


def build_denial_output(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _evaluate_built_in_edit_candidate(candidate: str, cwd: Path, workspace_root: Path) -> str | None:
    resolved = _resolve_candidate(candidate, cwd, workspace_root)
    if resolved is None:
        return None
    if not _is_relative_to(resolved, workspace_root):
        return _built_in_edit_outside_workspace_denial()

    # Load and inspect .triton-agent/state.json to determine the active
    # optimize workflow phase and enforce the corresponding path policy.
    state_path = workspace_root / WORKFLOW_STATE_RELATIVE_PATH
    try:
        state = _load_json(state_path)
        phase = _require_state_string(state, "phase")
        source_operator = _require_state_string(state, "source_operator")
    except (FileNotFoundError, ValueError, TypeError):
        return _built_in_edit_missing_state_denial()

    relative = resolved.relative_to(workspace_root).as_posix()
    if phase == "baseline":
        if _is_allowed_baseline_edit_path(relative, source_operator):
            return None
        return _baseline_phase_built_in_edit_denial()

    if phase == "awaiting_round_start":
        return _awaiting_round_start_built_in_edit_denial()

    if phase == "round_active":
        round_dir = _active_round_dir(state)
        if round_dir is None:
            return _built_in_edit_missing_state_denial()
        if relative == round_dir or relative.startswith(f"{round_dir}/"):
            return None
        return _round_active_built_in_edit_denial(round_dir)

    return _built_in_edit_missing_state_denial()


def _require_state_string(state: dict[str, Any], key: str) -> str:
    value = state.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"workflow state {key} must be a non-empty string")
    return value


def _is_allowed_baseline_edit_path(relative_path: str, source_operator: str) -> bool:
    if relative_path == "baseline" or relative_path.startswith("baseline/"):
        return True
    if relative_path == source_operator:
        return True
    if "/" in relative_path:
        return False
    name = Path(relative_path).name
    return (
        name.startswith("test_")
        or name.startswith("differential_test_")
        or name.startswith("bench_")
    )


def _active_round_dir(state: dict[str, Any]) -> str | None:
    current_round = state.get("current_round")
    rounds = state.get("rounds")
    if not isinstance(current_round, int) or not isinstance(rounds, dict):
        return None
    round_entry = rounds.get(str(current_round))
    if not isinstance(round_entry, dict):
        return None
    status = round_entry.get("status")
    round_dir = round_entry.get("round_dir")
    if status != "active" or not isinstance(round_dir, str) or not round_dir:
        return None
    return round_dir


def _built_in_edit_missing_state_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Optimize workflow state is missing or invalid at `.triton-agent/state.json`. "
        "Ask the runner to restart the optimize session so workflow state can be rebuilt. "
        "First-version scope: only built-in edit tools are blocked here; Bash file writes are not blocked."
    )


def _built_in_edit_outside_workspace_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Keep built-in edits inside the current optimize workspace. "
        "First-version scope: only built-in edit tools are blocked here; Bash file writes are not blocked."
    )


def _baseline_phase_built_in_edit_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Current phase is baseline. During baseline, built-in edits are limited to the baseline-minimal file set: "
        "the source operator, root-level test/bench harness files, and `baseline/` artifacts. "
        "Finish or repair baseline, then submit it through `triton-npu-optimize-submit-baseline` before opening a round. "
        "First-version scope: only built-in edit tools are blocked here; Bash file writes are not blocked."
    )


def _awaiting_round_start_built_in_edit_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Current phase is awaiting_round_start, so no optimize round is active yet. "
        "Use `triton-npu-optimize-start-round` to open the next `opt-round-N/` before editing. "
        "First-version scope: only built-in edit tools are blocked here; Bash file writes are not blocked."
    )


def _round_active_built_in_edit_denial(round_dir: str) -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        f"Current active round is {round_dir}. Built-in edits must stay inside `{round_dir}/`. "
        "Edit the round-local snapshot and round artifacts instead of top-level workspace files. "
        "When this round is ready, use `triton-npu-optimize-submit-round` to submit it before moving on. "
        "First-version scope: only built-in edit tools are blocked here; Bash file writes are not blocked."
    )


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


def _protected_script_roots(policy: dict[str, Any], workspace_root: Path) -> list[Path]:
    roots: list[Path] = []
    raw_roots = policy.get("protected_script_roots", [])
    if isinstance(raw_roots, list):
        for raw_root in raw_roots:
            root = _resolve_policy_path(raw_root)
            if root is not None and root not in roots:
                roots.append(root)
    if roots:
        return roots
    return [workspace_root / ".codex" / "skills"]


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
