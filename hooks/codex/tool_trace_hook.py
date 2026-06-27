#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Static policy constants
# ---------------------------------------------------------------------------

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
    r"(?P<path>(?:/|\.\.?/|\.codex/|\.opencode/)[A-Za-z0-9_./*?{}+@%:,=-]+)"
)
WINDOWS_PATH_FRAGMENT_RE = re.compile(
    r"(?P<path>[A-Za-z]:[\\/][A-Za-z0-9_ .\\/(){}+@%:,=-]+)"
)
TRACE_PATH_ENV = "TRITON_AGENT_OTEL_TRACE_PATH"
TRACE_RUN_ID_ENV = "TRITON_AGENT_OTEL_RUN_ID"
TRACE_WORKSPACE_ROOT_ENV = "TRITON_AGENT_WORKSPACE_ROOT"
READ_TOOLS = {"Read", "Grep", "Glob"}
EDIT_TOOLS = {"Edit", "MultiEdit", "Write"}


# ---------------------------------------------------------------------------
# Hook entrypoints
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--event", required=True, choices=["PreToolUse", "PostToolUse"])
    args = parser.parse_args(argv)

    try:
        policy = _load_json(Path(args.policy))
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"triton-agent codex hook failed open: {exc}", file=sys.stderr)
        return 0

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    try:
        if args.event == "PreToolUse":
            reason = deny_reason_for_tool_use(policy, payload)
            append_trace_events(policy, payload, blocked=reason is not None, event=args.event)
            if reason is not None:
                json.dump(build_denial_output(reason), sys.stdout)
        elif args.event == "PostToolUse":
            append_posttooluse_trace(policy, payload)
    except Exception as exc:
        print(f"triton-agent codex hook trace failed open: {exc}", file=sys.stderr)

    return 0


# ---------------------------------------------------------------------------
# Tool-use denial checks
# ---------------------------------------------------------------------------


def deny_reason_for_tool_use(policy: dict[str, Any], payload: dict[str, Any]) -> str | None:
    guard_policy = _guard_policy(policy)
    if not guard_policy.get("enabled", True):
        return None

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
    allow_read_roots = _allow_read_roots(guard_policy, workspace_root)
    deny_read_globs = [str(item) for item in guard_policy.get("deny_read_globs", []) if isinstance(item, str)]
    deny_message = str(guard_policy.get("deny_message") or "This read is blocked by workspace policy.")

    for path_text in _collect_command_path_references(command, tokens):
        resolved_path = _resolve_path_text(path_text, cwd, workspace_root)
        if resolved_path is None:
            continue
        if not _is_under_any_root(resolved_path, allow_read_roots):
            return deny_message
        if _matches_any_glob(resolved_path, deny_read_globs):
            return deny_message

    return None


# ---------------------------------------------------------------------------
# Trace output
# ---------------------------------------------------------------------------


def append_trace_events(policy: dict[str, Any], payload: dict[str, Any], *, blocked: bool, event: str) -> None:
    trace_policy = _trace_policy(policy)
    if not trace_policy.get("enabled", bool(os.environ.get(TRACE_PATH_ENV))):
        return
    trace_path = str(trace_policy.get("path") or os.environ.get(TRACE_PATH_ENV) or "")
    if not trace_path:
        return

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_name, str):
        tool_name = "unknown"
    if not isinstance(tool_input, dict):
        tool_input = {}

    timestamp = _timestamp()
    run_id = str(trace_policy.get("run_id") or os.environ.get(TRACE_RUN_ID_ENV) or "")

    tool_call_start_event = {
        "timestamp": timestamp,
        "schema_version": 1,
        "run_id": run_id,
        "type": "tool_call",
        "phase": "start",
        "tool": tool_name,
        "start_time": timestamp,
        "status": "blocked" if blocked else "started",
        "summary": _tool_summary(tool_name, tool_input),
        "tool_use_id": payload.get("tool_use_id"),
        "source": "codex_posttooluse",
        "confidence": "high",
    }
    _append_trace_event(Path(trace_path), tool_call_start_event)

    if tool_name == "Bash":
        command = tool_input.get("command")
        if isinstance(command, str):
            command_event = {
                "timestamp": timestamp,
                "schema_version": 1,
                "run_id": run_id,
                "type": "command",
                "phase": "start",
                "command_kind": _classify_command(command),
                "command": _unwrap_powershell(command),
                "raw_command": command,
                "status": "blocked" if blocked else "started",
                "source": "codex_posttooluse",
                "confidence": "high",
            }
            _append_trace_event(Path(trace_path), command_event)

            tokens = _split_command(command)
            if _contains_read_command(tokens):
                workspace_root = _resolve_policy_path(
                    os.environ.get(TRACE_WORKSPACE_ROOT_ENV) or policy.get("workspace_root")
                )
                if workspace_root is not None:
                    cwd = _resolve_cwd(tool_input.get("cwd"), workspace_root)
                    for path_text in _collect_command_path_references(command, tokens):
                        resolved_path = _resolve_path_text(path_text, cwd, workspace_root)
                        if resolved_path is None:
                            continue
                        file_event = {
                            "timestamp": timestamp,
                            "schema_version": 1,
                            "run_id": run_id,
                            "type": "file_access",
                            "phase": "instant",
                            "action": "read",
                            "path": _display_path(resolved_path, workspace_root),
                            "status": "blocked" if blocked else "started",
                            "source": "codex_posttooluse",
                            "confidence": "high",
                        }
                        _append_trace_event(Path(trace_path), file_event)
        return

    workspace_root = _resolve_policy_path(os.environ.get(TRACE_WORKSPACE_ROOT_ENV) or policy.get("workspace_root"))
    if workspace_root is None:
        return
    cwd = _resolve_cwd(tool_input.get("cwd"), workspace_root)
    if tool_name in READ_TOOLS:
        _append_non_bash_file_access_trace(
            Path(trace_path), trace_policy, tool_name=tool_name, tool_input=tool_input,
            timestamp=timestamp, workspace_root=workspace_root, cwd=cwd, blocked=blocked, run_id=run_id,
        )
    elif tool_name in EDIT_TOOLS:
        _append_non_bash_edit_trace(
            Path(trace_path), trace_policy, tool_name=tool_name, tool_input=tool_input,
            timestamp=timestamp, workspace_root=workspace_root, cwd=cwd, blocked=blocked, run_id=run_id,
        )


def append_posttooluse_trace(policy: dict[str, Any], payload: dict[str, Any]) -> None:
    """Write PostToolUse trace events - phase=end with duration and return code."""
    trace_policy = _trace_policy(policy)
    if not trace_policy.get("enabled", bool(os.environ.get(TRACE_PATH_ENV))):
        return
    trace_path = str(trace_policy.get("path") or os.environ.get(TRACE_PATH_ENV) or "")
    if not trace_path:
        return

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    timestamp = _timestamp()
    run_id = str(trace_policy.get("run_id") or os.environ.get(TRACE_RUN_ID_ENV) or "")
    tool_use_id = payload.get("tool_use_id")

    # Get tool result info from payload
    status = "ok"
    return_code = None
    result_obj = payload.get("result") or payload.get("tool_result") or {}
    if isinstance(result_obj, dict):
        status = str(result_obj.get("status", result_obj.get("outcome", "ok")))
        return_code = result_obj.get("return_code") or result_obj.get("exit_code")

    # Write tool_call end event
    tool_event = {
        "timestamp": timestamp,
        "schema_version": 1,
        "run_id": run_id,
        "type": "tool_call",
        "phase": "end",
        "tool": tool_name if isinstance(tool_name, str) else "unknown",
        "tool_use_id": tool_use_id,
        "end_time": timestamp,
        "status": status,
        "return_code": return_code,
        "summary": _tool_summary(tool_name if isinstance(tool_name, str) else "unknown", tool_input),
        "source": "codex_posttooluse",
        "confidence": "high",
        "duration_ms": 0,
        "duration_source": "hook_clock_join",
    }
    _append_trace_event(Path(trace_path), tool_event)

    # If it's a Bash command, write a command end event
    if tool_name == "Bash" and isinstance(tool_input.get("command"), str):
        command = tool_input["command"]
        command_event = {
            "timestamp": timestamp,
            "schema_version": 1,
            "run_id": run_id,
            "type": "command",
            "phase": "end",
            "tool_use_id": tool_use_id,
            "command_kind": _classify_command(command),
            "command": _unwrap_powershell(command),
            "raw_command": command,
            "status": status,
            "return_code": return_code,
            "source": "codex_posttooluse",
            "confidence": "high",
            "duration_ms": 0,
            "duration_source": "hook_clock_join",
        }
        _append_trace_event(Path(trace_path), command_event)


# ---------------------------------------------------------------------------
# Shared trace metadata helpers
# ---------------------------------------------------------------------------


def _guard_policy(policy: dict[str, Any]) -> dict[str, Any]:
    guard = policy.get("guard")
    if isinstance(guard, dict):
        return guard
    return policy


def _trace_policy(policy: dict[str, Any]) -> dict[str, Any]:
    trace = policy.get("trace")
    if isinstance(trace, dict):
        return trace
    return {"enabled": bool(os.environ.get(TRACE_PATH_ENV))}


def _append_trace_event(trace_path: Path, event: dict[str, Any]) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tool_summary(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "Bash":
        command = tool_input.get("command")
        return command if isinstance(command, str) else "Bash command"
    for key in ("file_path", "path", "pattern"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value
    return tool_name


# ---------------------------------------------------------------------------
# Non-Bash trace helpers
# ---------------------------------------------------------------------------


def _append_non_bash_file_access_trace(
    trace_path: Path,
    trace_policy: dict[str, Any],
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    timestamp: str,
    workspace_root: Path,
    cwd: Path,
    blocked: bool,
    run_id: str,
) -> None:
    for raw_path in _tool_input_paths(tool_name, tool_input):
        resolved_path = _resolve_path_text(raw_path, cwd, workspace_root)
        if resolved_path is None:
            continue
        event: dict[str, Any] = {
            "timestamp": timestamp,
            "schema_version": 1,
            "run_id": run_id,
            "type": "file_access",
            "phase": "instant",
            "action": "search" if tool_name in {"Grep", "Glob"} else "read",
            "path": _display_path(resolved_path, workspace_root),
            "resolved_under_workspace": _is_relative_to(resolved_path, workspace_root),
            "status": "blocked" if blocked else "started",
            "source": "codex_posttooluse",
            "confidence": "high",
        }
        try:
            if resolved_path.is_file():
                event["bytes"] = resolved_path.stat().st_size
        except OSError:
            pass
        _append_trace_event(trace_path, event)


def _append_non_bash_edit_trace(
    trace_path: Path,
    trace_policy: dict[str, Any],
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    timestamp: str,
    workspace_root: Path,
    cwd: Path,
    blocked: bool,
    run_id: str,
) -> None:
    raw_path = _first_tool_input_path(tool_input)
    if raw_path is None:
        return
    resolved_path = _resolve_path_text(raw_path, cwd, workspace_root)
    if resolved_path is None:
        return
    added_lines, removed_lines, digest = _edit_stats(tool_name, tool_input)
    event: dict[str, Any] = {
        "timestamp": timestamp,
        "schema_version": 1,
        "run_id": run_id,
        "type": "edit",
        "phase": "instant",
        "path": _display_path(resolved_path, workspace_root),
        "edit_kind": _classify_edit_path(resolved_path),
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "diff_digest": digest,
        "status": "blocked" if blocked else "started",
        "source": "codex_posttooluse",
        "confidence": "high",
    }
    _append_trace_event(trace_path, event)


def _tool_input_paths(tool_name: str, tool_input: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for key in ("file_path", "path", "filePath", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            paths.append(value)
    if tool_name == "Glob":
        pattern = tool_input.get("pattern")
        if isinstance(pattern, str) and pattern:
            paths.append(pattern)
    return paths


def _first_tool_input_path(tool_input: dict[str, Any]) -> str | None:
    for key in ("file_path", "path", "filePath", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _edit_stats(tool_name: str, tool_input: dict[str, Any]) -> tuple[int, int, str]:
    added_lines = 0
    removed_lines = 0
    digest_parts: list[str] = [tool_name]
    if tool_name == "MultiEdit" and isinstance(tool_input.get("edits"), list):
        for edit in tool_input.get("edits", []):
            if not isinstance(edit, dict):
                continue
            old_text = edit.get("old_string")
            new_text = edit.get("new_string")
            if isinstance(old_text, str):
                removed_lines += _line_count(old_text)
                digest_parts.append(old_text)
            if isinstance(new_text, str):
                added_lines += _line_count(new_text)
                digest_parts.append(new_text)
    else:
        old_text = tool_input.get("old_string")
        new_text = tool_input.get("new_string")
        content = tool_input.get("content")
        if isinstance(old_text, str):
            removed_lines += _line_count(old_text)
            digest_parts.append(old_text)
        if isinstance(new_text, str):
            added_lines += _line_count(new_text)
            digest_parts.append(new_text)
        if isinstance(content, str):
            added_lines += _line_count(content)
            digest_parts.append(content)
    digest = "sha256:" + hashlib.sha256("\n".join(digest_parts).encode("utf-8", errors="replace")).hexdigest()
    return added_lines, removed_lines, digest


def _line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines()) or 1


def _classify_edit_path(path: Path) -> str:
    name = path.name.lower()
    normalized = path.as_posix().lower()
    if "/opt-round-" in normalized:
        return "round_artifact"
    if name.startswith(("test_", "differential_test_")):
        return "test_harness"
    if name.startswith("bench_"):
        return "bench_harness"
    if name.endswith((".md", ".txt", ".json", ".yaml", ".yml", ".toml")):
        return "metadata" if name.endswith((".json", ".yaml", ".yml", ".toml")) else "documentation"
    if name.endswith(".py"):
        return "operator"
    return "unknown"


# ---------------------------------------------------------------------------
# Command and path helpers
# ---------------------------------------------------------------------------


def _classify_command(command: str) -> str:
    normalized = command.lower()
    if "compare-perf" in normalized:
        return "compare_perf"
    if "compare-result" in normalized:
        return "compare_result"
    if "check-baseline" in normalized:
        return "check_baseline"
    if "check-round" in normalized:
        return "check_round"
    if "run-test" in normalized or "pytest" in normalized or "differential_test_" in normalized:
        return "correctness_test"
    if "run-bench" in normalized or "bench_" in normalized:
        if "ssh" in normalized:
            return "remote_bench"
        return "benchmark"
    if "msprof" in normalized or "profile export" in normalized:
        return "profile"
    if _extract_remote(command) is not None:
        return "remote_command"
    return "local_command"


def _unwrap_powershell(command: str) -> str:
    """Strip PowerShell wrapper from command."""
    ps_re = re.compile(
        r'^"([^"]*\\powershell\.exe)"\s+-Command\s+"(.+)"$',
        re.IGNORECASE,
    )
    match = ps_re.match(command.strip())
    if match:
        inner = match.group(2)
        if inner.startswith('"') and inner.endswith('"'):
            inner = inner[1:-1]
        return inner
    return command


def _extract_remote(command: str) -> str | None:
    tokens = _split_command(command)
    for index, token in enumerate(tokens):
        if Path(token).name != "ssh":
            continue
        if index + 1 < len(tokens):
            return tokens[index + 1]
        return "ssh"
    return None


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


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
        return shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return []


def _contains_read_command(tokens: list[str]) -> bool:
    return any(_is_read_command_token(token) for token in tokens)


def _is_read_command_token(token: str) -> bool:
    return Path(token).name in READ_COMMANDS


def _collect_command_path_references(command: str, tokens: list[str]) -> list[str]:
    path_texts: list[str] = []
    for token in tokens:
        if _is_read_command_token(token):
            continue
        if _looks_like_path(token):
            path_texts.append(token)

    for match in PATH_FRAGMENT_RE.finditer(command):
        path_text = match.group("path")
        if not _is_read_command_token(path_text):
            path_texts.append(path_text)
    for match in WINDOWS_PATH_FRAGMENT_RE.finditer(command):
        path_text = match.group("path").rstrip("'\"),")
        if not _is_read_command_token(path_text):
            path_texts.append(path_text)

    return path_texts


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
        or token.startswith(".opencode/")
        or "\\" in token
        or path.suffix != ""
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


def _allow_read_roots(policy: dict[str, Any], workspace_root: Path) -> list[Path]:
    roots = [workspace_root]
    for raw_root in policy.get("allow_read_roots", []):
        root = _resolve_policy_path(raw_root)
        if root is not None and root not in roots:
            roots.append(root)
    return roots


def _resolve_path_text(path_text: str, cwd: Path, workspace_root: Path) -> Path | None:
    if "*" in path_text or "?" in path_text or "{" in path_text or "}" in path_text:
        return None
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve()
    except OSError:
        return (workspace_root / path_text).resolve()


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
