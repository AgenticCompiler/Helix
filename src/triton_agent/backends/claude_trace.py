from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, cast

from triton_agent.otel_trace import append_trace_event, utc_timestamp


@dataclass
class _ClaudeToolLifecycle:
    """Buffers a tool_use start until the matching tool_result arrives."""
    tool_use_id: str
    tool: str
    tool_input: dict[str, Any]
    start_time: str  # ISO timestamp


class ClaudeJsonLineParser:
    """Parses Claude stream-json output, joins tool_use -> tool_result,
    derives command/file_access/edit events, renders human-readable output."""

    _COMMAND_TOOLS = {"Bash", "PowerShell", "bash", "powershell", "shell"}
    _FILE_ACCESS_TOOLS = {"Read", "Grep", "Glob", "read", "grep", "glob"}
    _EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "write", "edit", "multiedit"}

    _ROUTE: dict[str, str] = {
        "assistant": "_handle_assistant",
        "user": "_handle_user",
        "system": "_handle_system",
        "result": "_handle_result",
    }

    def __init__(self, trace_path: Path, extra_env: dict[str, str] | None = None) -> None:
        self._trace_path = trace_path
        self._extra_env = extra_env or {}
        self._run_id = self._extra_env.get("TRITON_AGENT_OTEL_RUN_ID", "")
        self._role = self._extra_env.get("TRITON_AGENT_OTEL_ROLE", "worker")
        self._workspace_root = self._extra_env.get("TRITON_AGENT_WORKSPACE_ROOT", "")
        self._pending: dict[str, _ClaudeToolLifecycle] = {}
        self._seen: set[tuple[str, str, str]] = set()
        self._session_id: str | None = None
        self._first_event_received = False

    # ── main entry ──────────────────────────────────────────────

    def parse_line(self, line: str) -> str | None:
        try:
            return self._parse_line_inner(line)
        except Exception as exc:
            self._write_diagnostic("claude_parse_error", f"Failed to parse line: {exc}")
            return line

    def _parse_line_inner(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return line  # non-JSON line, pass through

        if not isinstance(event, dict):
            return line

        event = cast(dict[str, Any], event)

        if not self._first_event_received:
            self._first_event_received = True
            self._write_diagnostic("claude_native_json_active",
                                   "Claude native stream-json event stream is active")

        event_type = event.get("type", "")
        handler_name = self._ROUTE.get(event_type, "_handle_unknown")
        handler = cast(Callable[[dict[str, Any]], str | None], getattr(self, handler_name, self._handle_unknown))
        return handler(event)

    # ── event handlers ──────────────────────────────────────────

    def _handle_assistant(self, event: dict[str, Any]) -> str | None:
        message = cast(dict[str, Any], event.get("message", {}))
        content = cast(list[Any], message.get("content", []))

        human_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block = cast(dict[str, Any], block)
            block_type = block.get("type", "")

            if block_type == "tool_use":
                tool_use_id = block.get("id", "")
                tool_name = block.get("name", "unknown")
                tool_input = block.get("input", {})
                start_time = utc_timestamp()

                if tool_use_id:
                    self._pending[tool_use_id] = _ClaudeToolLifecycle(
                        tool_use_id=tool_use_id,
                        tool=tool_name,
                        tool_input=tool_input,
                        start_time=start_time,
                    )

                self._write_event({
                    "schema_version": 1,
                    "type": "tool_call",
                    "phase": "start",
                    "tool": tool_name,
                    "tool_use_id": tool_use_id,
                    "tool_input": tool_input,
                    "start_time": start_time,
                    "status": "started",
                    "source": "claude_native_json",
                    "confidence": "high",
                    "run_id": self._run_id,
                    "role": self._role,
                    "session_id": self._session_id or event.get("session_id", ""),
                })

                human_parts.append(
                    f"[tool] {tool_name} started: "
                    f"{self._summarize_input(tool_name, tool_input)}"
                )

            elif block_type == "text":
                text = block.get("text", "")
                if text:
                    human_parts.append(text)

            elif block_type == "thinking":
                pass  # skip — don't render, don't trace

        return "\n".join(human_parts) if human_parts else None

    def _handle_user(self, event: dict[str, Any]) -> str | None:
        message = cast(dict[str, Any], event.get("message", {}))
        content = cast(list[Any], message.get("content", []))

        end_time = cast(str, event.get("timestamp", utc_timestamp()))
        tool_use_result = cast(dict[str, Any], event.get("tool_use_result", {}))
        human_parts: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block = cast(dict[str, Any], block)
            if block.get("type") != "tool_result":
                continue

            tool_use_id = block.get("tool_use_id", "")
            is_error = block.get("is_error", False)

            # Join with buffered start
            cached = self._pending.pop(tool_use_id, None) if tool_use_id else None
            tool_name = cached.tool if cached else "unknown"
            tool_input = cached.tool_input if cached else {}
            start_time = cached.start_time if cached else end_time
            duration_ms = self._duration_ms(end_time, start_time)
            status = "error" if is_error else "ok"

            # Write tool_call end event
            self._write_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "end",
                "tool": tool_name,
                "tool_use_id": tool_use_id,
                "tool_input": tool_input,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "duration_source": "claude_native_json",
                "status": status,
                "source": "claude_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "role": self._role,
                "session_id": self._session_id or event.get("session_id", ""),
            })

            # Derive command / file_access / edit events
            self._derive_command_event(tool_name, tool_use_id, tool_input,
                                       tool_use_result, start_time, end_time,
                                       duration_ms, status)
            self._derive_file_access_event(tool_name, tool_use_id, tool_input,
                                           start_time, end_time, duration_ms)
            self._derive_edit_event(tool_name, tool_use_id, tool_input,
                                    start_time, end_time, duration_ms)

            human_parts.append(
                f"[tool] {tool_name} done in {duration_ms}ms"
                + (" (error)" if is_error else "")
            )

        return "\n".join(human_parts) if human_parts else None

    def _handle_system(self, event: dict[str, Any]) -> str | None:
        subtype = event.get("subtype", "")
        if subtype == "init":
            self._session_id = event.get("session_id", "")
            self._write_diagnostic("claude_session_init",
                                   f"Session initialized: {self._session_id}")
            return f"[system] Session: {self._session_id}"
        return None

    def _handle_result(self, event: dict[str, Any]) -> str | None:
        subtype = event.get("subtype", "unknown")
        duration_ms = event.get("duration_ms", 0)
        num_turns = event.get("num_turns", 0)
        stop_reason = event.get("stop_reason", "")
        self._write_diagnostic(
            "claude_result",
            f"Run completed: {subtype}, duration={duration_ms}ms, "
            f"turns={num_turns}, stop={stop_reason}",
        )
        return f"[result] {subtype}: {duration_ms}ms, {num_turns} turns"

    def _handle_unknown(self, event: dict[str, Any]) -> str:
        return json.dumps(event)

    # ── derived events ──────────────────────────────────────────

    def _derive_command_event(
        self, tool: str, tool_use_id: str, tool_input: dict[str, Any],
        tool_use_result: dict[str, Any],
        start_time: str, end_time: str, duration_ms: int, status: str,
    ) -> None:
        if tool not in self._COMMAND_TOOLS:
            return

        raw_command = tool_input.get("command", "")
        if not raw_command:
            return

        unwrapped = self._unwrap_powershell(raw_command)
        command_kind = self._classify_command(unwrapped)
        remote = self._extract_remote(unwrapped)

        self._write_event({
            "schema_version": 1,
            "type": "command",
            "phase": "end",
            "tool_use_id": tool_use_id,
            "command": unwrapped,
            "raw_command": raw_command,
            "shell": "powershell" if "powershell" in raw_command.lower() else "bash",
            "command_kind": command_kind,
            "remote": remote,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "return_code": None,  # stream-json does not directly expose it
            "status": status,
            "stdout_excerpt": self._excerpt(tool_use_result.get("stdout", "")),
            "stderr_excerpt": self._excerpt(tool_use_result.get("stderr", "")),
            "source": "claude_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })

    def _derive_file_access_event(
        self, tool: str, tool_use_id: str, tool_input: dict[str, Any],
        start_time: str, end_time: str, duration_ms: int,
    ) -> None:
        if tool not in self._FILE_ACCESS_TOOLS:
            return

        file_path = (
            tool_input.get("file_path") or tool_input.get("path")
            or tool_input.get("pattern") or ""
        )

        self._write_event({
            "schema_version": 1,
            "type": "file_access",
            "phase": "end",
            "tool_use_id": tool_use_id,
            "tool": tool,
            "file_path": file_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "source": "claude_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })

    def _derive_edit_event(
        self, tool: str, tool_use_id: str, tool_input: dict[str, Any],
        start_time: str, end_time: str, duration_ms: int,
    ) -> None:
        if tool not in self._EDIT_TOOLS:
            return

        file_path = tool_input.get("file_path", "")

        self._write_event({
            "schema_version": 1,
            "type": "edit",
            "phase": "end",
            "tool_use_id": tool_use_id,
            "tool": tool,
            "file_path": file_path,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "source": "claude_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })

    # ── helpers ─────────────────────────────────────────────────

    def _classify_command(self, command: str) -> str:
        lower = command.lower()
        if "compare-perf" in lower:
            return "compare_perf"
        if "compare-result" in lower:
            return "compare_result"
        if "check-baseline" in lower:
            return "check_baseline"
        if "check-round" in lower:
            return "check_round"
        if "run-test" in lower or "pytest" in lower or "differential_test_" in lower:
            return "correctness_test"
        if "run-bench" in lower or "bench_" in lower:
            if "ssh" in lower or self._extract_remote(command):
                return "remote_bench"
            return "benchmark"
        if "msprof" in lower or "profile export" in lower:
            return "profile"
        if self._extract_remote(command):
            return "remote_command"
        return "local_command"

    def _unwrap_powershell(self, command: str) -> str:
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

    def _extract_remote(self, command: str) -> str | None:
        tokens = command.split()
        for i, token in enumerate(tokens):
            if Path(token).name == "ssh" and i + 1 < len(tokens):
                return tokens[i + 1]
        return None

    def _duration_ms(self, end_time_str: str, start_time_str: str) -> int:
        start = _parse_timestamp(start_time_str)
        end = _parse_timestamp(end_time_str)
        if start and end:
            delta = end - start
            return max(0, int(delta.total_seconds() * 1000))
        return 0

    def _excerpt(self, text: str, limit: int = 2000) -> str:
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + "\n<truncated>"

    def _summarize_input(self, tool: str, tool_input: dict[str, Any]) -> str:
        if tool in self._COMMAND_TOOLS:
            cmd = tool_input.get("command", "")
            return cmd[:120] if cmd else "(no command)"
        if tool in self._FILE_ACCESS_TOOLS:
            return tool_input.get("pattern", tool_input.get("path", tool_input.get("file_path", "(no path)")))[:80]
        if tool in self._EDIT_TOOLS:
            return tool_input.get("file_path", "(no file)")[:80]
        keys = list(tool_input.keys())[:3]
        return ", ".join(f"{k}={str(tool_input[k])[:40]}" for k in keys)

    # ── write helpers ───────────────────────────────────────────

    def _write_event(self, event: dict[str, Any]) -> None:
        """Write trace event, deduplicating on (tool_use_id, phase, type)."""
        key = (str(event.get("tool_use_id", "")),
               str(event.get("phase", "")),
               str(event.get("type", "")))
        if key in self._seen:
            return
        self._seen.add(key)
        append_trace_event(self._trace_path, event)

    def _write_diagnostic(self, code: str, detail: str) -> None:
        append_trace_event(self._trace_path, {
            "schema_version": 1,
            "type": "diagnostic",
            "phase": "instant",
            "code": code,
            "detail": detail,
            "source": "claude_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
            "timestamp": utc_timestamp(),
        })

    def flush(self) -> None:
        """Called when stream ends. Write any pending tool_start as incomplete."""
        for tool_use_id, cached in list(self._pending.items()):
            self._write_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "end",
                "tool": cached.tool,
                "tool_use_id": tool_use_id,
                "tool_input": cached.tool_input,
                "start_time": cached.start_time,
                "end_time": utc_timestamp(),
                "duration_ms": 0,
                "duration_source": "claude_native_json",
                "status": "unknown",
                "source": "claude_native_json",
                "confidence": "medium",
                "run_id": self._run_id,
                "role": self._role,
            })
        self._pending.clear()


class ClaudeJsonOutputFilter:
    """OutputFilter that parses Claude stream-json, writes trace events,
    and returns human-readable text for show-output log."""

    def __init__(self, trace_path: Path, extra_env: dict[str, str] | None = None) -> None:
        self._parser = ClaudeJsonLineParser(trace_path, extra_env)
        self._buffer = ""

    def feed(self, text: str, *, flush: bool = False) -> str:
        self._buffer += text
        emitted: list[str] = []

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                break
            line = self._buffer[:newline_index + 1]
            self._buffer = self._buffer[newline_index + 1:]
            result = self._parser.parse_line(line)
            if result:
                emitted.append(result)

        if flush and self._buffer:
            result = self._parser.parse_line(self._buffer)
            self._buffer = ""
            if result:
                emitted.append(result)
            self._parser.flush()

        return "".join(emitted)

    @property
    def parser(self) -> ClaudeJsonLineParser:
        return self._parser


def build_claude_trace_env(
    existing: dict[str, str] | None,
    *,
    trace_path: Path,
    run_id: str,
    role: str,
    workspace_root: Path,
) -> dict[str, str]:
    """Build environment dict with trace variables for Claude JSON capture."""
    env = dict(existing or {})
    env["TRITON_AGENT_OTEL_TRACE_PATH"] = str(trace_path)
    env["TRITON_AGENT_OTEL_RUN_ID"] = run_id
    env["TRITON_AGENT_OTEL_ROLE"] = role
    env["TRITON_AGENT_WORKSPACE_ROOT"] = str(workspace_root)
    return env


def _parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO timestamp string to datetime, or return None."""
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None
