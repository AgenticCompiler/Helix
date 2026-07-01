from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from triton_agent.trace.core import append_trace_event, utc_timestamp


_THINKING_ENV = "TRITON_AGENT_SHOW_OUTPUT_THINKING"
_THINKING_MODES = {"full", "excerpt", "summary", "presence", "off"}
_DEFAULT_EXCERPT_LIMIT = 2000
_FIELD_EXCERPT_LIMIT = 1000
COMMAND_TOOLS = {"Bash", "PowerShell", "bash", "powershell", "shell"}
FILE_ACCESS_TOOLS = {"Read", "Grep", "Glob", "read", "grep", "glob"}
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "write", "edit", "multiedit"}


@dataclass
class _ClaudeToolLifecycle:
    tool_use_id: str
    tool: str
    tool_input: dict[str, Any]
    start_time: str


@dataclass
class ClaudeShowOutputStats:
    events: int = 0
    tools: int = 0
    errors: int = 0
    thinking_blocks: int = 0


@dataclass(frozen=True)
class _Excerpt:
    text: str
    truncated: bool


class ClaudeShowOutputRenderer:
    """Render Claude stream-json events as readable show-output timeline blocks."""

    _COMMAND_TOOLS = COMMAND_TOOLS
    _FILE_ACCESS_TOOLS = FILE_ACCESS_TOOLS
    _EDIT_TOOLS = EDIT_TOOLS

    def __init__(self, thinking_mode: str | None = None) -> None:
        self.thinking_mode = _resolve_thinking_mode(thinking_mode)
        self.stats = ClaudeShowOutputStats()
        self.thinking_observed = False

    def render_system(self, text: str) -> str:
        return self._block(f"[system] {text}")

    def render_assistant_text(self, text: str) -> str | None:
        text = text.strip()
        if not text:
            return None
        return self._block(text)

    def render_thinking(self, text: str) -> str | None:
        self.thinking_observed = True
        self.stats.thinking_blocks += 1
        stripped = text.strip()
        if self.thinking_mode == "off":
            return None
        if not stripped:
            return self._block(
                "[think:full]",
                "Claude thinking block observed, but full content was not present in stdout.",
            )
        if self.thinking_mode == "presence":
            return self._block(
                "[think:presence]",
                "Claude native thinking block observed; content omitted by "
                f"{_THINKING_ENV}=presence.",
            )
        if self.thinking_mode == "summary":
            return self._block("[think:summary]", _summarize_text(stripped))
        if self.thinking_mode == "excerpt":
            excerpt = _excerpt_text(stripped, limit=_DEFAULT_EXCERPT_LIMIT)
            body = excerpt.text
            if excerpt.truncated:
                body += "\nomitted: thinking truncated in show-output"
            return self._block("[think:excerpt]", body)
        return self._block("[think:full]", stripped)

    def render_missing_thinking_notice(self) -> str | None:
        if self.thinking_observed or self.thinking_mode == "off":
            return None
        return self._block("[think:full]", "No Claude native thinking block was present in stdout.")

    def render_tool_start(self, tool: str, tool_use_id: str, tool_input: dict[str, Any]) -> str:
        self.stats.tools += 1
        del tool_use_id
        lines = [f"[tool:start] {tool}"]
        lines.extend(self._input_lines(tool, tool_input))
        return self._block(*lines)

    def render_tool_end(
        self,
        *,
        tool: str,
        tool_use_id: str,
        duration_ms: int,
        status: str,
        return_code: int | None,
        content_text: str,
        tool_use_result: dict[str, Any],
    ) -> str:
        if status == "error":
            self.stats.errors += 1
        rc_text = str(return_code) if return_code is not None else "unknown"
        lines = [
            f"[tool:end] {tool} "
            f"{status} in {duration_ms}ms rc={rc_text}"
        ]
        combined = _join_non_empty(
            str(tool_use_result.get("stderr", "")),
            str(tool_use_result.get("stdout", "")),
            content_text,
        )
        error_line = _extract_error_line(combined)
        if status == "error" and error_line:
            lines.append(f"  error: {error_line}")

        stdout_text = _string_or_empty(tool_use_result.get("stdout"))
        stderr_text = _string_or_empty(tool_use_result.get("stderr"))
        if stdout_text:
            lines.extend(_excerpt_lines("stdout", stdout_text))
        if stderr_text:
            lines.extend(_excerpt_lines("stderr", stderr_text))
        if not stdout_text and not stderr_text and content_text:
            lines.extend(_excerpt_lines("result", content_text))
        return self._block(*lines)

    def render_result(self, event: dict[str, Any]) -> str:
        subtype = event.get("subtype", "unknown")
        duration_ms = event.get("duration_ms", 0)
        num_turns = event.get("num_turns", 0)
        stop_reason = event.get("stop_reason", "")
        stop_suffix = f" stop={stop_reason}" if stop_reason else ""
        return self._block(f"[result] {subtype} duration={duration_ms}ms turns={num_turns}{stop_suffix}")

    def render_unknown_event(self, event: dict[str, Any]) -> str:
        event_type = event.get("type") or "unknown"
        return self.render_system(f"Skipped unsupported Claude stream-json event type: {event_type}")

    def render_plain_stdout(self, line: str) -> str:
        return line

    def _input_lines(self, tool: str, tool_input: dict[str, Any]) -> list[str]:
        if not tool_input:
            return ["  input: {}"]
        if tool in self._COMMAND_TOOLS:
            command = _string_or_empty(tool_input.get("command"))
            lines = [f"  command: {_one_line_excerpt(command, limit=_FIELD_EXCERPT_LIMIT)}"] if command else []
            if command:
                lines.append(f"  kind: {_classify_command(_unwrap_powershell(command))}")
            return lines or ["  command: (empty)"]
        if tool in self._FILE_ACCESS_TOOLS:
            keys = ("file_path", "path", "pattern", "glob", "offset", "limit", "output_mode")
            return _selected_input_lines(tool_input, keys)
        if tool in self._EDIT_TOOLS:
            keys = ("file_path", "old_string", "new_string")
            return _selected_input_lines(tool_input, keys)
        return _selected_input_lines(tool_input, tuple(tool_input.keys())[:4])

    def _block(self, *lines: str) -> str:
        self.stats.events += 1
        return "\n".join(lines).rstrip() + "\n\n"


class ClaudeJsonLineParser:
    """Parse Claude stream-json, optionally write trace events, and render readable output."""

    _COMMAND_TOOLS = COMMAND_TOOLS
    _FILE_ACCESS_TOOLS = FILE_ACCESS_TOOLS
    _EDIT_TOOLS = EDIT_TOOLS

    _ROUTE: dict[str, str] = {
        "assistant": "_handle_assistant",
        "user": "_handle_user",
        "system": "_handle_system",
        "result": "_handle_result",
    }

    def __init__(
        self,
        trace_path: Path | None,
        extra_env: dict[str, str] | None = None,
        *,
        run_id: str = "",
        workspace_root: str = "",
    ) -> None:
        self._trace_path = trace_path
        self._extra_env = extra_env or {}
        self._run_id = run_id
        self._workspace_root = workspace_root
        self._pending: dict[str, _ClaudeToolLifecycle] = {}
        self._seen: set[tuple[str, str, str]] = set()
        self._session_id: str | None = None
        self._first_event_received = False
        self._renderer = ClaudeShowOutputRenderer(self._extra_env.get(_THINKING_ENV))

    def parse_line(self, line: str) -> str | None:
        try:
            return self._parse_line_inner(line)
        except Exception as exc:
            self._write_diagnostic("claude_parse_error", f"Failed to parse line: {exc}")
            return self._renderer.render_system(f"Failed to parse Claude stream-json line: {exc}")

    def _parse_line_inner(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None

        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            return self._renderer.render_plain_stdout(line)

        if not isinstance(event, dict):
            return self._renderer.render_plain_stdout(line)

        event = cast(dict[str, Any], event)

        if not self._first_event_received:
            self._first_event_received = True
            self._write_diagnostic(
                "claude_native_json_active",
                "Claude native stream-json event stream is active",
            )

        event_type = event.get("type", "")
        handler_name = self._ROUTE.get(event_type, "_handle_unknown")
        handler = cast(Callable[[dict[str, Any]], str | None], getattr(self, handler_name))
        return handler(event)

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
                tool_use_id = _string_or_empty(block.get("id"))
                tool_name = _string_or_empty(block.get("name")) or "unknown"
                tool_input = block.get("input", {})
                if not isinstance(tool_input, dict):
                    tool_input = {}
                tool_input = cast(dict[str, Any], tool_input)
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
                    "session_id": self._session_id or event.get("session_id", ""),
                })

                human_parts.append(self._renderer.render_tool_start(tool_name, tool_use_id, tool_input))

            elif block_type == "text":
                rendered = self._renderer.render_assistant_text(_string_or_empty(block.get("text")))
                if rendered:
                    human_parts.append(rendered)

            elif block_type == "thinking":
                thinking_text = _string_or_empty(block.get("thinking")) or _string_or_empty(block.get("text"))
                rendered = self._renderer.render_thinking(thinking_text)
                if rendered:
                    human_parts.append(rendered)

            elif block_type == "redacted_thinking":
                rendered = self._renderer.render_thinking("")
                if rendered:
                    human_parts.append(rendered)

        return "".join(human_parts) if human_parts else None

    def _handle_user(self, event: dict[str, Any]) -> str | None:
        message = cast(dict[str, Any], event.get("message", {}))
        content = cast(list[Any], message.get("content", []))

        end_time = cast(str, event.get("timestamp", utc_timestamp()))
        raw_tool_use_result = event.get("tool_use_result", {})
        tool_use_result = cast(dict[str, Any], raw_tool_use_result) if isinstance(raw_tool_use_result, dict) else {}
        human_parts: list[str] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            block = cast(dict[str, Any], block)
            if block.get("type") != "tool_result":
                continue

            tool_use_id = _string_or_empty(block.get("tool_use_id"))
            is_error = bool(block.get("is_error", False))
            content_text = _content_to_text(block.get("content", ""))

            cached = self._pending.pop(tool_use_id, None) if tool_use_id else None
            tool_name = cached.tool if cached else "unknown"
            tool_input = cached.tool_input if cached else {}
            start_time = cached.start_time if cached else end_time
            duration_ms = self._duration_ms(end_time, start_time)
            status = "error" if is_error else "ok"
            return_code = _extract_return_code(tool_use_result, content_text)
            if return_code is None and tool_name in self._COMMAND_TOOLS and status == "ok":
                return_code = 0

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
                "return_code": return_code,
                "source": "claude_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "session_id": self._session_id or event.get("session_id", ""),
            })

            self._derive_command_event(
                tool_name,
                tool_use_id,
                tool_input,
                tool_use_result,
                start_time,
                end_time,
                duration_ms,
                status,
                return_code,
            )
            self._derive_file_access_event(tool_name, tool_use_id, tool_input, start_time, end_time, duration_ms)
            self._derive_edit_event(tool_name, tool_use_id, tool_input, start_time, end_time, duration_ms)

            human_parts.append(
                self._renderer.render_tool_end(
                    tool=tool_name,
                    tool_use_id=tool_use_id,
                    duration_ms=duration_ms,
                    status=status,
                    return_code=return_code,
                    content_text=content_text,
                    tool_use_result=tool_use_result,
                )
            )

        return "".join(human_parts) if human_parts else None

    def _handle_system(self, event: dict[str, Any]) -> str | None:
        subtype = event.get("subtype", "")
        if subtype == "init":
            self._session_id = _string_or_empty(event.get("session_id"))
            self._write_diagnostic("claude_session_init", f"Session initialized: {self._session_id}")
            return self._renderer.render_system(f"Claude session {self._session_id or 'unknown'}")
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
        return self._renderer.render_result(event)

    def _handle_unknown(self, event: dict[str, Any]) -> str:
        return self._renderer.render_unknown_event(event)

    def _derive_command_event(
        self,
        tool: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
        tool_use_result: dict[str, Any],
        start_time: str,
        end_time: str,
        duration_ms: int,
        status: str,
        return_code: int | None,
    ) -> None:
        if tool not in self._COMMAND_TOOLS:
            return

        raw_command = _string_or_empty(tool_input.get("command"))
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
            "return_code": return_code,
            "status": status,
            "stdout_excerpt": self._excerpt(_string_or_empty(tool_use_result.get("stdout"))),
            "stderr_excerpt": self._excerpt(_string_or_empty(tool_use_result.get("stderr"))),
            "source": "claude_native_json",
            "confidence": "high",
            "run_id": self._run_id,
        })

    def _derive_file_access_event(
        self,
        tool: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
        start_time: str,
        end_time: str,
        duration_ms: int,
    ) -> None:
        if tool not in self._FILE_ACCESS_TOOLS:
            return

        file_path = (
            tool_input.get("file_path")
            or tool_input.get("path")
            or tool_input.get("pattern")
            or ""
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
        })

    def _derive_edit_event(
        self,
        tool: str,
        tool_use_id: str,
        tool_input: dict[str, Any],
        start_time: str,
        end_time: str,
        duration_ms: int,
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
        })

    def _classify_command(self, command: str) -> str:
        return _classify_command(command)

    def _unwrap_powershell(self, command: str) -> str:
        return _unwrap_powershell(command)

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

    def _excerpt(self, text: str, limit: int = _DEFAULT_EXCERPT_LIMIT) -> str:
        return _excerpt_text(text, limit=limit).text

    def _write_event(self, event: dict[str, Any]) -> None:
        key = (
            str(event.get("tool_use_id", "")),
            str(event.get("phase", "")),
            str(event.get("type", "")),
        )
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
            "timestamp": utc_timestamp(),
        })

    def flush(self) -> str | None:
        human_parts: list[str] = []
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
                "return_code": None,
                "source": "claude_native_json",
                "confidence": "medium",
                "run_id": self._run_id,
            })
            human_parts.append(
                self._renderer.render_tool_end(
                    tool=cached.tool,
                    tool_use_id=tool_use_id,
                    duration_ms=0,
                    status="unknown",
                    return_code=None,
                    content_text="tool result was not observed before stream end",
                    tool_use_result={},
                )
            )
        self._pending.clear()
        if self._first_event_received:
            missing_thinking = self._renderer.render_missing_thinking_notice()
            if missing_thinking:
                human_parts.append(missing_thinking)
        return "".join(human_parts) if human_parts else None

    @property
    def stats(self) -> ClaudeShowOutputStats:
        return self._renderer.stats


class ClaudeJsonOutputFilter:
    """OutputFilter that turns Claude stream-json into readable show-output text."""

    def __init__(
        self,
        trace_path: Path | None,
        extra_env: dict[str, str] | None = None,
        *,
        run_id: str = "",
        workspace_root: str = "",
    ) -> None:
        self._parser = ClaudeJsonLineParser(
            trace_path, extra_env, run_id=run_id, workspace_root=workspace_root
        )
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

        if flush:
            if self._buffer:
                result = self._parser.parse_line(self._buffer)
                self._buffer = ""
                if result:
                    emitted.append(result)
            trailing = self._parser.flush()
            if trailing:
                emitted.append(trailing)

        return "".join(emitted)

    @property
    def parser(self) -> ClaudeJsonLineParser:
        return self._parser


def build_claude_trace_env(
    existing: dict[str, str] | None,
    *,
    trace_path: Path,
    run_id: str,
    workspace_root: Path,
) -> dict[str, str]:
    env = dict(existing or {})
    env["TRITON_AGENT_OTEL_TRACE_PATH"] = str(trace_path)
    env["TRITON_AGENT_OTEL_RUN_ID"] = run_id
    env["TRITON_AGENT_WORKSPACE_ROOT"] = str(workspace_root)
    return env


def _resolve_thinking_mode(value: str | None) -> str:
    raw_value = value or os.environ.get(_THINKING_ENV, "full")
    normalized = raw_value.strip().lower()
    if normalized in _THINKING_MODES:
        return normalized
    return "full"


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _unwrap_powershell(command: str) -> str:
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


def _classify_command(command: str) -> str:
    lower = command.lower()
    if "compare-perf" in lower:
        return "compare_perf"
    if "compare-result" in lower:
        return "compare_result"
    if "submit-baseline" in lower:
        return "check_baseline"
    if "submit-round" in lower:
        return "check_round"
    if "run-test" in lower or "run-test-baseline" in lower or "run-test-optimize" in lower or "pytest" in lower or "differential_test_" in lower:
        return "correctness_test"
    if "run-bench" in lower or "bench_" in lower:
        if "ssh" in lower or _command_has_remote(command):
            return "remote_bench"
        return "benchmark"
    if "msprof" in lower or "profile export" in lower:
        return "profile"
    if _command_has_remote(command):
        return "remote_command"
    return "local_command"


def _command_has_remote(command: str) -> bool:
    tokens = command.split()
    return any(Path(token).name == "ssh" for token in tokens)


def _extract_return_code(tool_use_result: Mapping[str, Any], content_text: str) -> int | None:
    for key in ("return_code", "exit_code", "rc"):
        value = tool_use_result.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
            return int(value.strip())
    combined = _join_non_empty(
        _string_or_empty(tool_use_result.get("stderr")),
        _string_or_empty(tool_use_result.get("stdout")),
        content_text,
    )
    match = re.search(r"\bExit code\s+(-?\d+)\b", combined, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bexit status\s+(-?\d+)\b", combined, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_error_line(text: str) -> str:
    if not text:
        return ""
    patterns = (
        r"ModuleNotFoundError:\s*.*",
        r"ValueError:\s*.*",
        r"RuntimeError:\s*.*",
        r"AssertionError:\s*.*",
        r"FAIL:\s*.*",
        r"round check requires fixes:\s*.*",
        r"submit-round requires fixes:\s*.*",
        r"usage:\s*.*",
        r".*\berror:\s*.*",
        r"No op_statistic csv.*",
        r".*\bub overflow\b.*",
        r"Exit code\s+-?\d+",
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for pattern in patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        for line in lines:
            match = regex.search(line)
            if match:
                return _one_line_excerpt(match.group(0), limit=500)
    if any(line.startswith("Traceback") for line in lines):
        for line in reversed(lines):
            if re.search(r"\w+(Error|Exception):", line):
                return _one_line_excerpt(line, limit=500)
    return ""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in cast(list[Any], content):
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                text = item_dict.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(item_dict, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _selected_input_lines(tool_input: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for key in keys:
        if key not in tool_input:
            continue
        lines.append(f"  {key}: {_format_input_value(tool_input[key])}")
    return lines or ["  input: {}"]


def _format_input_value(value: Any) -> str:
    if isinstance(value, str):
        return _one_line_excerpt(value, limit=_FIELD_EXCERPT_LIMIT)
    if isinstance(value, (int, float, bool)) or value is None:
        return str(value)
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        text = str(value)
    return _one_line_excerpt(text, limit=_FIELD_EXCERPT_LIMIT)


def _excerpt_lines(label: str, text: str) -> list[str]:
    excerpt = _excerpt_text(text)
    if not excerpt.text:
        return []
    if "\n" not in excerpt.text:
        lines = [f"  {label}: {excerpt.text}"]
    else:
        lines = [f"  {label} excerpt:"]
        lines.extend(f"    {line}" for line in excerpt.text.splitlines())
    if excerpt.truncated:
        lines.append(f"  omitted: {label} truncated in show-output")
    return lines


def _excerpt_text(text: str, *, limit: int = _DEFAULT_EXCERPT_LIMIT) -> _Excerpt:
    if not text:
        return _Excerpt("", False)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(normalized) <= limit:
        return _Excerpt(normalized.rstrip(), False)
    head_limit = max(1, limit - 80)
    return _Excerpt(normalized[:head_limit].rstrip() + "\n<truncated>", True)


def _one_line_excerpt(text: str, *, limit: int) -> str:
    compact = " ".join(text.replace("\r", "\n").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 15)].rstrip() + " ... <truncated>"


def _summarize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "Claude native thinking block was present but empty."
    first = _one_line_excerpt(lines[0], limit=500)
    if len(lines) == 1:
        return first
    return f"{first}\nsummary: {len(lines)} non-empty lines, {len(text)} characters"


def _join_non_empty(*parts: str) -> str:
    return "\n".join(part for part in parts if part)


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""
