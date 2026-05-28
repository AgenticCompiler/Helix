from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from triton_agent.otel_trace import append_trace_event, utc_timestamp


@dataclass
class _ToolLifecycle:
    """Buffers a tool_start event until the matching tool_end arrives."""
    tool_use_id: str
    tool: str
    start_time: str
    tool_input: dict[str, Any] | None = None
    start_counter: float | None = None
    phase: str = "start"


class CodexJsonLineParser:
    """
    Parses Codex JSONL output lines, buffers tool_start events,
    joins with tool_end to compute duration_ms, and writes trace events.

    Also produces human-readable lines for show-output log.
    """

    def __init__(
        self,
        trace_path: Path | None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._trace_path = trace_path
        self._extra_env = extra_env or {}
        self._run_id = self._extra_env.get("TRITON_AGENT_OTEL_RUN_ID", "")
        self._role = self._extra_env.get("TRITON_AGENT_OTEL_ROLE", "worker")
        self._workspace_root = self._extra_env.get("TRITON_AGENT_WORKSPACE_ROOT", "")

        # tool_use_id -> ToolLifecycle buffer for join
        self._pending: dict[str, _ToolLifecycle] = {}
        # deduplication key includes optional command/path details.
        self._seen: set[tuple[str, str, str, str, str, str]] = set()
        self._first_event_received = False

    def parse_line(self, line: str) -> str | None:
        """
        Parse one line of Codex JSONL output.

        Returns:
          - Human-readable text for show-output log, or None to suppress line
          - Raises: does not raise; failed parses are logged and return None
        """
        stripped = line.strip()
        if not stripped:
            return None

        # Attempt JSON parse
        try:
            loaded: Any = json.loads(stripped)
        except json.JSONDecodeError:
            # Not a JSON line - pass through as raw text for show-output
            return line

        if not isinstance(loaded, dict):
            return line
        event = cast(dict[str, Any], loaded)

        # Mark that we received at least one JSON event
        prefix = ""
        if not self._first_event_received:
            self._first_event_received = True
            # Write diagnostic: native JSON events are being received
            self._write_diagnostic("codex_native_json_active", "Codex native JSON event stream is active")
            prefix = _block("[system] Codex native JSON event stream is active")

        # Route by event type - try multiple candidate field names
        event_type = self._get_event_type(event)
        handler_name = self._ROUTE.get(event_type, "_handle_unknown")
        handler = cast(Callable[[dict[str, Any]], str | None], getattr(self, handler_name, self._handle_unknown))
        rendered = handler(event)
        return prefix + rendered if rendered else prefix or None

    def _get_event_type(self, event: dict[str, Any]) -> str:
        # Try multiple candidate field names for event type
        for key in ("type", "event", "event_type"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
        return ""

    def _handle_unknown(self, event: dict[str, Any]) -> str:
        event_type = self._get_event_type(event) or "unknown"
        return _block(f"[system] Skipped unsupported Codex JSON event type: {event_type}")

    def _handle_item_started(self, event: dict[str, Any]) -> str | None:
        item = _event_item(event)
        if item is None:
            return self._handle_unknown(event)
        item_type = _string(item.get("type"))
        item_id = _string(item.get("id"))
        timestamp = self._get_timestamp(event)

        if item_type == "command_execution":
            raw_command = _string(item.get("command"))
            tool_input = {"command": raw_command}
            self._buffer_tool_start(
                tool_use_id=item_id,
                tool="exec",
                timestamp=timestamp,
                tool_input=tool_input,
                start_counter=time.perf_counter(),
            )
            self._write_trace_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "start",
                "tool": "exec",
                "tool_use_id": item_id,
                "tool_input": tool_input,
                "start_time": timestamp,
                "status": "started",
                "summary": self._summarize_command(raw_command),
                "source": "codex_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "role": self._role,
            })
            return _block(
                f"[tool:start] exec {item_id or 'unknown'}",
                f"  command: {_one_line_excerpt(raw_command, limit=1000)}",
            )

        if item_type == "file_change":
            changes = _item_changes(item)
            tool_input = {"changes": changes}
            self._buffer_tool_start(
                tool_use_id=item_id,
                tool="file_change",
                timestamp=timestamp,
                tool_input=tool_input,
                start_counter=time.perf_counter(),
            )
            self._write_trace_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "start",
                "tool": "file_change",
                "tool_use_id": item_id,
                "tool_input": tool_input,
                "start_time": timestamp,
                "status": "started",
                "summary": self._summarize_changes(changes),
                "source": "codex_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "role": self._role,
            })
            return _block(
                f"[tool:start] file_change {item_id or 'unknown'}",
                f"  changes: {self._summarize_changes(changes)}",
            )

        return self._render_item_message(item)

    def _handle_item_completed(self, event: dict[str, Any]) -> str | None:
        item = _event_item(event)
        if item is None:
            return self._handle_unknown(event)
        item_type = _string(item.get("type"))
        item_id = _string(item.get("id"))
        timestamp = self._get_timestamp(event)

        if item_type == "command_execution":
            pending = self._pending.pop(item_id, None) if item_id else None
            raw_command = _string(item.get("command"))
            if not raw_command and pending is not None and isinstance(pending.tool_input, dict):
                raw_command = _string(pending.tool_input.get("command"))
            start_time = pending.start_time if pending is not None else timestamp
            duration_ms = self._duration_from_pending(pending, timestamp, start_time)
            return_code = item.get("exit_code")
            status = self._command_status(item)
            output = _string(item.get("aggregated_output"))
            tool_input = {"command": raw_command}

            self._write_trace_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "end",
                "tool": "exec",
                "tool_use_id": item_id,
                "tool_input": tool_input,
                "start_time": start_time,
                "end_time": timestamp,
                "duration_ms": duration_ms,
                "duration_source": "runner_clock",
                "status": status,
                "return_code": return_code,
                "summary": self._summarize_command(raw_command),
                "source": "codex_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "role": self._role,
            })
            self._derive_command_event(
                raw_command=raw_command,
                tool_use_id=item_id,
                start_time=start_time,
                end_time=timestamp,
                duration_ms=duration_ms,
                status=status,
                return_code=return_code,
                stdout=output,
            )
            self._derive_file_access_events(
                raw_command=raw_command,
                tool_use_id=item_id,
                start_time=start_time,
                end_time=timestamp,
                duration_ms=duration_ms,
                status=status,
            )
            return self._render_completed_command(
                raw_command,
                status,
                duration_ms,
                output,
                tool_use_id=item_id,
                return_code=return_code if isinstance(return_code, int) else None,
            )

        if item_type == "file_change":
            pending = self._pending.pop(item_id, None) if item_id else None
            start_time = pending.start_time if pending is not None else timestamp
            duration_ms = self._duration_from_pending(pending, timestamp, start_time)
            changes = _item_changes(item)
            status = _string(item.get("status")) or "completed"
            normalized_status = "ok" if status == "completed" else status
            self._write_trace_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "end",
                "tool": "file_change",
                "tool_use_id": item_id,
                "tool_input": {"changes": changes},
                "start_time": start_time,
                "end_time": timestamp,
                "duration_ms": duration_ms,
                "duration_source": "runner_clock",
                "status": normalized_status,
                "summary": self._summarize_changes(changes),
                "source": "codex_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "role": self._role,
            })
            self._derive_edit_events(
                changes=changes,
                tool_use_id=item_id,
                start_time=start_time,
                end_time=timestamp,
                duration_ms=duration_ms,
                status=normalized_status,
            )
            return _block(
                f"[tool:end] file_change {item_id or 'unknown'} {normalized_status} "
                f"in {duration_ms}ms rc=unknown",
                f"  changes: {self._summarize_changes(changes)}",
            )

        return self._render_item_message(item)

    def _render_item_message(self, item: dict[str, Any]) -> str | None:
        if item.get("type") != "agent_message":
            return None
        text = item.get("text")
        if not isinstance(text, str) or not text:
            return None
        text = text.strip()
        if not text:
            return None
        return _block(text)

    def _buffer_tool_start(
        self,
        *,
        tool_use_id: str,
        tool: str,
        timestamp: str,
        tool_input: dict[str, Any] | None,
        start_counter: float | None = None,
    ) -> None:
        if not tool_use_id:
            return
        self._pending[tool_use_id] = _ToolLifecycle(
            tool_use_id=tool_use_id,
            tool=tool,
            start_time=timestamp,
            tool_input=tool_input,
            start_counter=start_counter,
            phase="start",
        )

    def _handle_tool_start(self, event: dict[str, Any]) -> str:
        tool_use_id = self._get_field(event, "tool_use_id", "id", "call_id") or ""
        tool = self._get_field(event, "tool", "tool_name", "name") or "unknown"
        timestamp = self._get_timestamp(event)

        human = _block(f"[tool:start] {tool} {tool_use_id or 'unknown'}")

        if tool_use_id:
            self._buffer_tool_start(
                tool_use_id=tool_use_id,
                tool=tool,
                timestamp=timestamp,
                tool_input=None,
            )

        # Write start event to trace
        self._write_trace_event({
            "schema_version": 1,
            "type": "tool_call",
            "phase": "start",
            "tool": tool,
            "tool_use_id": tool_use_id,
            "start_time": timestamp,
            "status": "started",
            "source": "codex_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })
        return human

    def _handle_tool_end(self, event: dict[str, Any]) -> str:
        tool_use_id = self._get_field(event, "tool_use_id", "id", "call_id") or ""
        tool = self._get_field(event, "tool", "tool_name", "name") or "unknown"
        timestamp = self._get_timestamp(event)
        status = self._get_field(event, "status", "result", "outcome") or "unknown"
        return_code = self._get_field(event, "return_code", "exit_code", "exitStatus")

        # Compute duration from buffered start event
        duration_ms = 0
        start_time = timestamp
        if tool_use_id in self._pending:
            pending = self._pending.pop(tool_use_id)
            duration_ms = self._duration_ms(timestamp, pending.start_time)
            start_time = pending.start_time

        rc_text = str(return_code) if return_code is not None else "unknown"
        human = _block(
            f"[tool:end] {tool} {tool_use_id or 'unknown'} "
            f"{status} in {duration_ms}ms rc={rc_text}"
        )

        # Write end event to trace
        self._write_trace_event({
            "schema_version": 1,
            "type": "tool_call",
            "phase": "end",
            "tool": tool,
            "tool_use_id": tool_use_id,
            "start_time": start_time,
            "end_time": timestamp,
            "duration_ms": duration_ms,
            "duration_source": "codex_native_json",
            "status": status,
            "return_code": return_code,
            "source": "codex_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })

        # Also derive command/file_access/edit events
        self._maybe_derive_command_event(event, tool, tool_use_id, start_time, timestamp, duration_ms, status, return_code)

        return human

    def _duration_from_pending(self, pending: _ToolLifecycle | None, end_time: str, start_time: str) -> int:
        if pending is not None and pending.start_counter is not None:
            return max(0, int((time.perf_counter() - pending.start_counter) * 1000))
        return self._duration_ms(end_time, start_time)

    def _command_status(self, item: dict[str, Any]) -> str:
        exit_code = item.get("exit_code")
        if isinstance(exit_code, int):
            return "ok" if exit_code == 0 else "error"
        status = _string(item.get("status"))
        if status == "completed":
            return "ok"
        if status:
            return status
        return "unknown"

    def _render_completed_command(
        self,
        raw_command: str,
        status: str,
        duration_ms: int,
        output: str,
        *,
        tool_use_id: str,
        return_code: int | None,
    ) -> str:
        del raw_command
        rc_text = str(return_code) if return_code is not None else "unknown"
        lines: list[str] = [
            f"[tool:end] exec {tool_use_id or 'unknown'} {status} in {duration_ms}ms rc={rc_text}",
        ]
        if output:
            lines.extend(_excerpt_lines("stdout", output))
        return _block(*lines)

    def _maybe_derive_command_event(
        self,
        event: dict[str, Any],
        tool: str,
        tool_use_id: str,
        start_time: str,
        end_time: str,
        duration_ms: int,
        status: str,
        return_code: Any,
    ) -> None:
        """If tool is Bash/exec, derive a command event from tool_input."""
        if tool not in {"Bash", "exec", "shell", "Command"}:
            return

        raw_tool_input: Any = event.get("tool_input")
        if raw_tool_input is None:
            raw_tool_input = event.get("input")
        if raw_tool_input is None:
            raw_tool_input = {}
        if isinstance(raw_tool_input, dict):
            tool_input = cast(dict[str, Any], raw_tool_input)
            raw_command_value = tool_input.get("command") or tool_input.get("cmd") or ""
            raw_command = raw_command_value if isinstance(raw_command_value, str) else ""
        else:
            raw_command = str(raw_tool_input)

        if not raw_command:
            return

        unwrapped = self._unwrap_powershell(raw_command)
        command_kind = self._classify_command(unwrapped)
        remote = self._extract_remote(unwrapped)

        self._write_trace_event({
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
            "source": "codex_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })

    def _derive_command_event(
        self,
        *,
        raw_command: str,
        tool_use_id: str,
        start_time: str,
        end_time: str,
        duration_ms: int,
        status: str,
        return_code: Any,
        stdout: str = "",
    ) -> None:
        if not raw_command:
            return

        unwrapped = self._unwrap_powershell(raw_command)
        command_kind = self._classify_command(unwrapped)
        remote = self._extract_remote(unwrapped)

        self._write_trace_event({
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
            "duration_source": "runner_clock",
            "return_code": return_code,
            "status": status,
            "stdout_excerpt": self._excerpt(stdout),
            "source": "codex_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
        })

    def _derive_file_access_events(
        self,
        *,
        raw_command: str,
        tool_use_id: str,
        start_time: str,
        end_time: str,
        duration_ms: int,
        status: str,
    ) -> None:
        command = self._unwrap_powershell(raw_command)
        for raw_path in self._candidate_read_paths(command):
            resolved = self._resolve_path(raw_path)
            if resolved is None:
                continue
            display_path = self._display_path(resolved)
            path_class, skill_name = self._classify_path(display_path)
            event: dict[str, Any] = {
                "schema_version": 1,
                "type": "file_access",
                "phase": "end",
                "tool_use_id": tool_use_id,
                "action": "read",
                "path": display_path,
                "path_class": path_class,
                "skill_name": skill_name,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "duration_source": "runner_clock",
                "status": status,
                "source": "codex_native_json",
                "confidence": "medium",
                "run_id": self._run_id,
                "role": self._role,
            }
            try:
                if resolved.is_file():
                    event["bytes"] = resolved.stat().st_size
            except OSError:
                pass
            self._write_trace_event(event)

    def _derive_edit_events(
        self,
        *,
        changes: list[dict[str, Any]],
        tool_use_id: str,
        start_time: str,
        end_time: str,
        duration_ms: int,
        status: str,
    ) -> None:
        for change in changes:
            raw_path = _string(change.get("path"))
            if not raw_path:
                continue
            resolved = self._resolve_path(raw_path)
            display_path = self._display_path(resolved) if resolved is not None else raw_path
            self._write_trace_event({
                "schema_version": 1,
                "type": "edit",
                "phase": "end",
                "tool_use_id": tool_use_id,
                "path": display_path,
                "edit_kind": self._classify_edit_path(display_path),
                "change_kind": _string(change.get("kind")) or "unknown",
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "duration_source": "runner_clock",
                "status": status,
                "source": "codex_native_json",
                "confidence": "high",
                "run_id": self._run_id,
                "role": self._role,
            })

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
            return "remote_bench" if "ssh" in lower or remote_from_command(command) else "benchmark"
        if "msprof" in lower or "profile export" in lower:
            return "profile"
        remote = self._extract_remote(command)
        if remote:
            return "remote_command"
        return "local_command"

    def _unwrap_powershell(self, command: str) -> str:
        """
        Detect and strip PowerShell wrapper from raw_command.

        e.g.: "powershell.exe" -Command "python ./.codex/skills/.../run-command.py run-bench ..."
        -> "python ./.codex/skills/.../run-command.py run-bench ..."
        """
        ps_re = re.compile(
            r"""^"([^"]*\\powershell\.exe)"\s+-Command\s+(?P<quote>['"])(?P<inner>.+)(?P=quote)$""",
            re.IGNORECASE,
        )
        match = ps_re.match(command.strip())
        if match:
            inner = match.group("inner")
            # Strip outermost quotes if the inner command is quoted
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

    def _candidate_read_paths(self, command: str) -> list[str]:
        paths: list[str] = []
        get_content_re = re.compile(
            r"""(?ix)
            \bGet-Content\b
            (?:
                \s+
                (?:
                    -Path|-LiteralPath
                    |-[A-Za-z]+(?:\s+\S+)?
                )
            )*
            \s+
            (?P<path>'[^']+'|"[^"]+"|[^\s;)]+)
            """
        )
        for match in get_content_re.finditer(command):
            paths.append(_strip_shell_quotes(match.group("path")))

        read_command_re = re.compile(
            r"""(?ix)
            \b(?:cat|type|head|tail|more|less)\b
            (?:\s+-\S+)*
            \s+
            (?P<path>'[^']+'|"[^"]+"|[^\s;)]+)
            """
        )
        for match in read_command_re.finditer(command):
            paths.append(_strip_shell_quotes(match.group("path")))

        return _dedupe(paths)

    def _resolve_path(self, raw_path: str) -> Path | None:
        if not raw_path or any(char in raw_path for char in "*?{}"):
            return None
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            workspace_root = self._workspace_path()
            path = (workspace_root / path) if workspace_root is not None else path
        try:
            return path.resolve()
        except OSError:
            return path

    def _workspace_path(self) -> Path | None:
        if not self._workspace_root:
            return None
        try:
            return Path(self._workspace_root).resolve()
        except OSError:
            return Path(self._workspace_root)

    def _display_path(self, path: Path) -> str:
        workspace_root = self._workspace_path()
        if workspace_root is not None:
            try:
                return path.relative_to(workspace_root).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def _classify_path(self, display_path: str) -> tuple[str, str | None]:
        normalized = display_path.replace("\\", "/")
        parts = normalized.split("/")
        for marker in (".codex", ".opencode"):
            if len(parts) >= 4 and parts[0] == marker and parts[1] == "skills":
                skill_name = parts[2]
                if len(parts) == 4 and parts[3] == "SKILL.md":
                    return "skill_md", skill_name
                if len(parts) >= 4 and parts[3] == "references":
                    return "skill_reference", skill_name
                if len(parts) >= 4 and parts[3] == "scripts":
                    return "skill_script", skill_name
                return "skill_other", skill_name
        if "/opt-round-" in f"/{normalized}":
            return "round_artifact", None
        if normalized.startswith("baseline/"):
            return "baseline_artifact", None
        if normalized.startswith("triton-agent-logs/"):
            return "log_output", None
        return "workspace_source", None

    def _classify_edit_path(self, display_path: str) -> str:
        normalized = display_path.replace("\\", "/").lower()
        name = normalized.rsplit("/", 1)[-1]
        if "/opt-round-" in f"/{normalized}":
            return "round_artifact"
        if name.startswith(("test_", "differential_test_")):
            return "test_harness"
        if name.startswith("bench_"):
            return "bench_harness"
        if name.endswith((".md", ".txt")):
            return "documentation"
        if name.endswith((".json", ".yaml", ".yml", ".toml")):
            return "metadata"
        if name.endswith(".py"):
            return "operator"
        return "unknown"

    def _summarize_command(self, command: str) -> str:
        unwrapped = self._unwrap_powershell(command)
        return self._excerpt(unwrapped.replace("\r", " ").replace("\n", " "), limit=200)

    def _summarize_changes(self, changes: list[dict[str, Any]]) -> str:
        if not changes:
            return "file change"
        rendered: list[str] = []
        for change in changes[:3]:
            kind = _string(change.get("kind")) or "change"
            path = _string(change.get("path")) or "unknown"
            resolved = self._resolve_path(path)
            rendered_path = self._display_path(resolved) if resolved is not None else path
            rendered.append(f"{kind} {rendered_path}")
        if len(changes) > 3:
            rendered.append(f"... {len(changes) - 3} more")
        return ", ".join(rendered)

    def _excerpt(self, text: str, limit: int = 2000) -> str:
        if not text:
            return ""
        return text if len(text) <= limit else text[:limit] + "\n<truncated>"

    def _duration_ms(self, end_time_str: str, start_time_str: str) -> int:
        """Parse ISO timestamps and compute duration_ms. Returns 0 on parse failure."""
        start = _parse_timestamp(start_time_str)
        end = _parse_timestamp(end_time_str)
        if start and end:
            delta = end - start
            return max(0, int(delta.total_seconds() * 1000))
        return 0

    def _write_trace_event(self, event: dict[str, Any]) -> None:
        """Write trace event, deduplicating if already seen."""
        if self._trace_path is None:
            return
        key = (
            str(event.get("tool_use_id", "")),
            str(event.get("phase", "")),
            str(event.get("type", "")),
            str(event.get("path", "")),
            str(event.get("command", "")),
            str(event.get("change_kind", "")),
        )
        if key in self._seen:
            return
        self._seen.add(key)
        append_trace_event(self._trace_path, event)

    def _write_diagnostic(self, code: str, detail: str) -> None:
        """Write a diagnostic event to the trace."""
        if self._trace_path is None:
            return
        append_trace_event(self._trace_path, {
            "schema_version": 1,
            "type": "diagnostic",
            "phase": "instant",
            "code": code,
            "detail": detail,
            "source": "codex_native_json",
            "confidence": "high",
            "run_id": self._run_id,
            "role": self._role,
            "timestamp": utc_timestamp(),
        })

    def _get_field(self, event: dict[str, Any], *keys: str) -> Any:
        """Try multiple candidate field names, return first non-null value."""
        for key in keys:
            value = event.get(key)
            if value is not None:
                return value
        return None

    def _get_timestamp(self, event: dict[str, Any]) -> str:
        """Get timestamp from event, falling back to now()."""
        for key in ("timestamp", "time", "end_time", "start_time"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value
        return utc_timestamp()

    def flush(self) -> None:
        """Called when stream ends. Write any pending tool_start as incomplete."""
        for tool_use_id, pending in list(self._pending.items()):
            self._write_trace_event({
                "schema_version": 1,
                "type": "tool_call",
                "phase": "end",
                "tool": pending.tool,
                "tool_use_id": tool_use_id,
                "start_time": pending.start_time,
                "end_time": utc_timestamp(),
                "duration_ms": 0,
                "duration_source": "codex_native_json",
                "status": "unknown",
                "source": "codex_native_json",
                "confidence": "medium",
                "run_id": self._run_id,
                "role": self._role,
            })
        self._pending.clear()

    _ROUTE: dict[str, str] = {
        # Codex JSON event type -> handler method name
        # Keep compatibility with older hypothetical names and current item events.
        "tool_start": "_handle_tool_start",
        "tool_start_event": "_handle_tool_start",
        "tool_called": "_handle_tool_start",
        "tool_end": "_handle_tool_end",
        "tool_end_event": "_handle_tool_end",
        "tool_completed": "_handle_tool_end",
        "item.started": "_handle_item_started",
        "item.completed": "_handle_item_completed",
    }


def _parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO timestamp string to datetime, or return None."""
    if not value:
        return None
    try:
        # Support both ISO format with/without timezone
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def remote_from_command(command: str) -> bool:
    """Check if command appears to target a remote system."""
    lower = command.lower()
    return "ssh" in lower or "scp" in lower or "@" in command


def _event_item(event: dict[str, Any]) -> dict[str, Any] | None:
    item = event.get("item")
    return cast(dict[str, Any], item) if isinstance(item, dict) else None


def _item_changes(item: dict[str, Any]) -> list[dict[str, Any]]:
    raw_changes = item.get("changes")
    if not isinstance(raw_changes, list):
        return []
    changes = cast(list[Any], raw_changes)
    return [cast(dict[str, Any], change) for change in changes if isinstance(change, dict)]


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _strip_shell_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return stripped


def _block(*lines: str) -> str:
    return "\n".join(lines).rstrip() + "\n\n"


def _excerpt_lines(label: str, text: str, limit: int = 2000) -> list[str]:
    excerpt = _excerpt_text(text, limit=limit)
    if not excerpt:
        return []
    if "\n" not in excerpt:
        return [f"  {label}: {excerpt}"]
    lines = [f"  {label} excerpt:"]
    lines.extend(f"    {line}" for line in excerpt.splitlines())
    return lines


def _excerpt_text(text: str, limit: int = 2000) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").rstrip()
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(1, limit - 15)].rstrip() + "\n<truncated>"


def _one_line_excerpt(text: str, *, limit: int) -> str:
    compact = " ".join(text.replace("\r", "\n").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 15)].rstrip() + " ... <truncated>"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


class CodexJsonOutputFilter:
    """
    OutputFilter implementation that:
    1. Parses each line as Codex JSONL
    2. Writes trace events to trace.jsonl
    3. Returns human-readable text for show-output log
    """

    def __init__(
        self,
        trace_path: Path | None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self._parser = CodexJsonLineParser(trace_path, extra_env)
        self._buffer = ""

    def feed(self, text: str, *, flush: bool = False) -> str:
        self._buffer += text
        emitted: list[str] = []

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                break
            line = self._buffer[: newline_index + 1]
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
            self._parser.flush()

        return "".join(emitted)

    @property
    def parser(self) -> CodexJsonLineParser:
        return self._parser


def build_codex_trace_env(
    existing: dict[str, str] | None,
    *,
    trace_path: Path,
    run_id: str,
    role: str,
    workspace_root: Path,
) -> dict[str, str]:
    """Build environment dict with trace variables for Codex JSON capture."""
    env = dict(existing or {})
    env["TRITON_AGENT_OTEL_TRACE_PATH"] = str(trace_path)
    env["TRITON_AGENT_OTEL_RUN_ID"] = run_id
    env["TRITON_AGENT_OTEL_ROLE"] = role
    env["TRITON_AGENT_WORKSPACE_ROOT"] = str(workspace_root)
    return env
