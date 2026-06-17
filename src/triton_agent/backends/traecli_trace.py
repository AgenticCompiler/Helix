from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_THINKING_ENV = "TRITON_AGENT_SHOW_OUTPUT_THINKING"
_THINKING_MODES = {"full", "excerpt", "summary", "presence", "off"}
_DEFAULT_EXCERPT_LIMIT = 2000


@dataclass
class _Excerpt:
    text: str
    truncated: bool


@dataclass
class TraeCliShowOutputStats:
    events: int = 0
    tools: int = 0
    thinking_blocks: int = 0


class TraeCliShowOutputRenderer:
    def __init__(self, thinking_mode: str | None = None) -> None:
        self.thinking_mode = _resolve_thinking_mode(thinking_mode)
        self.stats = TraeCliShowOutputStats()

    def render_banner(self, model: str) -> str:
        model_name = model.strip() or "unknown"
        return f"> build · {model_name}\n\n"

    def render_thinking_delta(self, text: str, *, first_chunk: bool) -> str | None:
        if self.thinking_mode == "off":
            return None
        stripped = text.strip()
        if not stripped and not first_chunk:
            return None
        self.stats.thinking_blocks += 1
        if self.thinking_mode == "presence":
            if not first_chunk:
                return None
            return (
                "Thinking: [reasoning stream observed; content omitted by "
                f"{_THINKING_ENV}=presence]\n\n"
            )
        prefix = "Thinking: " if first_chunk else ""
        if self.thinking_mode == "summary" and not first_chunk:
            return None
        if self.thinking_mode == "summary" and first_chunk:
            return f"Thinking: {_summarize_text(stripped)}\n\n"
        if self.thinking_mode == "excerpt" and first_chunk:
            excerpt = _excerpt_text(stripped, limit=_DEFAULT_EXCERPT_LIMIT)
            body = excerpt.text
            if excerpt.truncated:
                body += "\nomitted: thinking truncated in show-output"
            return f"Thinking: {body}\n\n"
        return f"{prefix}{text}"

    def render_thinking_block(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        if self.thinking_mode == "off":
            return None
        if self.thinking_mode == "presence":
            return (
                "Thinking: [reasoning block observed; content omitted by "
                f"{_THINKING_ENV}=presence]\n\n"
            )
        if self.thinking_mode == "summary":
            return f"Thinking: {_summarize_text(stripped)}\n\n"
        if self.thinking_mode == "excerpt":
            excerpt = _excerpt_text(stripped, limit=_DEFAULT_EXCERPT_LIMIT)
            body = excerpt.text
            if excerpt.truncated:
                body += "\nomitted: thinking truncated in show-output"
            return f"Thinking: {body}\n\n"
        return f"Thinking: {stripped}\n\n"

    def render_tool(self, tool: str, arguments: str) -> str:
        self.stats.tools += 1
        return f"→ {_format_tool_summary(tool, arguments)}\n"

    def render_assistant_text(self, text: str) -> str | None:
        stripped = text.strip()
        if not stripped:
            return None
        return f"{stripped}\n\n"


class TraeCliJsonLineParser:
    _ROUTE: dict[str, str] = {
        "system": "_handle_system",
        "stream_event": "_handle_stream_event",
        "assistant": "_handle_assistant",
        "result": "_handle_result",
    }

    def __init__(self, extra_env: dict[str, str] | None = None) -> None:
        self._renderer = TraeCliShowOutputRenderer(extra_env.get(_THINKING_ENV) if extra_env else None)
        self._banner_emitted = False
        self._thinking_open = False
        self._streamed_reasoning = False
        self._session_id: str | None = None

    def parse_line(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped + "\n" if not stripped.startswith("{") else None
        if not isinstance(loaded, dict):
            return None
        event = cast(dict[str, Any], loaded)
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            self._session_id = session_id
        handler_name = self._ROUTE.get(str(event.get("type", "")), "_handle_unknown")
        handler = getattr(self, handler_name)
        return handler(event)

    def flush(self) -> str | None:
        if self._thinking_open:
            self._thinking_open = False
            return "\n\n"
        return None

    @property
    def stats(self) -> TraeCliShowOutputStats:
        return self._renderer.stats

    def _handle_unknown(self, _event: dict[str, Any]) -> str | None:
        return None

    def _handle_system(self, event: dict[str, Any]) -> str | None:
        if event.get("subtype") != "init" or self._banner_emitted:
            return None
        model = _string_or_empty(event.get("model"))
        self._banner_emitted = True
        return self._renderer.render_banner(model)

    def _handle_stream_event(self, event: dict[str, Any]) -> str | None:
        delta = event.get("delta")
        if not isinstance(delta, dict):
            return None
        delta = cast(dict[str, Any], delta)

        reasoning = _string_or_empty(delta.get("reasoning_content"))
        if reasoning:
            rendered = self._renderer.render_thinking_delta(
                reasoning,
                first_chunk=not self._thinking_open,
            )
            if rendered is not None:
                self._thinking_open = True
                self._streamed_reasoning = True
                return rendered
            if not self._thinking_open:
                self._thinking_open = True
                self._streamed_reasoning = True
            return reasoning

        content = _string_or_empty(delta.get("content"))
        if content:
            return self._close_thinking() + (self._renderer.render_assistant_text(content) or "")
        return None

    def _handle_assistant(self, event: dict[str, Any]) -> str | None:
        message = event.get("message")
        if not isinstance(message, dict):
            return None
        message = cast(dict[str, Any], message)

        parts: list[str] = []
        reasoning = _string_or_empty(message.get("reasoning_content"))
        if reasoning and not self._streamed_reasoning:
            rendered = self._renderer.render_thinking_block(reasoning)
            if rendered:
                parts.append(rendered)
                self._thinking_open = False

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in cast(list[Any], tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                tool_call = cast(dict[str, Any], tool_call)
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                function = cast(dict[str, Any], function)
                tool_name = _string_or_empty(function.get("name")) or "unknown"
                arguments = _string_or_empty(function.get("arguments"))
                parts.append(self._close_thinking())
                parts.append(self._renderer.render_tool(tool_name, arguments))

        content = _string_or_empty(message.get("content"))
        if content:
            parts.append(self._close_thinking())
            rendered = self._renderer.render_assistant_text(content)
            if rendered:
                parts.append(rendered)

        combined = "".join(part for part in parts if part)
        return combined or None

    def _handle_result(self, _event: dict[str, Any]) -> str | None:
        if self._thinking_open:
            self._thinking_open = False
            return "\n\n"
        return None

    def _close_thinking(self) -> str:
        if not self._thinking_open:
            return ""
        self._thinking_open = False
        return "\n\n"


class TraeCliJsonOutputFilter:
    def __init__(self, extra_env: dict[str, str] | None = None) -> None:
        self._parser = TraeCliJsonLineParser(extra_env)
        self._buffer = ""

    def feed(self, text: str, *, flush: bool = False) -> str:
        self._buffer += text
        emitted: list[str] = []

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
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
    def parser(self) -> TraeCliJsonLineParser:
        return self._parser


def _format_tool_summary(tool: str, arguments: str) -> str:
    args = _parse_tool_arguments(arguments)
    if tool == "Skill":
        skill_name = _first_string(args, "skill", "name", "skill_name")
        if skill_name:
            return f'Skill "{skill_name}"'
    if tool == "Read":
        file_path = _first_string(args, "file_path", "path")
        if file_path:
            return f"Read {_display_path(file_path)}"
    if tool in {"Bash", "bash", "shell", "PowerShell", "powershell"}:
        command = _first_string(args, "command", "cmd")
        if command:
            return f"{tool} {command}"
    if args:
        preview = json.dumps(args, ensure_ascii=False)
        if len(preview) > 120:
            preview = preview[:117] + "..."
        return f"{tool} {preview}"
    return tool


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    if not arguments.strip():
        return {}
    try:
        loaded = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    if isinstance(loaded, dict):
        return cast(dict[str, Any], loaded)
    return {}


def _first_string(args: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _display_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if ".opencode/skills/" in normalized or ".traecli/skills/" in normalized:
        marker = ".opencode/skills/" if ".opencode/skills/" in normalized else ".traecli/skills/"
        return normalized.split(marker, 1)[1]
    return Path(normalized).name or normalized


def _string_or_empty(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def _resolve_thinking_mode(value: str | None) -> str:
    raw_value = value or os.environ.get(_THINKING_ENV, "full")
    normalized = raw_value.strip().lower()
    if normalized in _THINKING_MODES:
        return normalized
    return "full"


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
        return "TraeCLI reasoning block was present but empty."
    first = _one_line_excerpt(lines[0], limit=500)
    if len(lines) == 1:
        return first
    return f"{first}\nsummary: {len(lines)} non-empty lines, {len(text)} characters"
