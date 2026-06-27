from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, TextIO, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.backends.codex_hooks import prepare_codex_hooks
from triton_agent.backends.hook_common import cleanup_hook_stage, describe_cleanup, describe_prepare
from triton_agent.mcp import resolve_managed_mcp_servers
from triton_agent.models import AgentRequest
from triton_agent.otel_trace import trace_path_from_request
from triton_agent.verbose import emit_verbose_lines

if TYPE_CHECKING:
    from triton_agent.backends.codex_trace import CodexJsonOutputFilter


class CodexRunner(AgentRunner):
    def __init__(self, executable: str = "codex", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def supports_mcp_servers(self) -> bool:
        return True

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            command = [self.executable, "--cd", str(request.workdir)]
            if request.command_kind == request.command_kind.OPTIMIZE and request.no_agent_session:
                command.append("--ephemeral")
            command.append(request.prompt)
            return command
        command = [
            self.executable,
            "exec",
            "--cd",
            str(request.workdir),
            "--skip-git-repo-check",
            "--sandbox",
            "danger-full-access",
        ]
        if request.command_kind != request.command_kind.OPTIMIZE or request.no_agent_session:
            command.append("--ephemeral")
        if request.stream_output or request.log_tools:
            command.append("--json")
        command.append(request.prompt)
        return command

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return _extract_session_id

    def output_filter(self, request: AgentRequest) -> "CodexJsonOutputFilter | _UnifiedDiffFilter | None":
        if request.interact:
            return None
        trace_path = trace_path_from_request(request)
        if request.stream_output or request.log_tools:
            from triton_agent.backends.codex_trace import CodexJsonOutputFilter, build_codex_trace_env
            extra_env = request.extra_env
            if request.log_tools and trace_path is not None:
                extra_env = build_codex_trace_env(
                    request.extra_env,
                    trace_path=trace_path,
                    run_id=request.run_id,
                    workspace_root=request.workdir,
                )
            return CodexJsonOutputFilter(
                trace_path if request.log_tools else None,
                extra_env,
                run_id=request.run_id,
                workspace_root=str(request.workdir),
            )
        return _UnifiedDiffFilter()

    @contextmanager
    def _prepare_run_context(
        self,
        request: AgentRequest,
        stderr: Optional[TextIO] = None,
    ) -> Iterator[None]:
        config_path: Path | None = None
        hook_state = None
        if request.mcp_servers:
            config_path = _write_codex_mcp_config(request)
        if request.enable_agent_hooks or request.log_tools:
            hook_state = prepare_codex_hooks(
                _hooks_root(),
                request.workdir,
                self._hook_options(request),
                extra_allowed_read_roots=self._extra_allowed_read_roots(request),
            )
            if request.verbose:
                emit_verbose_lines(stderr or sys.stderr, "hooks", describe_prepare(hook_state))
        try:
            yield
        finally:
            if hook_state is not None:
                if request.verbose:
                    emit_verbose_lines(stderr or sys.stderr, "hooks", describe_cleanup(hook_state))
                cleanup_warnings = cleanup_hook_stage(hook_state)
                if cleanup_warnings:
                    emit_verbose_lines(stderr or sys.stderr, "hooks", cleanup_warnings)
            if config_path is not None and config_path.exists():
                config_path.unlink()
                parent = config_path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()


def _hooks_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "hooks"


class _UnifiedDiffFilter:
    _HUNK_HEADER_RE = re.compile(
        r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
        r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?: .*)?$"
    )
    _DIFF_METADATA_PREFIXES = (
        "diff --git ",
        "index ",
        "--- ",
        "+++ ",
        "new file mode ",
        "deleted file mode ",
        "similarity index ",
        "rename from ",
        "rename to ",
        "old mode ",
        "new mode ",
        "Binary files ",
        "\\ No newline at end of file",
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._in_diff = False
        self._in_hunk = False
        self._old_hunk_lines_remaining = 0
        self._new_hunk_lines_remaining = 0
        self._last_line_was_hunk_content = False

    def feed(self, text: str, *, flush: bool = False) -> str:
        self._buffer += text
        emitted: list[str] = []

        while True:
            newline_index = self._buffer.find("\n")
            if newline_index == -1:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            kept = self._process_line(line)
            if kept:
                emitted.append(kept)

        if flush and self._buffer:
            kept = self._process_line(self._buffer)
            self._buffer = ""
            if kept:
                emitted.append(kept)

        return "".join(emitted)

    def _process_line(self, line: str) -> str:
        bare = line.rstrip("\r\n")
        if not self._in_diff:
            if bare.startswith("diff --git "):
                self._start_diff()
                return ""
            if self._start_hunk(bare):
                return ""
            return line

        if self._in_hunk:
            consumed = self._consume_hunk_line(bare)
            if consumed is not None:
                return consumed
            self._reset_diff_state()
            return self._process_line(line)

        if bare == "\\ No newline at end of file" and self._last_line_was_hunk_content:
            self._last_line_was_hunk_content = False
            return ""

        if bare.startswith("diff --git "):
            self._start_diff()
            return ""
        if self._start_hunk(bare):
            return ""
        if bare.startswith(self._DIFF_METADATA_PREFIXES):
            self._last_line_was_hunk_content = False
            return ""

        self._reset_diff_state()
        return line

    def _start_diff(self) -> None:
        self._in_diff = True
        self._in_hunk = False
        self._old_hunk_lines_remaining = 0
        self._new_hunk_lines_remaining = 0
        self._last_line_was_hunk_content = False

    def _reset_diff_state(self) -> None:
        self._in_diff = False
        self._in_hunk = False
        self._old_hunk_lines_remaining = 0
        self._new_hunk_lines_remaining = 0
        self._last_line_was_hunk_content = False

    def _start_hunk(self, line: str) -> bool:
        match = self._HUNK_HEADER_RE.match(line)
        if match is None:
            return False
        self._start_diff()
        self._in_hunk = True
        self._old_hunk_lines_remaining = _parse_hunk_count(match.group("old_count"))
        self._new_hunk_lines_remaining = _parse_hunk_count(match.group("new_count"))
        return True

    def _consume_hunk_line(self, line: str) -> str | None:
        if line == "\\ No newline at end of file":
            self._last_line_was_hunk_content = False
            return ""

        if line.startswith(" "):
            if self._old_hunk_lines_remaining <= 0 or self._new_hunk_lines_remaining <= 0:
                return None
            self._old_hunk_lines_remaining -= 1
            self._new_hunk_lines_remaining -= 1
        elif line.startswith("-"):
            if self._old_hunk_lines_remaining <= 0:
                return None
            self._old_hunk_lines_remaining -= 1
        elif line.startswith("+"):
            if self._new_hunk_lines_remaining <= 0:
                return None
            self._new_hunk_lines_remaining -= 1
        else:
            return None

        self._last_line_was_hunk_content = True
        if self._old_hunk_lines_remaining == 0 and self._new_hunk_lines_remaining == 0:
            self._in_hunk = False
        return ""


def _parse_hunk_count(raw_count: str | None) -> int:
    if raw_count is None:
        return 1
    return int(raw_count)


def _extract_session_id(line: str) -> Optional[str]:
    try:
        payload: object = json.loads(line)
    except json.JSONDecodeError:
        payload = None

    candidates: list[object] = []
    if isinstance(payload, Mapping):
        payload_map = cast(Mapping[str, object], payload)
        for key in ("session_id", "sessionId", "thread_id", "threadId"):
            candidates.append(payload_map.get(key))
    candidates.append(line.strip())

    for candidate in candidates:
        if not candidate:
            continue
        text = str(candidate)
        for token in text.replace('"', " ").split():
            try:
                return str(uuid.UUID(token))
            except ValueError:
                continue
    return None


def _write_codex_mcp_config(request: AgentRequest) -> Path:
    resolved = resolve_managed_mcp_servers(
        workdir=request.workdir,
        server_names=request.mcp_servers,
    )
    config_path = request.workdir / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name, server in resolved.items():
        lines.append(f"[mcp_servers.{name}]")
        lines.append(f'url = {json.dumps(cast(str, server["url"]))}')
        lines.append("")
    config_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return config_path
