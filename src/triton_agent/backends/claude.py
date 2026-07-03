from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TextIO, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.backends.claude_hooks import prepare_claude_hooks
from triton_agent.backends.hook_common import cleanup_hook_stage, describe_cleanup, describe_prepare
from triton_agent.mcp import resolve_managed_mcp_servers
from triton_agent.models import AgentRequest
from triton_agent.otel_trace import trace_path_from_request
from triton_agent.verbose import emit_verbose_lines

if TYPE_CHECKING:
    from triton_agent.backends.claude_trace import ClaudeJsonOutputFilter


class ClaudeRunner(AgentRunner):
    def __init__(self, executable: str = "claude", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def supports_mcp_servers(self) -> bool:
        return True

    def build_command(self, request: AgentRequest) -> list[str]:
        command = [self.executable]
        if not request.interact:
            command.extend(["--print", "--dangerously-skip-permissions"])
            command.extend(["--output-format", "stream-json", "--verbose"])
            if sys.platform == "win32":
                command.extend(["--debug-file", "NUL"])
            else:
                command.extend(["--debug-file", "/dev/null"])
            if request.command_kind == request.command_kind.OPTIMIZE and request.no_agent_session:
                command.append("--no-session-persistence")
            if request.mcp_servers:
                command.extend(["--mcp-config", str(request.workdir / ".claude" / "mcp.json")])
        if request.enable_agent_hooks:
            command.extend(["--settings", str(request.workdir / ".claude" / "triton-agent-hooks" / "settings.json")])
        command.append(request.prompt)
        return command

    def output_filter(self, request: AgentRequest) -> "ClaudeJsonOutputFilter | None":
        if request.interact:
            return None
        trace_path = trace_path_from_request(request)
        from triton_agent.backends.claude_trace import (
            ClaudeJsonOutputFilter,
            build_claude_trace_env,
        )

        if request.log_tools and trace_path is not None:
            extra_env = build_claude_trace_env(
                request.extra_env,
                trace_path=trace_path,
                run_id=request.run_id,
                workspace_root=request.workdir,
            )
        else:
            extra_env = request.extra_env
        return ClaudeJsonOutputFilter(
            trace_path if request.log_tools else None,
            extra_env,
            run_id=request.run_id,
            workspace_root=str(request.workdir),
        )

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return _extract_claude_session_id

    @contextmanager
    def _prepare_run_context(
        self,
        request: AgentRequest,
        stderr: Optional[TextIO] = None,
    ) -> Iterator[None]:
        config_path: Path | None = None
        hook_state = None
        if request.mcp_servers:
            config_path = _write_claude_mcp_config(request)
        if request.enable_agent_hooks:
            hook_state = prepare_claude_hooks(
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


def _extract_claude_session_id(text: str) -> str | None:
    match = re.search(r'"session_id"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            event = cast(dict[str, Any], json.loads(stripped))
        except json.JSONDecodeError:
            continue
        session_id = event.get("session_id")
        if isinstance(session_id, str) and session_id:
            return session_id
    return None


def _write_claude_mcp_config(request: AgentRequest) -> Path:
    resolved = resolve_managed_mcp_servers(
        workdir=request.workdir,
        server_names=request.mcp_servers,
    )
    servers: dict[str, object] = {}
    for name, server in resolved.items():
        servers[name] = {
            "type": "http",
            "url": server["url"],
        }
    payload: dict[str, object] = {"mcpServers": servers}
    config_path = request.workdir / ".claude" / "mcp.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path
