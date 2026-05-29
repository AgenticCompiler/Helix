from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest
from triton_agent.otel_trace import TRACE_RUN_ID_ENV, trace_path_from_request

if TYPE_CHECKING:
    from triton_agent.backends.claude_trace import ClaudeJsonOutputFilter


class ClaudeRunner(AgentRunner):
    def __init__(self, executable: str = "claude", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

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
                run_id=(request.extra_env or {}).get(TRACE_RUN_ID_ENV, ""),
                role=request.optimize_role or "worker",
                workspace_root=request.workdir,
            )
        else:
            extra_env = request.extra_env
        return ClaudeJsonOutputFilter(trace_path, extra_env)

    def session_id_extractor(self) -> Callable[[str], str | None]:
        return _extract_claude_session_id


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
