from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest
from triton_agent.otel_trace import trace_path_from_request

if TYPE_CHECKING:
    from triton_agent.backends.claude_trace import ClaudeJsonOutputFilter


class ClaudeRunner(AgentRunner):
    def __init__(self, executable: str = "claude", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        command = [self.executable]
        if not request.interact:
            command.extend(["--print", "--dangerously-skip-permissions"])
            if request.log_tools:
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
        if request.log_tools and trace_path is not None:
            from triton_agent.backends.claude_trace import (
                ClaudeJsonOutputFilter,
                build_claude_trace_env,
            )
            extra_env = build_claude_trace_env(
                request.extra_env,
                trace_path=trace_path,
                run_id=trace_path.parent.name,
                role=request.optimize_role or "worker",
                workspace_root=request.workdir,
            )
            return ClaudeJsonOutputFilter(trace_path, extra_env)
        return None
