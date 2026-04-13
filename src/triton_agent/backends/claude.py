from __future__ import annotations

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest


class ClaudeRunner(AgentRunner):
    def __init__(self, executable: str = "claude", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        command = [self.executable]
        if not request.interact:
            command.extend(["--print", "--dangerously-skip-permissions"])
            if request.command_kind == request.command_kind.OPTIMIZE and request.no_agent_session:
                command.append("--no-session-persistence")
        command.append(request.prompt)
        return command
