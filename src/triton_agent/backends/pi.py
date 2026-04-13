from __future__ import annotations

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest


class PiRunner(AgentRunner):
    def __init__(self, executable: str = "pi", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        command = [self.executable]
        if not request.interact:
            command.append("--print")
        command.extend(
            [
                "--thinking",
                "high",
                "--no-extensions",
                "--no-skills",
                "--skill",
                str(request.workdir / ".pi" / "skills"),
            ]
        )
        if request.command_kind != request.command_kind.OPTIMIZE or request.no_agent_session:
            command.append("--no-session")
        command.append(request.prompt)
        return command
