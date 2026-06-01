from __future__ import annotations

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest


class TraeCLIRunner(AgentRunner):
    def __init__(self, executable: str = "traecli", stall_timeout_seconds: int | None = None) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            return [self.executable, request.prompt]
        return [self.executable, "--print", "--yolo", request.prompt]
