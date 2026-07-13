from __future__ import annotations

from helix.backends.base import AgentRunner
from helix.models import AgentRequest


class TraeCLIRunner(AgentRunner):
    def __init__(self, executable: str = "traecli", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            return [self.executable, request.prompt]
        return [self.executable, "--print", "--yolo", request.prompt]
