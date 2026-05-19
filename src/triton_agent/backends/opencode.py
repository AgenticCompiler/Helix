from __future__ import annotations

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest


class OpenCodeRunner(AgentRunner):
    def __init__(self, executable: str = "opencode", stall_timeout_seconds: int = 900) -> None:
        super().__init__(executable, stall_timeout_seconds)

    def build_command(self, request: AgentRequest) -> list[str]:
        if request.interact:
            command = [
                self.executable,
                str(request.workdir),
                "--prompt",
                request.prompt,
            ]
            if not request.enable_agent_hooks:
                command.insert(2, "--pure")
            return command

        command = [
            self.executable,
            "run",
            "--dir",
            str(request.workdir),
            "--dangerously-skip-permissions",
            "--thinking",
            request.prompt,
        ]
        if not request.enable_agent_hooks:
            command.insert(5, "--pure")
        return command
