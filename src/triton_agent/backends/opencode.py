from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, TextIO

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult


_OPENCODE_CONFIG_PATH = Path(".opencode") / "opencode.json"


def _opencode_workspace_config() -> dict[str, object]:
    permission = {"task": {"general": "deny", "explore": "deny"}}
    return {
        "$schema": "https://opencode.ai/config.json",
        "agent": {
            "build": {"mode": "primary", "permission": permission},
            "plan": {"mode": "primary", "permission": permission},
        },
    }


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
            if not request.enable_agent_hooks and not request.log_tools:
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
        if not request.enable_agent_hooks and not request.log_tools:
            command.insert(5, "--pure")
        return command

    def run(
        self,
        request: AgentRequest,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        config_path = request.workdir / _OPENCODE_CONFIG_PATH
        if config_path.exists() or config_path.is_symlink():
            warning_stream = stderr or sys.stderr
            print(
                f"Warning: Existing OpenCode workspace config detected; skipping staged config: {config_path}",
                file=warning_stream,
            )
            return super().run(request, stdout=stdout, stderr=stderr)

        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(_opencode_workspace_config(), indent=2) + "\n", encoding="utf-8")
        try:
            return super().run(request, stdout=stdout, stderr=stderr)
        finally:
            if config_path.exists() or config_path.is_symlink():
                config_path.unlink()
