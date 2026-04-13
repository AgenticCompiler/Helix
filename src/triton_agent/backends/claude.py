from __future__ import annotations

import sys
from typing import List, Optional, TextIO

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.process_runner import run_process
from triton_agent.prompts import build_optimize_resume_prompt
from triton_agent.verbose import emit_verbose_lines, format_command_messages


class ClaudeRunner(AgentRunner):
    def __init__(self, executable: str = "claude", stall_timeout_seconds: int = 900) -> None:
        self.executable = executable
        self.stall_timeout_seconds = stall_timeout_seconds

    def build_command(self, request: AgentRequest) -> List[str]:
        command = [self.executable]
        if not request.interact:
            command.extend(["--print", "--dangerously-skip-permissions"])
            if request.command_kind == request.command_kind.OPTIMIZE and request.no_agent_session:
                command.append("--no-session-persistence")
        command.append(request.prompt)
        return command

    def run(
        self,
        request: AgentRequest,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        command = self.build_command(request)
        if request.verbose:
            self._log_launch_command(command, stderr or sys.stderr)
        return run_process(
            command,
            str(request.workdir),
            mode=self._select_mode(request),
            stall_timeout_seconds=self.stall_timeout_seconds,
            session_id_extractor=lambda _line: None,
            stdout=stdout,
            interrupt_policy=self.interrupt_policy(request),
        )

    def resume(
        self,
        request: AgentRequest,
        summary: str,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        resumed_prompt = build_optimize_resume_prompt(
            summary,
            base_prompt=request.prompt,
            require_analysis=request.require_analysis,
            supervise=request.supervise,
        )
        return self.run(request.with_prompt(resumed_prompt), stdout=stdout, stderr=stderr)

    def _log_launch_command(self, command: List[str], stream: TextIO) -> None:
        emit_verbose_lines(stream, "agent", format_command_messages(command))

    def _select_mode(self, request: AgentRequest) -> str:
        if request.interact:
            return "interactive"
        if request.show_output:
            return "streaming"
        return "buffered"
