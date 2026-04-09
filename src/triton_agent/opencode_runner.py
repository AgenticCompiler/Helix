from __future__ import annotations

import sys
from typing import List, Optional, TextIO

from triton_agent.agent import AgentRunner
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.process_runner import run_process
from triton_agent.prompts import build_optimize_resume_prompt
from triton_agent.verbose import emit_verbose_lines, format_command_messages


class OpenCodeRunner(AgentRunner):
    def __init__(self, executable: str = "opencode", stall_timeout_seconds: int = 900) -> None:
        self.executable = executable
        self.stall_timeout_seconds = stall_timeout_seconds

    def build_command(self, request: AgentRequest) -> List[str]:
        if request.interact:
            return [
                self.executable,
                str(request.workdir),
                "--pure",
                "--thinking",
                "--prompt",
                request.prompt,
            ]
        return [
            self.executable,
            "run",
            "--dir",
            str(request.workdir),
            "--pure",
            "--thinking",
            request.prompt,
        ]

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
            require_analysis=request.require_analysis,
        )
        return self.run(
            AgentRequest(
                command_kind=request.command_kind,
                input_path=request.input_path,
                operator_path=request.operator_path,
                output_path=request.output_path,
                test_mode=request.test_mode,
                bench_mode=request.bench_mode,
                interact=request.interact,
                verbose=request.verbose,
                show_output=request.show_output,
                force_overwrite=request.force_overwrite,
                agent_name=request.agent_name,
                skill_name=request.skill_name,
                prompt=resumed_prompt,
                workdir=request.workdir,
                min_rounds=request.min_rounds,
                continue_optimize=request.continue_optimize,
                require_analysis=request.require_analysis,
                no_agent_session=request.no_agent_session,
                staged_skill_names=request.staged_skill_names,
            ),
            stdout=stdout,
            stderr=stderr,
        )

    def _log_launch_command(self, command: List[str], stream: TextIO) -> None:
        emit_verbose_lines(stream, "agent", format_command_messages(command))

    def _select_mode(self, request: AgentRequest) -> str:
        if request.interact:
            return "interactive"
        if request.show_output:
            return "streaming"
        return "buffered"
