from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol

from triton_agent.models import AgentRequest, AgentResult
from triton_agent.prompts import build_optimize_resume_prompt


class SupportsOptimizeRecovery(Protocol):
    def run(self, request: AgentRequest) -> AgentResult:
        ...

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        ...


class OptimizeSupervisor:
    def __init__(self, max_recovery_attempts: int = 2) -> None:
        self.max_recovery_attempts = max_recovery_attempts

    def run(self, runner: SupportsOptimizeRecovery, request: AgentRequest) -> AgentResult:
        attempt = 0
        current_request = request
        resume_summary: str | None = None

        while True:
            if resume_summary is None:
                result = runner.run(current_request)
            else:
                result = runner.resume(current_request, resume_summary)
            result, current_request = self._resume_until_round_requirement_satisfied(
                runner,
                request,
                current_request,
                result,
            )
            if result.succeeded:
                return result
            if current_request.interact:
                return result
            if not result.stalled or attempt >= self.max_recovery_attempts:
                return result

            attempt += 1
            resume_summary = self._build_summary(result)
            current_request = replace(
                current_request,
                prompt=self._build_continue_prompt(
                    request.prompt,
                    resume_summary,
                    require_analysis=request.require_analysis,
                ),
            )

    def _resume_until_round_requirement_satisfied(
        self,
        runner: SupportsOptimizeRecovery,
        base_request: AgentRequest,
        current_request: AgentRequest,
        result: AgentResult,
    ) -> tuple[AgentResult, AgentRequest]:
        while result.succeeded and self._needs_more_rounds(current_request):
            summary = self._build_rounds_summary(current_request)
            current_request = replace(
                current_request,
                prompt=self._build_continue_prompt(
                    base_request.prompt,
                    summary,
                    require_analysis=base_request.require_analysis,
                ),
            )
            result = runner.resume(current_request, summary)
        return result, current_request

    def _build_summary(self, result: AgentResult) -> str:
        output = result.stdout.strip() or result.stderr.strip() or "No progress output was captured."
        return output[-2000:]

    def _needs_more_rounds(self, request: AgentRequest) -> bool:
        if request.min_rounds is None:
            return False
        return self._count_round_directories(request.workdir) < request.min_rounds

    def _count_round_directories(self, workdir: Path) -> int:
        return sum(1 for path in workdir.glob("opt-round-*") if path.is_dir())

    def _build_rounds_summary(self, request: AgentRequest) -> str:
        completed_rounds = self._count_round_directories(request.workdir)
        required_rounds = request.min_rounds
        return (
            f"The optimize run exited, but only {completed_rounds} round directories exist "
            f"under the workspace and at least {required_rounds} are required."
        )

    def _build_continue_prompt(
        self,
        base_prompt: str,
        summary: str,
        *,
        require_analysis: bool = False,
    ) -> str:
        del base_prompt
        return build_optimize_resume_prompt(summary, require_analysis=require_analysis)
