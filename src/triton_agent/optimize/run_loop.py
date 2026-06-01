from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.prompts import build_optimize_resume_prompt


class SupportsOptimizeRecovery(Protocol):
    def run(self, request: AgentRequest) -> AgentResult:
        ...

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        ...


class OptimizeRunLoop:
    def __init__(self, max_recovery_attempts: int = 2) -> None:
        self.max_recovery_attempts = max_recovery_attempts

    def run(self, runner: object, request: AgentRequest) -> AgentResult:
        attempt = 0
        current_request = self._with_default_optimize_role(request)
        resume_summary: str | None = None

        if not _supports_recovery_runner(runner):
            raise TypeError("runner does not implement optimize recovery")
        recovery_runner = cast(SupportsOptimizeRecovery, runner)

        return self._run_continuous_loop(
            recovery_runner,
            current_request,
            attempt=attempt,
            resume_summary=resume_summary,
        )

    def _run_continuous_loop(
        self,
        runner: SupportsOptimizeRecovery,
        current_request: AgentRequest,
        *,
        attempt: int,
        resume_summary: str | None,
    ) -> AgentResult:
        while True:
            if resume_summary is None:
                result = runner.run(current_request)
            else:
                result = runner.resume(current_request, resume_summary)
            result, current_request = self._resume_until_round_requirement_satisfied(
                runner,
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

    def _resume_until_round_requirement_satisfied(
        self,
        runner: SupportsOptimizeRecovery,
        current_request: AgentRequest,
        result: AgentResult,
    ) -> tuple[AgentResult, AgentRequest]:
        if current_request.interact:
            return result, current_request
        while result.succeeded and self._needs_more_rounds(current_request):
            round_count_before_resume = self._count_round_directories(current_request.workdir)
            summary = self._build_rounds_summary(current_request)
            result = runner.resume(current_request, summary)
            round_count_after_resume = self._count_round_directories(current_request.workdir)
            if (
                result.succeeded
                and round_count_after_resume <= round_count_before_resume
            ):
                required_rounds = cast(int, current_request.min_rounds)
                return (
                    AgentResult(
                        return_code=1,
                        stdout=result.stdout,
                        stderr=(
                            "No progress: resume exited successfully but did not create a new "
                            f"`opt-round-*` directory ({round_count_after_resume}/{required_rounds}). "
                            "Ensure each round writes a new `opt-round-*` directory before rerunning."
                        ),
                        stalled=False,
                        session_id=result.session_id,
                    ),
                    current_request,
                )
        return result, current_request

    def _build_summary(self, result: AgentResult) -> str:
        output = result.stdout.strip() or result.stderr.strip() or "No progress output was captured."
        return output[-2000:]

    def _needs_more_rounds(self, request: AgentRequest) -> bool:
        required_rounds = cast(int, request.min_rounds)
        return self._count_round_directories(request.workdir) < required_rounds

    def _count_round_directories(self, workdir: Path) -> int:
        return sum(1 for path in workdir.glob("opt-round-*") if path.is_dir())

    def _build_rounds_summary(self, request: AgentRequest) -> str:
        completed_rounds = self._count_round_directories(request.workdir)
        required_rounds = cast(int, request.min_rounds)
        return (
            f"The optimize run exited, but only {completed_rounds} round directories exist "
            f"under the workspace and at least {required_rounds} are required."
        )

    def _with_default_optimize_role(self, request: AgentRequest) -> AgentRequest:
        if (
            request.command_kind != CommandKind.OPTIMIZE
            or request.optimize_role is not None
            or request.round_mode == "continuous"
        ):
            return request
        return replace(request, optimize_role="worker")

    def _build_continue_prompt(
        self,
        base_prompt: str,
        summary: str,
        *,
        round_mode: str = "checked",
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
        enable_subagent: bool = False,
    ) -> str:
        return build_optimize_resume_prompt(
            summary,
            base_prompt=base_prompt,
            round_mode=round_mode,
            optimize_target=optimize_target,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
            enable_subagent=enable_subagent,
        )


def _supports_recovery_runner(runner: object) -> bool:
    return hasattr(runner, "run") and hasattr(runner, "resume")
