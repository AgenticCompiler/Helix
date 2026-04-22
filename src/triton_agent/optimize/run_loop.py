from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.models import GateDecision, GateResult
from triton_agent.prompts import build_optimize_resume_prompt

_TRANSIENT_AGENT_FAILURE_PATTERNS = (
    "429 too many requests",
    "exceeded retry limit",
    "rate limit",
)


class SupportsOptimizeRecovery(Protocol):
    def run(self, request: AgentRequest) -> AgentResult:
        ...

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        ...


class SupportsSupervisedOptimizeAdapter(Protocol):
    def run_worker(self, request: AgentRequest) -> AgentResult:
        ...

    def run_supervisor(self, request: AgentRequest, result: AgentResult) -> GateResult:
        ...


class OptimizeRunLoop:
    def __init__(self, max_recovery_attempts: int = 2) -> None:
        self.max_recovery_attempts = max_recovery_attempts

    def run(self, runner: object, request: AgentRequest) -> AgentResult:
        attempt = 0
        current_request = self._with_default_optimize_role(request)
        resume_summary: str | None = None

        if _supports_supervised_optimize_adapter(runner):
            return self._run_supervised_loop(
                cast(SupportsSupervisedOptimizeAdapter, runner),
                current_request,
            )

        if not _supports_recovery_runner(runner):
            raise TypeError("runner does not implement optimize recovery or round gate interfaces")
        recovery_runner = cast(SupportsOptimizeRecovery, runner)

        return self._run_unsupervised_loop(
            recovery_runner,
            current_request,
            attempt=attempt,
            resume_summary=resume_summary,
        )

    def _run_unsupervised_loop(
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
            is_transient_failure = self._is_transient_agent_failure(result)
            if not (result.stalled or is_transient_failure) or attempt >= self.max_recovery_attempts:
                return result

            attempt += 1
            if is_transient_failure:
                self._sleep_before_retry(attempt)
            resume_summary = self._build_summary(result)

    def _resume_until_round_requirement_satisfied(
        self,
        runner: SupportsOptimizeRecovery,
        current_request: AgentRequest,
        result: AgentResult,
    ) -> tuple[AgentResult, AgentRequest]:
        while result.succeeded and self._needs_more_rounds(current_request):
            round_count_before_resume = self._count_round_directories(current_request.workdir)
            summary = self._build_rounds_summary(current_request)
            result = runner.resume(current_request, summary)
            round_count_after_resume = self._count_round_directories(current_request.workdir)
            if (
                result.succeeded
                and current_request.min_rounds is not None
                and round_count_after_resume <= round_count_before_resume
            ):
                return (
                    AgentResult(
                        return_code=1,
                        stdout=result.stdout,
                        stderr=(
                            "No progress: resume exited successfully but did not create a new "
                            f"`opt-round-*` directory ({round_count_after_resume}/{current_request.min_rounds}). "
                            "Ensure each round writes a new `opt-round-*` directory before rerunning."
                        ),
                        stalled=False,
                        session_id=result.session_id,
                    ),
                    current_request,
                )
        return result, current_request

    def _run_supervised_loop(
        self,
        runner: SupportsSupervisedOptimizeAdapter,
        request: AgentRequest,
    ) -> AgentResult:
        current_request = request
        while True:
            worker_attempt = 0
            while True:
                worker_result = runner.run_worker(current_request)
                if worker_result.succeeded:
                    break
                if current_request.interact or worker_attempt >= self.max_recovery_attempts:
                    return worker_result
                is_transient_worker_failure = self._is_transient_agent_failure(worker_result)
                if not (worker_result.stalled or is_transient_worker_failure):
                    return worker_result
                worker_attempt += 1
                if is_transient_worker_failure:
                    self._sleep_before_retry(worker_attempt)

            supervisor_attempt = 0
            while True:
                gate_result = runner.run_supervisor(current_request, worker_result)
                is_transient_gate_failure = self._is_transient_gate_failure(gate_result)
                if (
                    current_request.interact
                    or supervisor_attempt >= self.max_recovery_attempts
                    or not is_transient_gate_failure
                ):
                    break
                supervisor_attempt += 1
                self._sleep_before_retry(supervisor_attempt)

            if gate_result.decision == GateDecision.PASS_STOP:
                return worker_result
            if gate_result.decision in {GateDecision.PASS_CONTINUE, GateDecision.REVISE_METADATA}:
                current_request = replace(
                    current_request,
                    prompt=self._build_continue_prompt(
                        request.prompt,
                        self._build_gate_summary(gate_result),
                        supervise=current_request.supervise,
                        compiler_source_path=current_request.compiler_source_path,
                        compiler_source_commit=current_request.compiler_source_commit,
                    ),
                    optimize_role="worker",
                )
                continue
            if gate_result.decision == GateDecision.REVISE_REQUIRED:
                current_request = replace(
                    current_request,
                    prompt=self._build_continue_prompt(
                        request.prompt,
                        self._build_gate_summary(gate_result),
                        supervise=current_request.supervise,
                        compiler_source_path=current_request.compiler_source_path,
                        compiler_source_commit=current_request.compiler_source_commit,
                    ),
                    optimize_role="worker",
                )
                continue
            return AgentResult(
                return_code=1,
                stdout="",
                stderr=self._build_gate_summary(gate_result),
            )

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

    def _build_gate_summary(self, gate_result: GateResult) -> str:
        issues = "; ".join(gate_result.blocking_issues) if gate_result.blocking_issues else "none"
        return f"Gate decision: {gate_result.decision.value}. Blocking issues: {issues}"

    def _sleep_before_retry(self, retry_number: int) -> None:
        time.sleep(self._retry_delay_seconds(retry_number))

    def _retry_delay_seconds(self, retry_number: int) -> float:
        return float(2 ** (retry_number - 1))

    def _is_transient_agent_failure(self, result: AgentResult) -> bool:
        if result.stalled or result.return_code == 130:
            return False
        combined = f"{result.stdout}\n{result.stderr}".lower()
        return any(pattern in combined for pattern in _TRANSIENT_AGENT_FAILURE_PATTERNS)

    def _is_transient_gate_failure(self, gate_result: GateResult) -> bool:
        if gate_result.decision != GateDecision.HARD_FAIL:
            return False
        combined = "\n".join(gate_result.blocking_issues).lower()
        return any(pattern in combined for pattern in _TRANSIENT_AGENT_FAILURE_PATTERNS)

    def _with_default_optimize_role(self, request: AgentRequest) -> AgentRequest:
        if (
            request.command_kind != CommandKind.OPTIMIZE
            or request.optimize_role is not None
            or request.supervise != "on"
        ):
            return request
        return replace(request, optimize_role="worker")

    def _build_continue_prompt(
        self,
        base_prompt: str,
        summary: str,
        *,
        supervise: str = "on",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
    ) -> str:
        return build_optimize_resume_prompt(
            summary,
            base_prompt=base_prompt,
            supervise="off" if supervise == "off" else "on",
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )


def _supports_supervised_optimize_adapter(runner: object) -> bool:
    return hasattr(runner, "run_worker") and hasattr(runner, "run_supervisor")


def _supports_recovery_runner(runner: object) -> bool:
    return hasattr(runner, "run") and hasattr(runner, "resume")
