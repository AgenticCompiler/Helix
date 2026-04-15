from __future__ import annotations

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


class SupportsOptimizeRoundGate(Protocol):
    def run_worker(self, request: AgentRequest) -> AgentResult:
        ...

    def run_supervisor(self, request: AgentRequest, result: AgentResult) -> GateResult:
        ...


class OptimizeSupervisor:
    def __init__(self, max_recovery_attempts: int = 2) -> None:
        self.max_recovery_attempts = max_recovery_attempts

    def run(self, runner: object, request: AgentRequest) -> AgentResult:
        # `OptimizeSupervisor` is the top-level control loop for optimize runs.
        # In unsupervised mode it repeatedly drives a single backend runner via
        # run/resume. In supervised mode it instead delegates to a round-gated
        # loop where a worker round and a supervisor audit pass alternate.
        attempt = 0
        current_request = self._with_default_optimize_role(request)
        resume_summary: str | None = None

        if _supports_round_gate_runner(runner):
            return self._run_round_gate_loop(
                cast(SupportsOptimizeRoundGate, runner),
                current_request,
            )

        if not _supports_recovery_runner(runner):
            raise TypeError("runner does not implement optimize recovery or round gate interfaces")
        recovery_runner = cast(SupportsOptimizeRecovery, runner)

        while True:
            if resume_summary is None:
                result = recovery_runner.run(current_request)
            else:
                result = recovery_runner.resume(current_request, resume_summary)
            result, current_request = self._resume_until_round_requirement_satisfied(
                recovery_runner,
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
        # `min_rounds` is enforced outside the agent itself. If the agent exits
        # successfully too early, we resume from the existing session instead of
        # pretending the optimize run is complete.
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

    def _run_round_gate_loop(
        self,
        runner: SupportsOptimizeRoundGate,
        request: AgentRequest,
    ) -> AgentResult:
        current_request = request
        while True:
            # The worker owns exactly one optimization round. Transient failures
            # retry the same round; successful completion hands off to the
            # supervisor for a fact-based audit.
            worker_attempt = 0
            while True:
                worker_result = runner.run_worker(current_request)
                if worker_result.succeeded:
                    break
                if current_request.interact or worker_attempt >= self.max_recovery_attempts:
                    return worker_result
                if not (worker_result.stalled or self._is_transient_agent_failure(worker_result)):
                    return worker_result
                worker_attempt += 1

            supervisor_attempt = 0
            while True:
                # The supervisor never edits kernels directly here; it inspects
                # the latest round artifacts and returns a gate decision that
                # decides whether the next loop iteration should continue.
                gate_result = runner.run_supervisor(current_request, worker_result)
                if (
                    current_request.interact
                    or supervisor_attempt >= self.max_recovery_attempts
                    or not self._is_transient_gate_failure(gate_result)
                ):
                    break
                supervisor_attempt += 1

            if gate_result.decision == GateDecision.PASS_STOP:
                return worker_result
            if gate_result.decision in {GateDecision.PASS_CONTINUE, GateDecision.REVISE_METADATA}:
                # For the next worker round we rebuild the prompt from the
                # original optimize prompt plus a compact gate summary, so each
                # round starts from the same base contract with fresh feedback.
                current_request = replace(
                    current_request,
                    prompt=self._build_continue_prompt(
                        request.prompt,
                        self._build_gate_summary(gate_result),
                        require_analysis=current_request.require_analysis,
                        supervise=current_request.supervise,
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
                        require_analysis=current_request.require_analysis,
                        supervise=current_request.supervise,
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
        require_analysis: bool = False,
        supervise: str = "on",
    ) -> str:
        return build_optimize_resume_prompt(
            summary,
            base_prompt=base_prompt,
            require_analysis=require_analysis,
            supervise="off" if supervise == "off" else "on",
        )


def _supports_round_gate_runner(runner: object) -> bool:
    return hasattr(runner, "run_worker") and hasattr(runner, "run_supervisor")


def _supports_recovery_runner(runner: object) -> bool:
    return hasattr(runner, "run") and hasattr(runner, "resume")
