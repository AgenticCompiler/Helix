from __future__ import annotations

from dataclasses import dataclass, replace
import re
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.optimize.checks import check_baseline, check_round
from triton_agent.skill_loader import load_skill_script_module
from triton_agent.optimize.session_artifacts import (
    OptimizeSessionArtifactsManager,
    OptimizeSessionArtifactsState,
    SharedOptimizeSessionArtifactsState,
)
from triton_agent.optimize.models import (
    BaselinePreflightResult,
    BaselinePreflightState,
    GateDecision,
    GateResult,
)
from triton_agent.optimize.pattern_reminders import (
    resolve_generic_optimize_knowledge_skill_name,
)
from triton_agent.optimize.run_loop import OptimizeRunLoop
from triton_agent.optimize.prompts import (
    build_optimize_baseline_prompt,
    build_optimize_resume_prompt,
    build_optimize_supervisor_prompt,
)
from triton_agent.optimize.pt_cleanup import cleanup_workspace_pt_files
from triton_agent.otel_trace import (
    TRACE_PATH_ENV,
    TRACE_RUN_ID_ENV,
    TRACE_ROLE_ENV,
    TRACE_WORKSPACE_ROOT_ENV,
)
from triton_agent.verbose import emit_verbose, emit_verbose_lines


def _request_enables_cann_ext_api(request: AgentRequest) -> bool:
    return request.staged_skill_names is not None and "triton-npu-cann-ext-api-patterns" in request.staged_skill_names


def _request_optimize_knowledge_skill_name(request: AgentRequest) -> str | None:
    return resolve_generic_optimize_knowledge_skill_name(
        request.staged_skill_names,
        request.staged_skill_sources,
    )


class RecoveryRunnerAdapter:
    def __init__(
        self,
        runner: AgentRunner,
        artifacts_manager: OptimizeSessionArtifactsManager,
        artifacts_state: SharedOptimizeSessionArtifactsState,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._runner = runner
        self._artifacts_manager = artifacts_manager
        self._artifacts_state = artifacts_state
        self._stdout = stdout
        self._stderr = stderr

    def run(self, request: AgentRequest) -> AgentResult:
        traced_request = self._with_trace_env(request)
        if self._stdout is None and self._stderr is None:
            result = cast(Any, self._runner).run(traced_request)
        else:
            result = cast(Any, self._runner).run(traced_request, stdout=self._stdout, stderr=self._stderr)
        self._record_session(traced_request, result)
        return result

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        traced_request = self._with_trace_env(request)
        if self._stdout is None and self._stderr is None:
            result = cast(Any, self._runner).resume(traced_request, summary)
        else:
            result = cast(Any, self._runner).resume(
                traced_request,
                summary,
                stdout=self._stdout,
                stderr=self._stderr,
            )
        self._record_session(traced_request, result)
        return result

    def _with_trace_env(self, request: AgentRequest) -> AgentRequest:
        role = request.optimize_role or "worker"
        run_id = self._artifacts_state.archive.run_id
        env = dict(request.extra_env or {})
        env[TRACE_RUN_ID_ENV] = run_id
        env[TRACE_ROLE_ENV] = role
        env[TRACE_WORKSPACE_ROOT_ENV] = str(request.workdir)
        if request.log_tools:
            env[TRACE_PATH_ENV] = str(self._artifacts_state.otel_trace_path)
        return replace(request, run_id=run_id, extra_env=env)

    def _record_session(self, request: AgentRequest, result: AgentResult) -> None:
        self._artifacts_manager.record_agent_session(
            self._artifacts_state,
            role=request.optimize_role or "worker",
            session_id=result.session_id,
            agent=request.agent_name,
        )


def execute_continuous_optimize(
    runner: AgentRunner,
    artifacts_manager: OptimizeSessionArtifactsManager,
    request: AgentRequest,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    verbose_stream: TextIO,
) -> AgentResult:
    shared_artifacts_state = artifacts_manager.prepare_continuous_session(
        request.workdir,
        operator_path=request.input_path,
        test_mode=request.test_mode or "differential",
        bench_mode=request.bench_mode or "standalone",
        agent_name=request.agent_name,
        optimize_target=request.optimize_target,
        compiler_source_path=request.compiler_source_path,
        compiler_source_commit=request.compiler_source_commit,
        enable_cann_ext_api=_request_enables_cann_ext_api(request),
        enable_subagent=request.enable_subagent,
        optimize_knowledge_skill_name=_request_optimize_knowledge_skill_name(request),
    )
    if request.verbose:
        emit_verbose_lines(
            verbose_stream,
            "agents",
            artifacts_manager.describe_prepare_continuous_session(shared_artifacts_state),
        )
    try:
        return OptimizeRunLoop().run(
            RecoveryRunnerAdapter(
                runner,
                artifacts_manager,
                shared_artifacts_state,
                stdout=stdout,
                stderr=stderr,
            ),
            request,
        )
    finally:
        if request.verbose:
            emit_verbose_lines(
                verbose_stream,
                "agents",
                artifacts_manager.describe_cleanup_continuous_session(shared_artifacts_state),
            )
        warnings = artifacts_manager.cleanup_continuous_session(shared_artifacts_state)
        for warning in warnings:
            emit_verbose(verbose_stream, "agents", warning)
        try:
            cleaned_pt = cleanup_workspace_pt_files(request.workdir)
            if request.verbose and cleaned_pt:
                emit_verbose(verbose_stream, "agents", f"cleaned up {len(cleaned_pt)} unused pt file(s): {', '.join(cleaned_pt)}")
        except Exception:
            pass


def _latest_round_dir(workdir: Path) -> Path | None:
    round_dirs = sorted(_iter_completed_round_dirs(workdir), key=lambda path: _round_sort_key(path.name))
    if not round_dirs:
        return None
    return round_dirs[-1]


def _count_round_directories(workdir: Path) -> int:
    return len(_iter_completed_round_dirs(workdir))


def _round_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"opt-round-(\d+)$", name)
    if match is None:
        return (-1, name)
    return (int(match.group(1)), name)


def _next_round_name(latest_round_name: str | None) -> str | None:
    if latest_round_name is None:
        return None
    match = re.match(r"opt-round-(\d+)$", latest_round_name)
    if match is None:
        return None
    return f"opt-round-{int(match.group(1)) + 1}"


def _parse_supervisor_decision_from_report(report_content: str) -> str | None:
    match = re.search(r"^Decision:\s*(\S+)", report_content, re.MULTILINE)
    if match is None:
        return None
    return match.group(1)


def _normalize_supervisor_decision_value(parsed_decision: str) -> str:
    if parsed_decision in {"pass-stop", "pass-continue"}:
        return GateDecision.PASS.value
    return parsed_decision


def _parse_supervisor_blocking_issues_from_report(report_content: str) -> tuple[str, ...]:
    match = re.search(r"^Blocking issues:\s*(.+?)\s*$", report_content, re.MULTILINE)
    if match is None:
        return ()
    raw_value = match.group(1).strip()
    if not raw_value or raw_value.lower() == "none":
        return ()
    return tuple(issue.strip() for issue in raw_value.split(",") if issue.strip())


def _iter_completed_round_dirs(workdir: Path) -> tuple[Path, ...]:
    module = load_skill_script_module(
        "triton-npu-optimize-submit-round",
        "optimize_submit_round",
    )
    return tuple(cast(tuple[Path, ...], module.iter_completed_round_directories(workdir)))


@dataclass(frozen=True)
class _SupervisedPassOutcome:
    terminal_result: AgentResult | None
    continuation_summary: str
    repair_attempts: int
    active_repair_round: str | None
    continue_immediately: bool


def execute_multi_invocation_optimize(
    runner: AgentRunner,
    artifacts_manager: OptimizeSessionArtifactsManager,
    request: AgentRequest,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    verbose_stream: TextIO,
) -> AgentResult:
    if request.round_mode == "supervised":
        artifacts_state = artifacts_manager.prepare_supervised_session(
            request.workdir,
            agent_name=request.agent_name,
            optimize_target=request.optimize_target,
            compiler_source_path=request.compiler_source_path,
            compiler_source_commit=request.compiler_source_commit,
            enable_cann_ext_api=_request_enables_cann_ext_api(request),
            enable_subagent=request.enable_subagent,
            optimize_knowledge_skill_name=_request_optimize_knowledge_skill_name(request),
        )
        describe_prepare = artifacts_manager.describe_prepare_supervised_session
        describe_cleanup = artifacts_manager.describe_cleanup_supervised_session
        cleanup_session = artifacts_manager.cleanup_supervised_session
    else:
        artifacts_state = artifacts_manager.prepare_checked_session(
            request.workdir,
            agent_name=request.agent_name,
            optimize_target=request.optimize_target,
            compiler_source_path=request.compiler_source_path,
            compiler_source_commit=request.compiler_source_commit,
            enable_cann_ext_api=_request_enables_cann_ext_api(request),
            enable_subagent=request.enable_subagent,
            optimize_knowledge_skill_name=_request_optimize_knowledge_skill_name(request),
        )
        describe_prepare = artifacts_manager.describe_prepare_checked_session
        describe_cleanup = artifacts_manager.describe_cleanup_checked_session
        cleanup_session = artifacts_manager.cleanup_checked_session
    if request.verbose:
        emit_verbose_lines(
            verbose_stream,
            "agents",
            describe_prepare(artifacts_state),
        )
    try:
        controller = MultiInvocationOptimizeController(
            runner,
            artifacts_manager,
            artifacts_state=artifacts_state,
            stdout=stdout,
            stderr=stderr,
            verbose_stream=verbose_stream,
        )
        baseline_result = controller.preflight_baseline(request)
        if baseline_result.state is not BaselinePreflightState.READY:
            baseline_fix_result = controller.run_baseline_phase(request, baseline_result)
            if not baseline_fix_result.succeeded:
                return baseline_fix_result
            baseline_result = controller.preflight_baseline(request)
            if baseline_result.state is not BaselinePreflightState.READY:
                return AgentResult(
                    return_code=1,
                    stdout=baseline_fix_result.stdout,
                    stderr=(
                        "baseline preflight still failed after repair attempt:\n"
                        + "\n".join(baseline_result.issues)
                    ),
                )
        return controller.run_round_loop(request)
    finally:
        if request.verbose:
            emit_verbose_lines(
                verbose_stream,
                "agents",
                describe_cleanup(artifacts_state),
            )
        warnings = cleanup_session(artifacts_state)
        for warning in warnings:
            emit_verbose(verbose_stream, "agents", warning)
        try:
            cleaned_pt = cleanup_workspace_pt_files(request.workdir)
            if request.verbose and cleaned_pt:
                emit_verbose(
                    verbose_stream,
                    "agents",
                    f"cleaned up {len(cleaned_pt)} unused pt file(s): {', '.join(cleaned_pt)}",
                )
        except Exception:
            pass


class MultiInvocationOptimizeController:
    def __init__(
        self,
        runner: AgentRunner,
        artifacts_manager: OptimizeSessionArtifactsManager,
        artifacts_state: OptimizeSessionArtifactsState,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        verbose_stream: TextIO,
        max_repair_attempts: int = 2,
    ) -> None:
        self._runner = runner
        self._artifacts_manager = artifacts_manager
        self._artifacts_state = artifacts_state
        self._stdout = stdout
        self._stderr = stderr
        self._verbose_stream = verbose_stream
        self._max_repair_attempts = max_repair_attempts

    def preflight_baseline(self, request: AgentRequest) -> BaselinePreflightResult:
        baseline_dir = request.workdir / "baseline"
        if not baseline_dir.is_dir():
            return BaselinePreflightResult(
                state=BaselinePreflightState.NEEDS_PREPARE,
                issues=("baseline/ directory does not exist",),
            )
        check_result = check_baseline(baseline_dir)
        if check_result.decision == "pass":
            return BaselinePreflightResult(
                state=BaselinePreflightState.READY,
                issues=(),
            )
        return BaselinePreflightResult(
            state=BaselinePreflightState.NEEDS_REPAIR,
            issues=check_result.issues,
        )

    def run_baseline_phase(
        self,
        request: AgentRequest,
        preflight: BaselinePreflightResult,
    ) -> AgentResult:
        emit_verbose(
            self._verbose_stream,
            "optimize",
            f"baseline preflight: {preflight.state.value}, launching baseline repair",
        )
        baseline_request = replace(
            request,
            prompt=build_optimize_baseline_prompt(
                request.input_path,
                request.output_path,
                test_mode=request.test_mode,
                bench_mode=request.bench_mode,
                target_chip=request.target_chip,
                optimize_target=request.optimize_target,
                compiler_source_path=request.compiler_source_path,
                compiler_source_commit=request.compiler_source_commit,
                enable_cann_ext_api=_request_enables_cann_ext_api(request),
                baseline_state=preflight.state.value,
                base_prompt=request.prompt,
                remote=request.remote,
                remote_workdir=request.remote_workdir,
            ),
            optimize_role="baseline",
            interact=False,
        )
        return self._run_request(baseline_request, show_output_label="baseline")

    def run_round_loop(self, request: AgentRequest) -> AgentResult:
        current_request = request
        active_repair_round: str | None = None
        repair_attempts = 0
        round_index = 0
        while True:
            round_index += 1
            round_request = replace(
                current_request,
                optimize_role="worker",
            )
            round_result = self._run_request(round_request, show_output_label=f"round-{round_index}")
            if not round_result.succeeded:
                return round_result

            round_followup = self._determine_round_followup(current_request)
            latest_round_dir = _latest_round_dir(request.workdir)
            latest_round_name = latest_round_dir.name if latest_round_dir is not None else None
            cli_followup_summary = self._build_cli_followup_summary(
                round_followup,
                latest_round_name=latest_round_name,
            )
            if round_followup.decision == GateDecision.HARD_FAIL:
                return AgentResult(
                    return_code=1,
                    stdout=round_result.stdout,
                    stderr="\n".join(round_followup.blocking_issues),
                )
            if round_followup.decision == GateDecision.REVISE_REQUIRED:
                repair_attempts, active_repair_round = self._advance_repair_attempts(
                    repair_attempts,
                    active_repair_round,
                    latest_round_name,
                )
                if repair_attempts > self._max_repair_attempts:
                    return AgentResult(
                        return_code=1,
                        stdout=round_result.stdout,
                        stderr=self._build_repair_exhausted_message(round_followup, repair_attempts),
                    )
                current_request = self._request_with_continue_prompt(
                    current_request,
                    request.prompt,
                    cli_followup_summary,
                )
                continue

            continuation_summary = cli_followup_summary
            if request.round_mode == "supervised":
                supervised_outcome = self._handle_supervised_pass(
                    current_request,
                    round_result,
                    cli_followup_summary=cli_followup_summary,
                    latest_round_name=latest_round_name,
                    repair_attempts=repair_attempts,
                    active_repair_round=active_repair_round,
                )
                if supervised_outcome.terminal_result is not None:
                    return supervised_outcome.terminal_result
                continuation_summary = supervised_outcome.continuation_summary
                repair_attempts = supervised_outcome.repair_attempts
                active_repair_round = supervised_outcome.active_repair_round
                if supervised_outcome.continue_immediately:
                    current_request = self._request_with_continue_prompt(
                        current_request,
                        request.prompt,
                        continuation_summary,
                    )
                    continue
            else:
                repair_attempts = 0
                active_repair_round = None

            if round_followup.continue_required:
                current_request = self._request_with_continue_prompt(
                    current_request,
                    request.prompt,
                    continuation_summary,
                )
                continue

            return round_result

    def _determine_round_followup(self, request: AgentRequest) -> GateResult:
        """Translate a completed worker round into loop follow-up state."""
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=("missing opt-round-* directory after round run",),
                continue_required=False,
            )

        min_rounds = cast(int, request.min_rounds)
        check_result = check_round(
            latest_round_dir,
            min_rounds=min_rounds,
            optimize_target=request.optimize_target,
        )
        if check_result.decision == "hard-fail":
            return GateResult(
                decision=GateDecision.HARD_FAIL,
                blocking_issues=check_result.issues,
                continue_required=False,
            )
        if check_result.decision == "revise-required":
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=check_result.issues,
                continue_required=False,
            )

        round_count = _count_round_directories(request.workdir)
        if round_count < min_rounds:
            issues = list(check_result.issues)
            issues.append(
                f"minimum round requirement not yet satisfied: {round_count}/{min_rounds}"
            )
            return GateResult(
                decision=GateDecision.PASS,
                blocking_issues=tuple(issues),
                continue_required=True,
            )
        return GateResult(
            decision=GateDecision.PASS,
            blocking_issues=check_result.issues,
            continue_required=False,
        )

    def _run_supervisor_pass(
        self,
        request: AgentRequest,
        worker_result: AgentResult,
        *,
        cli_followup_summary: str | None = None,
    ) -> GateResult:
        del worker_result
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=("missing opt-round-* directory after worker run",),
                continue_required=False,
            )

        supervisor_request = replace(
            request,
            prompt=build_optimize_supervisor_prompt(
                request.workdir,
                latest_round_dir=latest_round_dir,
                cli_followup_summary=cli_followup_summary,
            ),
            skill_name="triton-npu-optimize",
            optimize_role="supervisor",
            interact=False,
            no_agent_session=True,
        )
        supervisor_result = self._run_request(supervisor_request, show_output_label="supervisor")
        if not supervisor_result.succeeded:
            return GateResult(
                decision=GateDecision.HARD_FAIL,
                blocking_issues=(supervisor_result.stdout.strip() or supervisor_result.stderr.strip() or "supervisor run failed",),
                continue_required=False,
            )

        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if supervisor_report_path is None:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=("supervisor report path is not configured for this optimize session",),
                continue_required=False,
            )
        try:
            report_content = supervisor_report_path.read_text(encoding="utf-8")
        except OSError:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=("failed to read supervisor report",),
                continue_required=False,
            )
        self._snapshot_live_handoff_files()

        parsed_decision = _parse_supervisor_decision_from_report(report_content)
        if parsed_decision is None:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=("missing supervisor decision line in supervisor-report.md",),
                continue_required=False,
            )
        normalized_decision = _normalize_supervisor_decision_value(parsed_decision)
        try:
            decision = GateDecision(normalized_decision)
        except ValueError:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=(f"invalid supervisor decision `{parsed_decision}` in supervisor-report.md",),
                continue_required=False,
            )
        return GateResult(
            decision=decision,
            blocking_issues=_parse_supervisor_blocking_issues_from_report(report_content),
            continue_required=False,
        )

    def _handle_supervised_pass(
        self,
        current_request: AgentRequest,
        round_result: AgentResult,
        *,
        cli_followup_summary: str,
        latest_round_name: str | None,
        repair_attempts: int,
        active_repair_round: str | None,
    ) -> _SupervisedPassOutcome:
        supervisor_gate = self._run_supervisor_pass(
            current_request,
            round_result,
            cli_followup_summary=cli_followup_summary,
        )
        if supervisor_gate.decision == GateDecision.HARD_FAIL:
            return _SupervisedPassOutcome(
                terminal_result=AgentResult(
                    return_code=1,
                    stdout=round_result.stdout,
                    stderr="\n".join(supervisor_gate.blocking_issues),
                ),
                continuation_summary=cli_followup_summary,
                repair_attempts=repair_attempts,
                active_repair_round=active_repair_round,
                continue_immediately=False,
            )

        supervisor_summary = self._combine_continue_summaries(
            cli_followup_summary,
            self._read_supervisor_report_summary(),
        )
        if supervisor_gate.decision in {
            GateDecision.REVISE_METADATA,
            GateDecision.REVISE_REQUIRED,
        }:
            next_repair_attempts, next_active_repair_round = self._advance_repair_attempts(
                repair_attempts,
                active_repair_round,
                latest_round_name,
            )
            if next_repair_attempts > self._max_repair_attempts:
                return _SupervisedPassOutcome(
                    terminal_result=AgentResult(
                        return_code=1,
                        stdout=round_result.stdout,
                        stderr=self._build_repair_exhausted_message(
                            supervisor_gate,
                            next_repair_attempts,
                        ),
                    ),
                    continuation_summary=supervisor_summary,
                    repair_attempts=next_repair_attempts,
                    active_repair_round=next_active_repair_round,
                    continue_immediately=False,
                )
            # The supervisor accepted the technical facts but requires a
            # metadata or workflow repair before the next worker round.
            return _SupervisedPassOutcome(
                terminal_result=None,
                continuation_summary=supervisor_summary,
                repair_attempts=next_repair_attempts,
                active_repair_round=next_active_repair_round,
                continue_immediately=True,
            )

        return _SupervisedPassOutcome(
            terminal_result=None,
            continuation_summary=supervisor_summary,
            repair_attempts=0,
            active_repair_round=None,
            continue_immediately=False,
        )

    def _run_request(self, request: AgentRequest, *, show_output_label: str = "") -> AgentResult:
        run_id = self._artifacts_state.archive.run_id
        env = dict(request.extra_env or {})
        env[TRACE_RUN_ID_ENV] = run_id
        env[TRACE_ROLE_ENV] = request.optimize_role or "worker"
        env[TRACE_WORKSPACE_ROOT_ENV] = str(request.workdir)
        if request.log_tools:
            env[TRACE_PATH_ENV] = str(self._artifacts_state.archive.otel_trace_path)
        request = replace(
            request,
            run_id=run_id,
            extra_env=env,
            show_output_label=show_output_label,
            no_agent_session=True,
            supervisor_report_path=self._artifacts_state.supervisor_report_path,
        )
        if self._stdout is None and self._stderr is None:
            result = cast(Any, self._runner).run(request)
        else:
            result = cast(Any, self._runner).run(request, stdout=self._stdout, stderr=self._stderr)
        self._artifacts_manager.record_agent_session(
            self._artifacts_state,
            role=request.optimize_role or "worker",
            session_id=result.session_id,
            agent=request.agent_name,
        )
        return result

    def _advance_repair_attempts(
        self,
        repair_attempts: int,
        active_repair_round: str | None,
        latest_round_name: str | None,
    ) -> tuple[int, str | None]:
        if latest_round_name is None:
            return repair_attempts + 1, active_repair_round
        if latest_round_name != active_repair_round:
            return 1, latest_round_name
        return repair_attempts + 1, active_repair_round

    def _build_cli_followup_summary(
        self,
        gate_result: GateResult,
        *,
        latest_round_name: str | None,
    ) -> str:
        next_round_name = _next_round_name(latest_round_name)
        lines = [
            "CLI round follow-up from the previous round:",
            f"- Decision: {gate_result.decision.value}",
        ]
        if latest_round_name is not None:
            lines.append(f"- Latest round: {latest_round_name}")
        if next_round_name is not None:
            lines.append(f"- Next round: {next_round_name}")
        lines.append(
            f"- Continue required: {'yes' if gate_result.continue_required else 'no'}"
        )
        if gate_result.blocking_issues:
            lines.append("- Issues:")
            lines.extend(f"  - {issue}" for issue in gate_result.blocking_issues)
        else:
            lines.append("- Issues: none")
        return "\n".join(lines)

    def _read_supervisor_report_summary(self) -> str:
        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if supervisor_report_path is None:
            return "Supervisor report from the previous round:\n(supervisor report path is unavailable)"
        try:
            report_content = supervisor_report_path.read_text(encoding="utf-8").strip()
        except OSError:
            return "Supervisor report from the previous round:\n(failed to read supervisor report)"
        if not report_content:
            return "Supervisor report from the previous round:\n(empty supervisor report)"
        return f"Supervisor report from the previous round:\n{report_content}"

    def _combine_continue_summaries(
        self,
        cli_followup_summary: str,
        supervisor_summary: str,
    ) -> str:
        return f"{cli_followup_summary}\n\n{supervisor_summary}"

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

    def _request_with_continue_prompt(
        self,
        current_request: AgentRequest,
        base_prompt: str,
        summary: str,
    ) -> AgentRequest:
        return replace(
            current_request,
            prompt=self._build_continue_prompt(
                base_prompt,
                summary,
                round_mode=current_request.round_mode,
                optimize_target=current_request.optimize_target,
                compiler_source_path=current_request.compiler_source_path,
                compiler_source_commit=current_request.compiler_source_commit,
                enable_subagent=current_request.enable_subagent,
            ),
        )

    def _build_repair_exhausted_message(
        self,
        gate_result: GateResult,
        repair_attempts: int,
    ) -> str:
        issues = "\n".join(gate_result.blocking_issues) or "none"
        return (
            "optimize repair loop made no acceptable progress after "
            f"{repair_attempts} attempt(s).\n"
            f"{issues}"
        )

    def _snapshot_live_handoff_files(self) -> None:
        history_dir = self._artifacts_state.supervisor_history_dir
        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if history_dir is None or supervisor_report_path is None:
            return
        history_dir.mkdir(parents=True, exist_ok=True)
        round_label = self._next_history_round_label(history_dir)
        report_content = supervisor_report_path.read_text(encoding="utf-8")
        (history_dir / f"{round_label}-supervisor-report.md").write_text(
            report_content,
            encoding="utf-8",
        )

    def _next_history_round_label(self, history_dir: Path) -> str:
        max_index = 0
        for path in history_dir.glob("round-*.md"):
            if not path.is_file():
                continue
            match = re.match(r"round-(\d+)-", path.name)
            if match is None:
                continue
            max_index = max(max_index, int(match.group(1)))
        return f"round-{max_index + 1:03d}"
