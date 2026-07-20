from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from pathlib import Path
from typing import Any, TextIO, cast

from helix.backends.base import AgentRunner
from helix.models import AgentRequest, AgentResult, CommandKind
from helix.optimize.checks import (
    best_completed_round_geomean_speedup,
    check_baseline,
    check_round,
    count_completed_round_directories,
    count_terminal_round_directories,
)
from helix.optimize.recovery import build_optimize_progress_probe
from helix.prompts import append_additional_user_instructions, build_prompt
from helix.skills.loader import load_skill_script_module
from helix.optimize.session_artifacts import (
    OptimizeSessionArtifactsManager,
    OptimizeSessionArtifactsState,
)
from helix.optimize.workflow_state import render_optimize_phase_summary
from helix.optimize.models import (
    BaselinePreflightResult,
    BaselinePreflightState,
)
from helix.optimize.pattern_reminders import (
    resolve_generic_optimize_knowledge_skill_name,
)
from helix.optimize.prompts import (
    build_optimize_baseline_prompt,
    build_optimize_supervisor_prompt,
)
from helix.trace.core import (
    TRACE_PATH_ENV,
    TRACE_RUN_ID_ENV,
    TRACE_WORKSPACE_ROOT_ENV,
    append_trace_event,
)
from helix.terminal.verbose import emit_verbose, emit_verbose_lines
from helix.eval.triton_runtime import triton_runtime_env, triton_runtime_prompt


def _request_enables_cann_ext_api(request: AgentRequest) -> bool:
    if request.staged_skill_names is None:
        return False
    language = getattr(request, "language", "triton") or "triton"
    cann_ext_skill = f"{language}-npu-cann-ext-api-patterns"
    return cann_ext_skill in request.staged_skill_names


def _request_optimize_knowledge_skill_name(request: AgentRequest) -> str | None:
    return resolve_generic_optimize_knowledge_skill_name(
        request.staged_skill_names,
        request.staged_skill_sources,
        language=request.language,
    )


def _request_user_prompt(request: AgentRequest) -> str | None:
    if request.user_prompt is not None:
        stripped = request.user_prompt.strip()
        return stripped or None

    marker = "Additional user instructions:"
    if marker not in request.prompt:
        return None
    extracted = request.prompt.split(marker, 1)[1].strip()
    return extracted or None


def _latest_round_dir(workdir: Path) -> Path | None:
    module = load_skill_script_module(
        "ascend-npu-optimize-state",
        "round/check",
    )
    round_dirs = sorted(
        cast(tuple[Path, ...], module.iter_terminal_round_directories(workdir)),
        key=lambda path: _round_sort_key(path.name),
    )
    if not round_dirs:
        return None
    return round_dirs[-1]


def _round_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"opt-round-(\d+)$", name)
    if match is None:
        return (-1, name)
    return (int(match.group(1)), name)


def _round_number(name: str) -> int | None:
    number, _ = _round_sort_key(name)
    if number < 0:
        return None
    return number


def _parse_supervisor_status_from_report(report_content: str) -> str | None:
    status_match = re.search(r"^Status:\s*(\S+)", report_content, re.MULTILINE)
    if status_match is not None:
        return status_match.group(1)

    decision_match = re.search(r"^Decision:\s*(\S+)", report_content, re.MULTILINE)
    if decision_match is None:
        return None
    decision = decision_match.group(1)
    if decision in {"pass", "pass-stop", "pass-continue"}:
        return "pass"
    if decision in {"revise-metadata", "revise-required", "hard-fail"}:
        return "fail"
    return decision


def _parse_supervisor_blocking_issues_from_report(report_content: str) -> tuple[str, ...]:
    match = re.search(r"^Blocking issues:\s*(.+?)\s*$", report_content, re.MULTILINE)
    if match is None:
        return ()
    raw_value = match.group(1).strip()
    if not raw_value or raw_value.lower() == "none":
        return ()
    return tuple(issue.strip() for issue in raw_value.split(",") if issue.strip())


@dataclass(frozen=True)
class _BatchCheckResult:
    summary: str
    has_failures: bool
    failed_round_numbers: tuple[int, ...] = ()
    supervisor_failed: bool = False
    terminal_result: AgentResult | None = None


@dataclass(frozen=True)
class _SupervisorCheckResult:
    payload: dict[str, object]
    terminal_result: AgentResult | None = None


@dataclass(frozen=True)
class _SessionProgress:
    terminal_round_count: int
    completed_round_count: int
    best_speedup: float | None


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
            enable_agent_hooks=request.enable_agent_hooks,
            source_operator_path=request.input_path,
            language=request.language,
            optimize_target=request.optimize_target,
            min_speedup=request.min_speedup,
            compiler_source_path=request.compiler_source_path,
            compiler_source_commit=request.compiler_source_commit,
            enable_cann_ext_api=_request_enables_cann_ext_api(request),
            enable_subagent=request.enable_subagent,
            optimize_knowledge_skill_name=_request_optimize_knowledge_skill_name(request),
            system_prompt_appendix=request.system_prompt,
        )
        describe_prepare = artifacts_manager.describe_prepare_supervised_session
        describe_cleanup = artifacts_manager.describe_cleanup_supervised_session
        cleanup_session = artifacts_manager.cleanup_supervised_session
    else:
        artifacts_state = artifacts_manager.prepare_checked_session(
            request.workdir,
            agent_name=request.agent_name,
            enable_agent_hooks=request.enable_agent_hooks,
            source_operator_path=request.input_path,
            language=request.language,
            optimize_target=request.optimize_target,
            min_speedup=request.min_speedup,
            compiler_source_path=request.compiler_source_path,
            compiler_source_commit=request.compiler_source_commit,
            enable_cann_ext_api=_request_enables_cann_ext_api(request),
            enable_subagent=request.enable_subagent,
            optimize_knowledge_skill_name=_request_optimize_knowledge_skill_name(request),
            system_prompt_appendix=request.system_prompt,
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
        if request.interact:
            return controller.run_interactive_round_request(request)
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


class MultiInvocationOptimizeController:
    _MAX_NO_PROGRESS_ATTEMPTS = 3

    def __init__(
        self,
        runner: AgentRunner,
        artifacts_manager: OptimizeSessionArtifactsManager,
        artifacts_state: OptimizeSessionArtifactsState,
        *,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        verbose_stream: TextIO,
    ) -> None:
        self._runner = runner
        self._artifacts_manager = artifacts_manager
        self._artifacts_state = artifacts_state
        self._stdout = stdout
        self._stderr = stderr
        self._verbose_stream = verbose_stream

    def preflight_baseline(self, request: AgentRequest) -> BaselinePreflightResult:
        baseline_dir = request.workdir / "baseline"
        if not baseline_dir.is_dir():
            return BaselinePreflightResult(
                state=BaselinePreflightState.NEEDS_PREPARE,
                issues=("baseline/ directory does not exist",),
            )
        check_result = check_baseline(baseline_dir)
        if check_result.status == "pass":
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
        phase_summary = render_optimize_phase_summary(self._artifacts_state.workflow_state_path)
        baseline_request = replace(
            request,
            prompt=build_optimize_baseline_prompt(
                request.input_path,
                request.output_path,
                language=request.language,
                test_mode=request.test_mode,
                bench_mode=request.bench_mode,
                target_chip=request.target_chip,
                optimize_target=request.optimize_target,
                min_speedup=request.min_speedup,
                compiler_source_path=request.compiler_source_path,
                compiler_source_commit=request.compiler_source_commit,
                enable_cann_ext_api=_request_enables_cann_ext_api(request),
                baseline_state=preflight.state.value,
                base_prompt=_request_user_prompt(request),
                remote=request.remote,
                remote_workdir=request.remote_workdir,
                workflow_phase_summary=phase_summary,
            ),
            interact=False,
            )
        return self._run_request(baseline_request, show_output_label="baseline")

    def run_round_loop(self, request: AgentRequest) -> AgentResult:
        min_rounds = cast(int, request.min_rounds)
        previous_batch_issues: str | None = None
        no_progress_attempts = 0
        loop_started = False
        batch_attempts: dict[tuple[int, int], int] = {}

        iteration = 0
        while True:
            progress_before = self._load_session_progress(request.workdir)
            if self._speedup_target_met(request, progress_before.best_speedup):
                return AgentResult(return_code=0, stdout="", stderr="")
            if progress_before.terminal_round_count >= min_rounds:
                return AgentResult(return_code=0, stdout="", stderr="")

            batch_start, batch_end = self._batch_bounds_from_terminal_rounds(
                request,
                terminal_round_count=progress_before.terminal_round_count,
            )
            iteration += 1
            batch_key = (batch_start, batch_end)
            batch_attempts[batch_key] = batch_attempts.get(batch_key, 0) + 1
            launch_attempt = batch_attempts[batch_key]
            launch_label = f"batch-{batch_start}-{batch_end}-r{launch_attempt}"
            trace_path = self._trace_path_for_launch_label(request, launch_label)
            if not loop_started:
                loop_started = True
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_loop_start",
                    min_rounds=min_rounds,
                    min_speedup=request.min_speedup,
                    max_no_progress_attempts=self._MAX_NO_PROGRESS_ATTEMPTS,
                    round_mode=request.round_mode,
                )
            self._append_optimize_trace_event(
                trace_path,
                "optimize_iteration_start",
                iteration=iteration,
                batch_start=batch_start,
                batch_end=batch_end,
                launch_attempt=launch_attempt,
                terminal_round_count_before=progress_before.terminal_round_count,
                completed_round_count_before=progress_before.completed_round_count,
                best_speedup_before=progress_before.best_speedup,
                no_progress_attempts_before=no_progress_attempts,
            )

            worker_request = self._request_with_fresh_batch_prompt(
                request,
                issues=previous_batch_issues,
                batch_start=batch_start,
                batch_end=batch_end,
            )
            round_result = self._run_request(
                worker_request,
                show_output_label=launch_label,
            )
            self._append_optimize_trace_event(
                trace_path,
                "optimize_worker_result",
                iteration=iteration,
                batch_start=batch_start,
                batch_end=batch_end,
                return_code=round_result.return_code,
                stalled=round_result.stalled,
                retryable_failure=round_result.retryable_failure,
                session_id=round_result.session_id,
            )

            try:
                progress_after = self._load_session_progress(request.workdir)
            except Exception as exc:
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_iteration_decision",
                    iteration=iteration,
                    decision="stop_failed_state_check",
                    reason=str(exc),
                    no_progress_attempts_after=no_progress_attempts,
                )
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_loop_stop",
                    final_terminal_round_count=progress_before.terminal_round_count,
                    final_completed_round_count=progress_before.completed_round_count,
                    final_best_speedup=progress_before.best_speedup,
                    final_return_code=round_result.return_code,
                    decision="stop_failed_state_check",
                )
                return AgentResult(
                    return_code=1,
                    stdout=round_result.stdout,
                    stderr=f"failed to recompute optimize session progress: {exc}",
                    stalled=round_result.stalled,
                    session_id=round_result.session_id,
                    retryable_failure=round_result.retryable_failure,
                )

            batch_check = self.check_batch_round(
                worker_request,
                batch_start=batch_start,
                batch_end=batch_end,
            )
            if batch_check.terminal_result is not None:
                return batch_check.terminal_result
            progress_made = (
                progress_after.terminal_round_count > progress_before.terminal_round_count
            )
            no_progress_attempts = 0 if progress_made else no_progress_attempts + 1
            latest_round_outcome = self._latest_round_outcome(
                progress_before=progress_before,
                progress_after=progress_after,
            )
            self._append_optimize_trace_event(
                trace_path,
                "optimize_progress_check",
                iteration=iteration,
                terminal_round_count_before=progress_before.terminal_round_count,
                terminal_round_count_after=progress_after.terminal_round_count,
                completed_round_count_before=progress_before.completed_round_count,
                completed_round_count_after=progress_after.completed_round_count,
                best_speedup_after=progress_after.best_speedup,
                latest_round_outcome=latest_round_outcome,
                progress_made=progress_made,
            )

            if self._speedup_target_met(request, progress_after.best_speedup):
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_iteration_decision",
                    iteration=iteration,
                    decision="stop_success_min_speedup",
                    reason="best completed-round speedup satisfies min_speedup",
                    no_progress_attempts_after=no_progress_attempts,
                )
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_loop_stop",
                    final_terminal_round_count=progress_after.terminal_round_count,
                    final_completed_round_count=progress_after.completed_round_count,
                    final_best_speedup=progress_after.best_speedup,
                    final_return_code=round_result.return_code,
                    decision="stop_success_min_speedup",
                )
                return AgentResult(
                    return_code=0,
                    stdout=round_result.stdout,
                    stderr=round_result.stderr,
                    session_id=round_result.session_id,
                )

            if progress_after.terminal_round_count >= min_rounds:
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_iteration_decision",
                    iteration=iteration,
                    decision="stop_success_min_rounds",
                    reason="terminal round count satisfies min_rounds",
                    no_progress_attempts_after=no_progress_attempts,
                )
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_loop_stop",
                    final_terminal_round_count=progress_after.terminal_round_count,
                    final_completed_round_count=progress_after.completed_round_count,
                    final_best_speedup=progress_after.best_speedup,
                    final_return_code=round_result.return_code,
                    decision="stop_success_min_rounds",
                )
                return AgentResult(
                    return_code=0,
                    stdout=round_result.stdout,
                    stderr=round_result.stderr,
                    session_id=round_result.session_id,
                )

            if not progress_made and no_progress_attempts >= self._MAX_NO_PROGRESS_ATTEMPTS:
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_iteration_decision",
                    iteration=iteration,
                    decision="stop_failed_no_progress_limit",
                    reason="no new terminal round progress",
                    no_progress_attempts_after=no_progress_attempts,
                )
                self._append_optimize_trace_event(
                    trace_path,
                    "optimize_loop_stop",
                    final_terminal_round_count=progress_after.terminal_round_count,
                    final_completed_round_count=progress_after.completed_round_count,
                    final_best_speedup=progress_after.best_speedup,
                    final_return_code=round_result.return_code,
                    decision="stop_failed_no_progress_limit",
                )
                return AgentResult(
                    return_code=1,
                    stdout=round_result.stdout,
                    stderr=self._build_no_progress_limit_message(
                        attempts=no_progress_attempts,
                        batch_summary=batch_check.summary if batch_check.has_failures else None,
                        last_error=round_result.stderr or round_result.stdout,
                    ),
                    stalled=round_result.stalled,
                    session_id=round_result.session_id,
                    retryable_failure=round_result.retryable_failure,
                )

            next_batch_start = progress_after.terminal_round_count + 1
            previous_batch_issues = self._continue_followup_summary(
                batch_check,
                next_batch_start=next_batch_start,
            )
            self._append_optimize_trace_event(
                trace_path,
                "optimize_iteration_decision",
                iteration=iteration,
                decision="continue",
                reason="progress not yet sufficient to stop",
                no_progress_attempts_after=no_progress_attempts,
            )

    def run_interactive_round_request(self, request: AgentRequest) -> AgentResult:
        worker_request = self._request_with_fresh_batch_prompt(
            request,
            issues=None,
            batch_start=request.current_round,
            batch_end=request.final_round,
        )
        return self._run_request(
            worker_request,
            show_output_label=f"batch-{worker_request.current_round}-{worker_request.final_round}-r1",
        )

    def check_batch_round(
        self,
        request: AgentRequest,
        *,
        batch_start: int,
        batch_end: int,
    ) -> _BatchCheckResult:
        lines: list[str] = []
        has_failures = False
        failed_round_numbers: list[int] = []
        supervisor_failed = False

        for round_number in range(batch_start, batch_end + 1):
            round_name = f"opt-round-{round_number}"
            round_dir = request.workdir / round_name
            if not round_dir.is_dir():
                payload = {
                    "kind": "round",
                    "status": "fail",
                    "issues": ["missing round directory"],
                    "summary": "round check requires fixes: missing round directory",
                }
                status = "fail"
            else:
                if request.min_speedup is not None:
                    check_result = check_round(
                        round_dir,
                        current_round=round_number,
                        final_round=batch_end,
                        optimize_target=request.optimize_target,
                        min_speedup=request.min_speedup,
                    )
                else:
                    check_result = check_round(
                        round_dir,
                        current_round=round_number,
                        final_round=batch_end,
                        optimize_target=request.optimize_target,
                    )
                payload = self._serialize_check_result(check_result)
                status = check_result.status

            lines.append(f"{round_name}: {json.dumps(payload, ensure_ascii=True)}")
            if status == "fail":
                has_failures = True
                failed_round_numbers.append(round_number)

        if request.round_mode == "supervised":
            latest_round_dir = _latest_round_dir(request.workdir)
            latest_round_number = (
                _round_number(latest_round_dir.name) if latest_round_dir is not None else None
            )
            should_run_supervisor = (
                latest_round_number is not None
                and batch_start <= latest_round_number <= batch_end
            )
        else:
            should_run_supervisor = False

        if should_run_supervisor:
            supervisor_check = self._run_supervisor_batch(
                request,
                batch_start=batch_start,
                batch_end=batch_end,
                batch_round_summary="\n".join(lines),
            )
            if supervisor_check.terminal_result is not None:
                return _BatchCheckResult(
                    summary="\n".join(lines),
                    has_failures=True,
                    terminal_result=supervisor_check.terminal_result,
            )
            lines.append(
                f"Supervisor guidance: {json.dumps(supervisor_check.payload, ensure_ascii=True)}"
            )
            if supervisor_check.payload.get("status") == "fail":
                has_failures = True
                supervisor_failed = True

        return _BatchCheckResult(
            summary="\n".join(lines),
            has_failures=has_failures,
            failed_round_numbers=tuple(failed_round_numbers),
            supervisor_failed=supervisor_failed,
        )

    def _run_supervisor_batch(
        self,
        request: AgentRequest,
        *,
        batch_start: int,
        batch_end: int,
        batch_round_summary: str,
    ) -> _SupervisorCheckResult:
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            return _SupervisorCheckResult(
                payload={
                    "status": "fail",
                    "issues": ["missing opt-round-* directory after worker run"],
                    "summary": "missing opt-round-* directory after worker run",
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                }
            )

        supervisor_request = replace(
            request,
            prompt=build_optimize_supervisor_prompt(
                request.workdir,
                language=request.language,
                latest_round_dir=latest_round_dir,
                optimize_target=request.optimize_target,
                cli_followup_summary=batch_round_summary,
                workflow_phase_summary=render_optimize_phase_summary(
                    self._artifacts_state.workflow_state_path
                ),
            ),
            skill_name=f"{request.language}-npu-optimize",
            interact=False,
            disable_backend_retry=False,
            progress_probe=None,
        )
        supervisor_result = self._run_request(supervisor_request, show_output_label="supervisor")
        if not supervisor_result.succeeded:
            return _SupervisorCheckResult(
                payload={
                    "status": "fail",
                    "issues": [supervisor_result.stdout.strip() or supervisor_result.stderr.strip() or "supervisor run failed"],
                    "summary": "supervisor run failed",
                },
                terminal_result=AgentResult(
                    return_code=1,
                    stdout=supervisor_result.stdout,
                    stderr=supervisor_result.stderr or "supervisor run failed",
                ),
            )

        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if supervisor_report_path is None:
            return _SupervisorCheckResult(
                payload={
                    "status": "fail",
                    "issues": ["supervisor report path is not configured for this optimize session"],
                    "summary": "supervisor report path is not configured for this optimize session",
                }
            )

        try:
            report_content = supervisor_report_path.read_text(encoding="utf-8")
        except OSError:
            return _SupervisorCheckResult(
                payload={
                    "status": "fail",
                    "issues": ["failed to read supervisor report"],
                    "summary": "failed to read supervisor report",
                }
            )

        self._snapshot_live_handoff_files()
        parsed_status = _parse_supervisor_status_from_report(report_content)
        issues = list(_parse_supervisor_blocking_issues_from_report(report_content))
        if parsed_status is None:
            issues.insert(0, "missing supervisor status line in supervisor-report.md")
            status = "fail"
        elif parsed_status not in {"pass", "fail"}:
            issues.insert(0, f"invalid supervisor status `{parsed_status}` in supervisor-report.md")
            status = "fail"
        else:
            status = parsed_status

        return _SupervisorCheckResult(
            payload={
                "status": status,
                "issues": issues,
                "summary": report_content.strip() or "empty supervisor report",
                "batch_start": batch_start,
                "batch_end": batch_end,
            }
        )

    def _run_request(self, request: AgentRequest, *, show_output_label: str = "") -> AgentResult:
        run_id = self._artifacts_state.archive.run_id
        launch_label = show_output_label or "run"
        env = dict(request.extra_env or {})
        if self._artifacts_state.triton_runtime is not None:
            env = triton_runtime_env(self._artifacts_state.triton_runtime, env)
        env[TRACE_RUN_ID_ENV] = run_id
        env[TRACE_WORKSPACE_ROOT_ENV] = str(request.workdir)
        if request.log_tools:
            env[TRACE_PATH_ENV] = str(self._artifacts_state.archive.trace_path(launch_label))
        request = replace(
            request,
            run_id=run_id,
            extra_env=env,
            prompt=(
                request.prompt
                if self._artifacts_state.triton_runtime is None
                else f"{request.prompt}\n\n{triton_runtime_prompt(self._artifacts_state.triton_runtime)}"
            ),
            show_output_label=show_output_label,
            supervisor_report_path=self._artifacts_state.supervisor_report_path,
        )
        if self._stdout is None and self._stderr is None:
            result = self._runner.run(request)
        else:
            result = self._runner.run(request, stdout=self._stdout, stderr=self._stderr)
        self._artifacts_manager.record_agent_session(
            self._artifacts_state,
            label=launch_label,
            session_id=result.session_id,
            agent=request.agent_name,
        )
        return result

    def _build_worker_batch_request(
        self,
        request: AgentRequest,
        batch_start: int,
        batch_end: int,
    ) -> AgentRequest:
        phase_summary = render_optimize_phase_summary(self._artifacts_state.workflow_state_path)
        prompt = append_additional_user_instructions(
            build_prompt(
                CommandKind.OPTIMIZE,
                request.input_path,
                request.operator_path,
                request.output_path,
                request.test_mode,
                request.bench_mode,
                request.force_overwrite,
                remote=request.remote,
                remote_workdir=request.remote_workdir,
                min_rounds=request.min_rounds,
                min_speedup=request.min_speedup,
                resume_existing_session=True,
                round_mode=cast(Any, request.round_mode),
                target_chip=request.target_chip,
                optimize_target=request.optimize_target,
                language=request.language,
                compiler_source_path=request.compiler_source_path,
                compiler_source_commit=request.compiler_source_commit,
                enable_cann_ext_api=_request_enables_cann_ext_api(request),
                enable_subagent=request.enable_subagent,
                current_round=batch_start,
                final_round=batch_end,
                round_batch_size=request.round_batch_size,
                optimize_baseline_ready=not request.interact,
                workflow_phase_summary=phase_summary,
            ),
            _request_user_prompt(request),
        )
        return replace(
            request,
            prompt=prompt,
            current_round=batch_start,
            final_round=batch_end,
            disable_backend_retry=True,
            progress_probe=build_optimize_progress_probe(request.workdir),
        )

    def _request_with_fresh_batch_prompt(
        self,
        request: AgentRequest,
        *,
        issues: str | None,
        batch_start: int,
        batch_end: int,
    ) -> AgentRequest:
        worker_request = self._build_worker_batch_request(request, batch_start, batch_end)
        if issues is None:
            return worker_request
        repair_lines = [
            f"This invocation needs to complete rounds {batch_start} through {batch_end}, "
            "but before that, fix the previous batch issues.",
            "CLI batch follow-up from the previous worker batch:",
            issues,
            "Repair those issues first using the existing round directories and artifacts.",
            "Do not use an already-benchmarked round for another code-changing optimization attempt.",
            "Carry any next optimization idea into the new round range owned by this invocation.",
        ]
        return replace(
            worker_request,
            prompt=f"{worker_request.prompt}\n\n" + "\n".join(repair_lines),
        )

    def _batch_bounds_from_terminal_rounds(
        self,
        request: AgentRequest,
        *,
        terminal_round_count: int,
    ) -> tuple[int, int]:
        min_rounds = cast(int, request.min_rounds)
        next_batch_start = terminal_round_count + 1
        if request.min_speedup is not None:
            next_batch_end = min(next_batch_start, min_rounds)
        else:
            next_batch_end = min(next_batch_start + request.round_batch_size - 1, min_rounds)
        return next_batch_start, next_batch_end

    def _load_session_progress(self, workdir: Path) -> _SessionProgress:
        return _SessionProgress(
            terminal_round_count=count_terminal_round_directories(workdir),
            completed_round_count=count_completed_round_directories(workdir),
            best_speedup=best_completed_round_geomean_speedup(workdir),
        )

    def _speedup_target_met(self, request: AgentRequest, best_speedup: float | None) -> bool:
        if request.min_speedup is None or best_speedup is None:
            return False
        return best_speedup >= request.min_speedup

    def _latest_round_outcome(
        self,
        *,
        progress_before: _SessionProgress,
        progress_after: _SessionProgress,
    ) -> str:
        terminal_round_delta = (
            progress_after.terminal_round_count - progress_before.terminal_round_count
        )
        completed_round_delta = (
            progress_after.completed_round_count - progress_before.completed_round_count
        )

        if terminal_round_delta <= 0:
            return "unresolved"
        if completed_round_delta > 0 and terminal_round_delta > completed_round_delta:
            return "completed_and_rejected_terminal"
        if completed_round_delta > 0:
            return "completed"
        return "rejected_terminal"

    def _trace_path_for_launch_label(
        self,
        request: AgentRequest,
        launch_label: str,
    ) -> Path | None:
        if not request.log_tools:
            return None
        return self._artifacts_state.archive.trace_path(launch_label)

    def _append_optimize_trace_event(
        self,
        trace_path: Path | None,
        event: str,
        **payload: object,
    ) -> None:
        append_trace_event(
            trace_path,
            {
                "event": event,
                **payload,
            },
        )

    def _build_no_progress_limit_message(
        self,
        *,
        attempts: int,
        batch_summary: str | None,
        last_error: str,
    ) -> str:
        lines = [
            f"optimize produced no new terminal round progress for {attempts} consecutive worker exits.",
        ]
        if batch_summary:
            lines.append(batch_summary)
        detail = last_error.strip()
        if detail:
            lines.append(detail)
        return "\n".join(lines)

    def _continue_followup_summary(
        self,
        batch_check: _BatchCheckResult,
        *,
        next_batch_start: int,
    ) -> str | None:
        if batch_check.supervisor_failed:
            return batch_check.summary
        if any(round_number < next_batch_start for round_number in batch_check.failed_round_numbers):
            return batch_check.summary
        return None

    def _serialize_check_result(self, result: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": getattr(result, "kind"),
            "status": getattr(result, "status"),
            "issues": list(getattr(result, "issues")),
            "summary": getattr(result, "summary"),
        }
        next_option = getattr(result, "next_option", None)
        if next_option is not None:
            payload["next_option"] = next_option
        return payload

    def _snapshot_live_handoff_files(self) -> None:
        handoff_dir = self._artifacts_state.supervisor_handoff_dir
        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if handoff_dir is None or supervisor_report_path is None:
            return
        handoff_dir.mkdir(parents=True, exist_ok=True)
        round_label = self._next_handoff_round_label(handoff_dir)
        report_content = supervisor_report_path.read_text(encoding="utf-8")
        (handoff_dir / f"{round_label}-supervisor-report.md").write_text(
            report_content,
            encoding="utf-8",
        )

    def _next_handoff_round_label(self, handoff_dir: Path) -> str:
        max_index = 0
        for path in handoff_dir.glob("round-*.md"):
            if not path.is_file():
                continue
            match = re.match(r"round-(\d+)-", path.name)
            if match is None:
                continue
            max_index = max(max_index, int(match.group(1)))
        return f"round-{max_index + 1:03d}"
