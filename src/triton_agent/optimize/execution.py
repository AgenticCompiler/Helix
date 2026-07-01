from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
import time
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.optimize.agent_exit_log import write_agent_exit_log
from triton_agent.transient_failures import (
    OPTIMIZE_WORKER_RETRY_DELAYS_SECONDS,
    is_optimize_worker_retryable,
)

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult, CommandKind
from triton_agent.optimize.checks import check_baseline, check_round
from triton_agent.optimize.recovery import (
    RecoveryBudget,
    build_optimize_progress_probe,
    classify_worker_failure,
    compute_range_progress,
    render_stall_recovery_note,
    render_transient_recovery_note,
)
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.skill_loader import load_skill_script_module
from triton_agent.optimize.session_artifacts import (
    OptimizeSessionArtifactsManager,
    OptimizeSessionArtifactsState,
)
from triton_agent.optimize.workflow_state import render_optimize_phase_summary
from triton_agent.optimize.models import (
    BaselinePreflightResult,
    BaselinePreflightState,
)
from triton_agent.optimize.pattern_reminders import (
    resolve_generic_optimize_knowledge_skill_name,
)
from triton_agent.optimize.prompts import (
    build_optimize_baseline_prompt,
    build_optimize_supervisor_prompt,
)
from triton_agent.optimize.pt_cleanup import cleanup_workspace_pt_files
from triton_agent.otel_trace import (
    TRACE_PATH_ENV,
    TRACE_RUN_ID_ENV,
    TRACE_WORKSPACE_ROOT_ENV,
)
from triton_agent.verbose import emit_verbose, emit_verbose_lines


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
    round_dirs = sorted(_iter_completed_round_dirs(workdir), key=lambda path: _round_sort_key(path.name))
    if not round_dirs:
        return None
    return round_dirs[-1]


def _count_round_directories(workdir: Path) -> int:
    return len(_iter_completed_round_dirs(workdir))


def count_round_directories(workdir: Path) -> int:
    return _count_round_directories(workdir)


def _round_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"opt-round-(\d+)$", name)
    if match is None:
        return (-1, name)
    return (int(match.group(1)), name)


def _iter_completed_round_dirs(workdir: Path) -> tuple[Path, ...]:
    module = load_skill_script_module(
        "ascend-npu-optimize-state",
        "round/check",
    )
    return tuple(cast(tuple[Path, ...], module.iter_completed_round_directories(workdir)))


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
    terminal_result: AgentResult | None = None


@dataclass(frozen=True)
class _SupervisorCheckResult:
    payload: dict[str, object]
    terminal_result: AgentResult | None = None


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
            enable_agent_hooks=request.enable_agent_hooks,
            source_operator_path=request.input_path,
            language=request.language,
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
        if not request.interact:
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
    ) -> None:
        self._runner = runner
        self._artifacts_manager = artifacts_manager
        self._artifacts_state = artifacts_state
        self._stdout = stdout
        self._stderr = stderr
        self._verbose_stream = verbose_stream
        self._worker_recovery_budget = RecoveryBudget()

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
        batch_start = request.current_round
        batch_end = request.final_round
        previous_batch_issues: str | None = None

        while batch_start <= min_rounds:
            worker_request = self._request_with_fresh_batch_prompt(
                request,
                issues=previous_batch_issues,
                batch_start=batch_start,
                batch_end=batch_end,
            )

            worker_request, round_result = self._run_worker_with_recovery(
                request,
                worker_request,
                issues=previous_batch_issues,
                original_batch_start=batch_start,
            )
            if not round_result.succeeded:
                return round_result

            batch_check = self.check_batch_round(
                worker_request,
                batch_start=batch_start,
                batch_end=batch_end,
            )
            if batch_check.terminal_result is not None:
                return batch_check.terminal_result

            is_final_batch = batch_end >= min_rounds
            if batch_check.has_failures:
                if is_final_batch:
                    return AgentResult(
                        return_code=1,
                        stdout=round_result.stdout,
                        stderr=self._build_final_batch_failure_message(batch_check.summary),
                    )
                previous_batch_issues = batch_check.summary
            else:
                previous_batch_issues = None

            if is_final_batch:
                return round_result
            batch_start, batch_end = self._advance_batch_bounds(request, current_batch_end=batch_end)

        return AgentResult(return_code=0, stdout="", stderr="")

    def _run_worker_with_recovery(
        self,
        request: AgentRequest,
        worker_request: AgentRequest,
        *,
        issues: str | None,
        original_batch_start: int,
    ) -> tuple[AgentRequest, AgentResult]:
        active_request = worker_request
        attempt = 0
        while True:
            attempt += 1
            result = self._run_request(
                active_request,
                show_output_label=(
                    f"batch-{active_request.current_round}-{active_request.final_round}-r{attempt}"
                ),
            )
            if result.succeeded:
                return active_request, result

            failure_kind = classify_worker_failure(result)
            if failure_kind == "fatal":
                return active_request, result

            progress = compute_range_progress(
                request.workdir,
                batch_start=active_request.current_round,
                batch_end=active_request.final_round,
                optimize_target=request.optimize_target,
            )
            unresolved_round = progress.first_unresolved_round
            if unresolved_round > active_request.final_round:
                return active_request, AgentResult(
                    return_code=0,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    session_id=result.session_id,
                )

            self._worker_recovery_budget.consume(unresolved_round)
            if self._worker_recovery_budget.exhausted(unresolved_round):
                return active_request, AgentResult(
                    return_code=1,
                    stdout=result.stdout,
                    stderr=self._build_recovery_budget_exhausted_message(
                        unresolved_round=unresolved_round,
                        failure_kind=failure_kind,
                        last_error=result.stderr or result.stdout,
                    ),
                    stalled=result.stalled,
                    session_id=result.session_id,
                    retryable_failure=result.retryable_failure,
                )

            next_batch_start = progress.first_unresolved_round if failure_kind == "stall" else active_request.current_round
            if failure_kind == "stall":
                note = render_stall_recovery_note(
                    original_batch_start=original_batch_start,
                    last_accepted_round=progress.last_accepted_round,
                    first_unresolved_round=progress.first_unresolved_round,
                    batch_end=active_request.final_round,
                )
            else:
                note = render_transient_recovery_note(
                    batch_start=next_batch_start,
                    batch_end=active_request.final_round,
                )

            active_request = self._request_with_recovery_note(
                request,
                issues=issues,
                batch_start=next_batch_start,
                batch_end=active_request.final_round,
                note=note,
            )

    def _request_with_recovery_note(
        self,
        request: AgentRequest,
        *,
        issues: str | None,
        batch_start: int,
        batch_end: int,
        note: str,
    ) -> AgentRequest:
        worker_request = self._request_with_fresh_batch_prompt(
            request,
            issues=issues,
            batch_start=batch_start,
            batch_end=batch_end,
        )
        return replace(worker_request, prompt=f"{worker_request.prompt}\n\n{note}")

    def check_batch_round(
        self,
        request: AgentRequest,
        *,
        batch_start: int,
        batch_end: int,
    ) -> _BatchCheckResult:
        lines: list[str] = []
        has_failures = False

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

        if request.round_mode == "supervised":
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

        return _BatchCheckResult(
            summary="\n".join(lines),
            has_failures=has_failures,
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

    def _run_request_with_retry(
        self,
        request: AgentRequest,
        *,
        batch_start: int,
        batch_end: int,
    ) -> AgentResult:
        label = f"batch-{batch_start}-{batch_end}"
        result = self._run_request(request, show_output_label=label)
        if result.succeeded or not is_optimize_worker_retryable(result):
            if not result.succeeded:
                emit_verbose(
                    self._verbose_stream,
                    "optimize",
                    (
                        f"{label} failure (rc={result.return_code}, stalled={result.stalled}) "
                        "is not eligible for optimize worker retry"
                    ),
                )
            return result

        for attempt, delay in enumerate(OPTIMIZE_WORKER_RETRY_DELAYS_SECONDS, start=1):
            emit_verbose(
                self._verbose_stream,
                "optimize",
                f"{label} failure (rc={result.return_code}, stalled={result.stalled}), "
                f"retry {attempt}/{len(OPTIMIZE_WORKER_RETRY_DELAYS_SECONDS)} in {delay}s",
            )
            time.sleep(delay)
            result = self._run_request(request, show_output_label=f"{label}-retry-{attempt}")
            if result.succeeded or not is_optimize_worker_retryable(result):
                return result

        emit_verbose(
            self._verbose_stream,
            "optimize",
            f"{label} exhausted {len(OPTIMIZE_WORKER_RETRY_DELAYS_SECONDS)} retries, giving up",
        )
        return result

    def _run_request(self, request: AgentRequest, *, show_output_label: str = "") -> AgentResult:
        run_id = self._artifacts_state.archive.run_id
        launch_label = show_output_label or "run"
        env = dict(request.extra_env or {})
        env[TRACE_RUN_ID_ENV] = run_id
        env[TRACE_WORKSPACE_ROOT_ENV] = str(request.workdir)
        if request.log_tools:
            env[TRACE_PATH_ENV] = str(self._artifacts_state.archive.trace_path(launch_label))
        request = replace(
            request,
            run_id=run_id,
            extra_env=env,
            show_output_label=show_output_label,
            supervisor_report_path=self._artifacts_state.supervisor_report_path,
        )
        start_counter = time.perf_counter()
        try:
            if self._stdout is None and self._stderr is None:
                result = self._runner.run(request)
            else:
                result = self._runner.run(request, stdout=self._stdout, stderr=self._stderr)
        finally:
            try:
                cleaned_pt = cleanup_workspace_pt_files(request.workdir)
                if request.verbose and cleaned_pt:
                    emit_verbose(
                        self._verbose_stream,
                        "agents",
                        f"cleaned up {len(cleaned_pt)} unused pt file(s): {', '.join(cleaned_pt)}",
                    )
            except Exception:
                pass
        self._artifacts_manager.record_agent_session(
            self._artifacts_state,
            label=launch_label,
            session_id=result.session_id,
            agent=request.agent_name,
        )
        write_agent_exit_log(
            workdir=request.workdir,
            run_id=run_id,
            label=launch_label,
            return_code=result.return_code,
            stderr=result.stderr,
            stalled=result.stalled,
            session_id=result.session_id,
            duration_ms=int((time.perf_counter() - start_counter) * 1000),
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
            "Repair those issues first using the existing round directories and artifacts, then continue to optimize.",
        ]
        return replace(
            worker_request,
            prompt=f"{worker_request.prompt}\n\n" + "\n".join(repair_lines),
        )

    def _advance_batch_bounds(
        self,
        request: AgentRequest,
        *,
        current_batch_end: int,
    ) -> tuple[int, int]:
        min_rounds = cast(int, request.min_rounds)
        next_batch_start = current_batch_end + 1
        next_batch_end = min(next_batch_start + request.round_batch_size - 1, min_rounds)
        return next_batch_start, next_batch_end

    def _build_final_batch_failure_message(self, batch_summary: str) -> str:
        return (
            "optimize batch check failed after the final scheduled batch.\n"
            f"{batch_summary}"
        )

    def _build_recovery_budget_exhausted_message(
        self,
        *,
        unresolved_round: int,
        failure_kind: str,
        last_error: str,
    ) -> str:
        lines = [
            f"optimize worker recovery budget exhausted for unresolved round {unresolved_round}.",
            f"Last recoverable failure kind: {failure_kind}.",
        ]
        detail = last_error.strip()
        if detail:
            lines.append(detail)
        return "\n".join(lines)

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
