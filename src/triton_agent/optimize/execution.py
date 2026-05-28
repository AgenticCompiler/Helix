from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.optimize.checks import check_baseline, check_round
from triton_agent.optimize.round_contract import load_round_state
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
from triton_agent.otel_trace import build_trace_env
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
        if not request.log_tools:
            return request
        role = request.optimize_role or "worker"
        return replace(
            request,
            extra_env=build_trace_env(
                request.extra_env,
                trace_path=self._artifacts_state.otel_trace_path,
                run_id=self._artifacts_state.archive.run_id,
                role=role,
                workspace_root=request.workdir,
            ),
        )

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
    round_dirs = sorted(_iter_numeric_round_dirs(workdir), key=lambda path: _round_sort_key(path.name))
    if not round_dirs:
        return None
    return round_dirs[-1]


def _count_round_directories(workdir: Path) -> int:
    return sum(1 for _ in _iter_numeric_round_dirs(workdir))


def _round_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"opt-round-(\d+)$", name)
    if match is None:
        return (-1, name)
    return (int(match.group(1)), name)


def _parse_supervisor_decision_from_report(report_content: str) -> str | None:
    match = re.search(r"^Decision:\s*(\S+)", report_content, re.MULTILINE)
    if match is None:
        return None
    return match.group(1)


def _parse_supervisor_blocking_issues_from_report(report_content: str) -> tuple[str, ...]:
    match = re.search(r"^Blocking issues:\s*(.+?)\s*$", report_content, re.MULTILINE)
    if match is None:
        return ()
    raw_value = match.group(1).strip()
    if not raw_value or raw_value.lower() == "none":
        return ()
    return tuple(issue.strip() for issue in raw_value.split(",") if issue.strip())


def _iter_numeric_round_dirs(workdir: Path) -> list[Path]:
    return [
        path
        for path in workdir.glob("opt-round-*")
        if path.is_dir() and re.match(r"opt-round-\d+$", path.name)
    ]


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
        return self._run_request(baseline_request)

    def run_round_loop(self, request: AgentRequest) -> AgentResult:
        current_request = request
        active_repair_round: str | None = None
        repair_attempts = 0
        while True:
            round_request = replace(
                current_request,
                optimize_role="worker",
            )
            round_result = self._run_request(round_request)
            if not round_result.succeeded:
                return round_result

            gate = self.technical_gate(current_request)
            latest_round_dir = _latest_round_dir(request.workdir)
            latest_round_name = latest_round_dir.name if latest_round_dir is not None else None
            if gate.decision == GateDecision.HARD_FAIL:
                return AgentResult(
                    return_code=1,
                    stdout=round_result.stdout,
                    stderr="\n".join(gate.blocking_issues),
                )
            if gate.decision == GateDecision.REVISE_REQUIRED:
                repair_attempts, active_repair_round = self._advance_repair_attempts(
                    repair_attempts,
                    active_repair_round,
                    latest_round_name,
                )
                if repair_attempts > self._max_repair_attempts:
                    return AgentResult(
                        return_code=1,
                        stdout=round_result.stdout,
                        stderr=self._build_repair_exhausted_message(gate, repair_attempts),
                    )
                self._write_round_brief(
                    gate.decision,
                    gate.blocking_issues,
                    latest_round_name=latest_round_name,
                )
                current_request = replace(
                    current_request,
                    prompt=self._build_continue_prompt(
                        request.prompt,
                        self._build_gate_summary(gate),
                        round_mode=current_request.round_mode,
                        optimize_target=current_request.optimize_target,
                        compiler_source_path=current_request.compiler_source_path,
                        compiler_source_commit=current_request.compiler_source_commit,
                    ),
                )
                continue

            if request.round_mode == "supervised" and gate.decision in {
                GateDecision.PASS_CONTINUE,
                GateDecision.PASS_STOP,
            }:
                supervisor_gate = self._run_supervisor_pass(current_request, round_result)
                if supervisor_gate.decision == GateDecision.PASS_STOP:
                    return round_result
                if supervisor_gate.decision == GateDecision.HARD_FAIL:
                    return AgentResult(
                        return_code=1,
                        stdout=round_result.stdout,
                        stderr="\n".join(supervisor_gate.blocking_issues),
                    )
                if supervisor_gate.decision in {
                    GateDecision.REVISE_METADATA,
                    GateDecision.REVISE_REQUIRED,
                }:
                    repair_attempts, active_repair_round = self._advance_repair_attempts(
                        repair_attempts,
                        active_repair_round,
                        latest_round_name,
                    )
                    if repair_attempts > self._max_repair_attempts:
                        return AgentResult(
                            return_code=1,
                            stdout=round_result.stdout,
                            stderr=self._build_repair_exhausted_message(
                                supervisor_gate,
                                repair_attempts,
                            ),
                        )
                if supervisor_gate.decision in {
                    GateDecision.PASS_CONTINUE,
                    GateDecision.REVISE_METADATA,
                    GateDecision.REVISE_REQUIRED,
                }:
                    current_request = replace(
                        current_request,
                        prompt=self._build_continue_prompt(
                            request.prompt,
                            self._build_gate_summary(supervisor_gate),
                            round_mode=current_request.round_mode,
                            optimize_target=current_request.optimize_target,
                            compiler_source_path=current_request.compiler_source_path,
                            compiler_source_commit=current_request.compiler_source_commit,
                        ),
                    )
                    continue

            repair_attempts = 0
            active_repair_round = None
            if gate.decision == GateDecision.PASS_STOP:
                return round_result

            if gate.decision == GateDecision.PASS_CONTINUE:
                self._write_round_brief(
                    gate.decision,
                    gate.blocking_issues,
                    latest_round_name=latest_round_name,
                )
                current_request = replace(
                    current_request,
                    prompt=self._build_continue_prompt(
                        request.prompt,
                        self._build_gate_summary(gate),
                        round_mode=current_request.round_mode,
                        optimize_target=current_request.optimize_target,
                        compiler_source_path=current_request.compiler_source_path,
                        compiler_source_commit=current_request.compiler_source_commit,
                    ),
                )
                continue

            return round_result

    def technical_gate(self, request: AgentRequest) -> GateResult:
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=("missing opt-round-* directory after round run",),
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
            )
        if check_result.decision == "revise-required":
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=check_result.issues,
            )

        round_state = load_round_state(latest_round_dir)
        round_count = _count_round_directories(request.workdir)
        if round_count < min_rounds:
            return GateResult(
                decision=GateDecision.PASS_CONTINUE,
                blocking_issues=(
                    f"minimum round requirement not yet satisfied: {round_count}/{min_rounds}",
                ),
            )
        decision = (
            GateDecision.PASS_STOP
            if round_state.round_disposition == "stop"
            else GateDecision.PASS_CONTINUE
        )
        return GateResult(decision=decision, blocking_issues=check_result.issues)

    def _run_supervisor_pass(
        self, request: AgentRequest, worker_result: AgentResult
    ) -> GateResult:
        del worker_result
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            return GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=("missing opt-round-* directory after worker run",),
            )

        supervisor_request = replace(
            request,
            prompt=build_optimize_supervisor_prompt(
                request.workdir,
                latest_round_dir=latest_round_dir,
            ),
            skill_name="triton-npu-optimize",
            optimize_role="supervisor",
            interact=False,
            no_agent_session=True,
        )
        supervisor_result = self._run_request(supervisor_request)
        if not supervisor_result.succeeded:
            return GateResult(
                decision=GateDecision.HARD_FAIL,
                blocking_issues=(supervisor_result.stdout.strip() or supervisor_result.stderr.strip() or "supervisor run failed",),
            )

        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if supervisor_report_path is None:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=("supervisor report path is not configured for this optimize session",),
            )
        try:
            report_content = supervisor_report_path.read_text(encoding="utf-8")
        except OSError:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=("failed to read supervisor report",),
            )
        self._snapshot_live_handoff_files()

        parsed_decision = _parse_supervisor_decision_from_report(report_content)
        if parsed_decision is None:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=("missing supervisor decision line in supervisor-report.md",),
            )
        try:
            decision = GateDecision(parsed_decision)
        except ValueError:
            return GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=(f"invalid supervisor decision `{parsed_decision}` in supervisor-report.md",),
            )
        return GateResult(
            decision=decision,
            blocking_issues=_parse_supervisor_blocking_issues_from_report(report_content),
        )

    def _run_request(self, request: AgentRequest) -> AgentResult:
        request = replace(
            request,
            no_agent_session=True,
            round_brief_path=self._artifacts_state.round_brief_path,
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

    def _build_gate_summary(self, gate_result: GateResult) -> str:
        issues = "; ".join(gate_result.blocking_issues) if gate_result.blocking_issues else "none"
        return f"Gate decision: {gate_result.decision.value}. Blocking issues: {issues}"

    def _build_continue_prompt(
        self,
        base_prompt: str,
        summary: str,
        *,
        round_mode: str = "checked",
        optimize_target: str = "kernel",
        compiler_source_path: Path | None = None,
        compiler_source_commit: str | None = None,
    ) -> str:
        return build_optimize_resume_prompt(
            summary,
            base_prompt=base_prompt,
            round_mode=round_mode,
            optimize_target=optimize_target,
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
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

    def _write_round_brief(
        self,
        decision: GateDecision,
        blocking_issues: tuple[str, ...],
        *,
        latest_round_name: str | None,
    ) -> None:
        lines = [
            "# Optimize Round Brief",
            "",
            f"Previous gate decision: {decision.value}",
        ]
        if blocking_issues:
            lines.append(f"Required focus: {'; '.join(blocking_issues)}")
        elif latest_round_name is not None:
            if decision == GateDecision.PASS_CONTINUE:
                lines.append(f"Continue from `{latest_round_name}`.")
            elif decision == GateDecision.PASS_STOP:
                lines.append(f"Stop after validating `{latest_round_name}`.")
            else:
                lines.append(f"Repair `{latest_round_name}` before continuing.")
        self._artifacts_state.round_brief_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _snapshot_live_handoff_files(self) -> None:
        history_dir = self._artifacts_state.history_dir
        supervisor_report_path = self._artifacts_state.supervisor_report_path
        if history_dir is None or supervisor_report_path is None:
            return
        history_dir.mkdir(parents=True, exist_ok=True)
        round_label = self._next_history_round_label(history_dir)
        brief_content = self._artifacts_state.round_brief_path.read_text(encoding="utf-8")
        report_content = supervisor_report_path.read_text(encoding="utf-8")
        (history_dir / f"{round_label}-brief.md").write_text(brief_content, encoding="utf-8")
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
