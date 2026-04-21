from __future__ import annotations

from dataclasses import replace
import re
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.models import AgentRequest, AgentResult
from triton_agent.optimize.guidance import (
    OptimizeGuidanceManager,
    OptimizeGuidanceState,
    SharedOptimizeGuidanceState,
)
from triton_agent.optimize.models import GateDecision, GateResult
from triton_agent.optimize.run_loop import OptimizeRunLoop
from triton_agent.prompts import build_optimize_supervisor_prompt
from triton_agent.verbose import emit_verbose, emit_verbose_lines


class RecoveryRunnerAdapter:
    def __init__(
        self,
        runner: AgentRunner,
        guidance_manager: OptimizeGuidanceManager,
        guidance_state: SharedOptimizeGuidanceState,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._runner = runner
        self._guidance_manager = guidance_manager
        self._guidance_state = guidance_state
        self._stdout = stdout
        self._stderr = stderr

    def run(self, request: AgentRequest) -> AgentResult:
        if self._stdout is None and self._stderr is None:
            result = cast(Any, self._runner).run(request)
        else:
            result = cast(Any, self._runner).run(request, stdout=self._stdout, stderr=self._stderr)
        self._record_session(request, result)
        return result

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        if self._stdout is None and self._stderr is None:
            result = cast(Any, self._runner).resume(request, summary)
        else:
            result = cast(Any, self._runner).resume(
                request,
                summary,
                stdout=self._stdout,
                stderr=self._stderr,
            )
        self._record_session(request, result)
        return result

    def _record_session(self, request: AgentRequest, result: AgentResult) -> None:
        self._guidance_manager.record_agent_session(
            self._guidance_state,
            role=request.optimize_role or "worker",
            session_id=result.session_id,
            agent=request.agent_name,
        )


class SupervisedOptimizeAdapter:
    def __init__(
        self,
        runner: AgentRunner,
        guidance_state: OptimizeGuidanceState,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
        guidance_manager: OptimizeGuidanceManager | None = None,
    ) -> None:
        self._runner = runner
        self._guidance_manager = guidance_manager or OptimizeGuidanceManager()
        self._guidance_state = guidance_state
        self._stdout = stdout
        self._stderr = stderr

    def run_worker(self, request: AgentRequest) -> AgentResult:
        worker_request = replace(
            request,
            optimize_role="worker",
            no_agent_session=True,
            round_brief_path=self._guidance_state.round_brief_path,
            supervisor_report_path=self._guidance_state.supervisor_report_path,
        )
        return self._run_request(worker_request)

    def run_supervisor(self, request: AgentRequest, result: AgentResult) -> GateResult:
        del result
        latest_round_dir = _latest_round_dir(request.workdir)
        if latest_round_dir is None:
            gate_result = GateResult(
                decision=GateDecision.REVISE_REQUIRED,
                blocking_issues=("missing opt-round-* directory after worker run",),
            )
            self._write_gate_handoff(request, gate_result, latest_round_dir=None)
            return gate_result

        supervisor_request = replace(
            request,
            prompt=build_optimize_supervisor_prompt(
                request.workdir,
                latest_round_dir=latest_round_dir,
                require_analysis=request.require_analysis,
            ),
            skill_name="triton-npu-optimize",
            optimize_role="supervisor",
            interact=False,
            no_agent_session=True,
            round_brief_path=self._guidance_state.round_brief_path,
            supervisor_report_path=self._guidance_state.supervisor_report_path,
        )
        supervisor_result = self._run_request(supervisor_request)
        if not supervisor_result.succeeded:
            output = supervisor_result.stdout.strip() or supervisor_result.stderr.strip() or "supervisor run failed"
            gate_result = GateResult(
                decision=GateDecision.HARD_FAIL,
                blocking_issues=(output[-2000:],),
            )
            self._write_gate_handoff(request, gate_result, latest_round_dir=latest_round_dir)
            return gate_result

        gate_result = self._read_supervisor_gate_result(latest_round_dir)
        if request.min_rounds is not None and _count_round_directories(request.workdir) < request.min_rounds:
            if gate_result.decision == GateDecision.PASS_STOP:
                gate_result = GateResult(
                    decision=GateDecision.PASS_CONTINUE,
                    blocking_issues=(
                        f"minimum round requirement not yet satisfied: "
                        f"{_count_round_directories(request.workdir)}/{request.min_rounds}",
                    ),
                )
                self._write_gate_handoff(request, gate_result, latest_round_dir=latest_round_dir)
                return gate_result
        self._snapshot_live_handoff_files()
        return gate_result

    def _read_supervisor_gate_result(self, latest_round_dir: Path) -> GateResult:
        report_path = self._guidance_state.supervisor_report_path
        try:
            report_content = report_path.read_text(encoding="utf-8")
        except OSError as exc:
            gate_result = GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=(f"failed to read supervisor report {report_path}: {exc}",),
            )
            self._write_gate_handoff(None, gate_result, latest_round_dir=latest_round_dir)
            return gate_result

        parsed_decision = self._parse_supervisor_decision(report_content)
        if parsed_decision is None:
            gate_result = GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=(
                    f"missing supervisor decision line in {report_path.name}",
                ),
            )
            self._write_gate_handoff(None, gate_result, latest_round_dir=latest_round_dir)
            return gate_result

        try:
            decision = GateDecision(parsed_decision)
        except ValueError:
            gate_result = GateResult(
                decision=GateDecision.REVISE_METADATA,
                blocking_issues=(
                    f"invalid supervisor decision `{parsed_decision}` in {report_path.name}",
                ),
            )
            self._write_gate_handoff(None, gate_result, latest_round_dir=latest_round_dir)
            return gate_result

        return GateResult(
            decision=decision,
            blocking_issues=self._parse_supervisor_blocking_issues(report_content),
        )

    def _write_gate_handoff(
        self,
        request: AgentRequest | None,
        gate_result: GateResult,
        *,
        latest_round_dir: Path | None,
    ) -> None:
        del request
        report_lines = [
            "# Optimize Supervisor Report",
            "",
            f"Decision: {gate_result.decision.value}",
            f"Blocking issues: {', '.join(gate_result.blocking_issues) or 'none'}",
        ]
        if latest_round_dir is not None:
            report_lines.append(f"Latest round: {latest_round_dir.name}")
        report_content = "\n".join(report_lines) + "\n"
        self._guidance_state.supervisor_report_path.write_text(report_content, encoding="utf-8")

        brief_lines = [
            "# Optimize Round Brief",
            "",
            f"Previous gate decision: {gate_result.decision.value}",
        ]
        if gate_result.blocking_issues:
            brief_lines.append(f"Required focus: {'; '.join(gate_result.blocking_issues)}")
        elif latest_round_dir is not None:
            brief_lines.append(f"Continue from `{latest_round_dir.name}`.")
        brief_content = "\n".join(brief_lines) + "\n"
        self._guidance_state.round_brief_path.write_text(brief_content, encoding="utf-8")

        self._snapshot_live_handoff_files()

    def _snapshot_live_handoff_files(self) -> None:
        history_dir = self._guidance_state.history_dir
        history_dir.mkdir(parents=True, exist_ok=True)
        round_label = self._next_history_round_label()
        brief_content = self._guidance_state.round_brief_path.read_text(encoding="utf-8")
        report_content = self._guidance_state.supervisor_report_path.read_text(encoding="utf-8")
        (history_dir / f"{round_label}-brief.md").write_text(brief_content, encoding="utf-8")
        (history_dir / f"{round_label}-supervisor-report.md").write_text(report_content, encoding="utf-8")

    def _next_history_round_label(self) -> str:
        history_dir = self._guidance_state.history_dir
        max_index = 0
        for path in history_dir.glob("round-*.md"):
            if not path.is_file():
                continue
            match = re.match(r"round-(\d+)-", path.name)
            if match is None:
                continue
            max_index = max(max_index, int(match.group(1)))
        return f"round-{max_index + 1:03d}"

    def _run_request(self, request: AgentRequest) -> AgentResult:
        if self._stdout is None and self._stderr is None:
            result = cast(Any, self._runner).run(request)
        else:
            result = cast(Any, self._runner).run(request, stdout=self._stdout, stderr=self._stderr)
        self._guidance_manager.record_agent_session(
            self._guidance_state,
            role=request.optimize_role or "worker",
            session_id=result.session_id,
            agent=request.agent_name,
        )
        return result

    def _parse_supervisor_decision(self, report_content: str) -> str | None:
        match = re.search(r"^Decision:\s*(.+?)\s*$", report_content, re.MULTILINE)
        if match is None:
            return None
        return match.group(1).strip()

    def _parse_supervisor_blocking_issues(self, report_content: str) -> tuple[str, ...]:
        match = re.search(r"^Blocking issues:\s*(.+?)\s*$", report_content, re.MULTILINE)
        if match is None:
            return ()
        raw_value = match.group(1).strip()
        if not raw_value or raw_value.lower() == "none":
            return ()
        return tuple(issue.strip() for issue in raw_value.split(",") if issue.strip())


def execute_supervised_optimize(
    runner: AgentRunner,
    guidance_manager: OptimizeGuidanceManager,
    request: AgentRequest,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    verbose_stream: TextIO,
) -> AgentResult:
    guidance_state = guidance_manager.prepare_supervised_session(
        request.workdir,
        agent_name=request.agent_name,
        require_analysis=request.require_analysis,
        compiler_source_path=request.compiler_source_path,
        compiler_source_commit=request.compiler_source_commit,
    )
    if request.verbose:
        emit_verbose_lines(
            verbose_stream,
            "agents",
            guidance_manager.describe_prepare_supervised_session(guidance_state),
        )
    try:
        adapter = SupervisedOptimizeAdapter(
            runner,
            guidance_state,
            stdout=stdout,
            stderr=stderr,
            guidance_manager=guidance_manager,
        )
        return OptimizeRunLoop().run(adapter, request)
    finally:
        if request.verbose:
            emit_verbose_lines(
                verbose_stream,
                "agents",
                guidance_manager.describe_cleanup_supervised_session(guidance_state),
            )
        warnings = guidance_manager.cleanup_supervised_session(guidance_state)
        for warning in warnings:
            emit_verbose(verbose_stream, "agents", warning)


def execute_unsupervised_optimize(
    runner: AgentRunner,
    guidance_manager: OptimizeGuidanceManager,
    request: AgentRequest,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    verbose_stream: TextIO,
) -> AgentResult:
    shared_guidance_state = guidance_manager.prepare_unsupervised_session(
        request.workdir,
        operator_path=request.input_path,
        test_mode=request.test_mode or "differential",
        bench_mode=request.bench_mode or "standalone",
        agent_name=request.agent_name,
        require_analysis=request.require_analysis,
        compiler_source_path=request.compiler_source_path,
        compiler_source_commit=request.compiler_source_commit,
    )
    if request.verbose:
        emit_verbose_lines(
            verbose_stream,
            "agents",
            guidance_manager.describe_prepare_unsupervised_session(shared_guidance_state),
        )
    try:
        return OptimizeRunLoop().run(
            RecoveryRunnerAdapter(
                runner,
                guidance_manager,
                shared_guidance_state,
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
                guidance_manager.describe_cleanup_unsupervised_session(shared_guidance_state),
            )
        warnings = guidance_manager.cleanup_unsupervised_session(shared_guidance_state)
        for warning in warnings:
            emit_verbose(verbose_stream, "agents", warning)


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


def _iter_numeric_round_dirs(workdir: Path) -> list[Path]:
    return [
        path
        for path in workdir.glob("opt-round-*")
        if path.is_dir() and re.match(r"opt-round-\d+$", path.name)
    ]
