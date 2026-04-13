from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.base import AgentRunner
from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.optimize.gate import evaluate_round_gate
from triton_agent.optimize.models import GateDecision, GateResult
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.resume import resolve_optimize_resume
from triton_agent.optimize_guidance import OptimizeGuidanceManager, OptimizeGuidanceState
from triton_agent.paths import default_generated_output_path
from triton_agent.prompts import build_optimize_supervisor_prompt, build_prompt
from triton_agent.skills import SkillLinkManager
from triton_agent.supervisor import OptimizeSupervisor
from triton_agent.verbose import emit_verbose, emit_verbose_lines


class RunnerWithStreams:
    def __init__(
        self,
        runner: AgentRunner,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._runner = runner
        self._stdout = stdout
        self._stderr = stderr

    def run(self, request: AgentRequest) -> AgentResult:
        return cast(Any, self._runner).run(request, stdout=self._stdout, stderr=self._stderr)

    def resume(self, request: AgentRequest, summary: str) -> AgentResult:
        return cast(Any, self._runner).resume(
            request,
            summary,
            stdout=self._stdout,
            stderr=self._stderr,
        )


class OptimizeLoopRunner:
    def __init__(
        self,
        runner: AgentRunner,
        guidance_state: OptimizeGuidanceState,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> None:
        self._runner = runner
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
            skill_name="optimize-supervisor",
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

        gate_result = evaluate_round_gate(latest_round_dir)
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

    def _write_gate_handoff(
        self,
        request: AgentRequest,
        gate_result: GateResult,
        *,
        latest_round_dir: Path | None,
    ) -> None:
        report_lines = [
            "# Optimize Supervisor Report",
            "",
            f"Decision: {gate_result.decision.value}",
            f"Blocking issues: {', '.join(gate_result.blocking_issues) or 'none'}",
        ]
        if latest_round_dir is not None:
            report_lines.append(f"Latest round: {latest_round_dir.name}")
        self._guidance_state.supervisor_report_path.write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        brief_lines = [
            "# Optimize Round Brief",
            "",
            f"Previous gate decision: {gate_result.decision.value}",
        ]
        if gate_result.blocking_issues:
            brief_lines.append(f"Required focus: {'; '.join(gate_result.blocking_issues)}")
        elif latest_round_dir is not None:
            brief_lines.append(f"Continue from `{latest_round_dir.name}`.")
        self._guidance_state.round_brief_path.write_text(
            "\n".join(brief_lines) + "\n",
            encoding="utf-8",
        )

    def _run_request(self, request: AgentRequest) -> AgentResult:
        if self._stdout is None and self._stderr is None:
            return cast(Any, self._runner).run(request)
        return cast(Any, self._runner).run(request, stdout=self._stdout, stderr=self._stderr)


def build_optimize_request(
    input_path: Path,
    workdir: Path,
    options: OptimizeRunOptions,
) -> AgentRequest:
    resolution = resolve_optimize_resume(
        input_path,
        workdir,
        resume_mode=options.resume_mode,
        requested_test_mode=options.test_mode,
        requested_bench_mode=options.bench_mode,
    )
    test_mode = resolution.test_mode or "differential"
    bench_mode = resolution.bench_mode or "standalone"

    output_path = (
        Path(options.output).expanduser().resolve()
        if options.output
        else default_generated_output_path(CommandKind.OPTIMIZE, input_path, test_mode=test_mode)
    )
    prompt = build_prompt(
        CommandKind.OPTIMIZE,
        input_path,
        input_path,
        output_path,
        test_mode,
        bench_mode,
        False,
        options.remote,
        options.remote_workdir,
        options.min_rounds,
        resolution.resume_existing_session,
        require_analysis=options.require_analysis,
    )
    return AgentRequest(
        command_kind=CommandKind.OPTIMIZE,
        input_path=input_path,
        operator_path=input_path,
        output_path=output_path,
        test_mode=test_mode,
        bench_mode=bench_mode,
        interact=options.interact,
        verbose=options.verbose,
        show_output=options.show_output,
        force_overwrite=False,
        agent_name=options.agent_name,
        skill_name=COMMAND_TO_SKILL[CommandKind.OPTIMIZE],
        prompt=prompt,
        workdir=workdir,
        min_rounds=options.min_rounds,
        continue_optimize=resolution.resume_existing_session,
        require_analysis=options.require_analysis,
        no_agent_session=options.no_agent_session,
        optimize_role="worker",
    )


def run_optimize_request(
    request: AgentRequest,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> AgentResult:
    repo_root = Path(__file__).resolve().parents[3]
    manager = SkillLinkManager(repo_root / "skills")
    links = manager.prepare_skills(request.agent_name, request.workdir)
    guidance_manager = OptimizeGuidanceManager()
    guidance_state = guidance_manager.prepare(
        request.workdir,
        request.input_path,
        test_mode=request.test_mode or "differential",
        bench_mode=request.bench_mode or "standalone",
        agent_name=request.agent_name,
        require_analysis=request.require_analysis,
    )
    verbose_stream = stderr or sys.stderr
    if request.verbose:
        emit_verbose_lines(verbose_stream, "skills", manager.describe_prepare(links))
    if request.verbose:
        emit_verbose_lines(verbose_stream, "agents", guidance_manager.describe_prepare(guidance_state))
    try:
        runner = create_runner(request.agent_name)
        loop_runner = OptimizeLoopRunner(runner, guidance_state, stdout=stdout, stderr=stderr)
        return OptimizeSupervisor().run(loop_runner, request)
    finally:
        if request.verbose:
            emit_verbose_lines(verbose_stream, "agents", guidance_manager.describe_cleanup(guidance_state))
        warnings = guidance_manager.cleanup(guidance_state)
        for warning in warnings:
            emit_verbose(verbose_stream, "agents", warning)
        if request.verbose:
            emit_verbose_lines(verbose_stream, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(verbose_stream, "skills", warning)


def _latest_round_dir(workdir: Path) -> Path | None:
    round_dirs = sorted(
        (path for path in workdir.glob("opt-round-*") if path.is_dir()),
        key=lambda path: path.name,
    )
    if not round_dirs:
        return None
    return round_dirs[-1]


def _count_round_directories(workdir: Path) -> int:
    return sum(1 for path in workdir.glob("opt-round-*") if path.is_dir())
