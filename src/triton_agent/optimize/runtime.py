from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.agent import AgentRunner
from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.resume import resolve_optimize_resume
from triton_agent.optimize_guidance import OptimizeGuidanceManager
from triton_agent.paths import default_generated_output_path
from triton_agent.prompts import build_prompt
from triton_agent.runner_factory import create_runner
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
        if stdout is not None or stderr is not None:
            return OptimizeSupervisor().run(RunnerWithStreams(runner, stdout=stdout, stderr=stderr), request)
        return OptimizeSupervisor().run(runner, request)
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
