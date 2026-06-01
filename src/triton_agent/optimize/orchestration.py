from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from triton_agent.backends.factory import create_runner

from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.optimize import execution as optimize_execution
from triton_agent.optimize.compiler_source import prepare_compiler_source
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsManager
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.resume import resolve_optimize_resume, reset_optimize_workspace
from triton_agent.paths import default_generated_output_path
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager
from triton_agent.verbose import emit_verbose, emit_verbose_lines


def build_optimize_request(
    input_path: Path,
    workdir: Path,
    options: OptimizeRunOptions,
) -> AgentRequest:
    if options.reset_optimize:
        reset_optimize_workspace(input_path, workdir)
    resolution = resolve_optimize_resume(
        input_path,
        workdir,
        resume_mode=options.resume_mode,
        reset_optimize=options.reset_optimize,
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
    compiler_source = None
    if options.compiler_source_analysis != "off":
        compiler_source = prepare_compiler_source(
            mode=options.compiler_source_analysis,
        )
    if compiler_source is not None:
        if options.enable_cann_ext_api:
            built_prompt = build_prompt(
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
                round_mode=options.round_mode,
                target_chip=options.target_chip,
                compiler_source_path=compiler_source.path,
                compiler_source_commit=compiler_source.commit,
                enable_cann_ext_api=True,
            )
        else:
            built_prompt = build_prompt(
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
                round_mode=options.round_mode,
                target_chip=options.target_chip,
                compiler_source_path=compiler_source.path,
                compiler_source_commit=compiler_source.commit,
            )
    else:
        if options.enable_cann_ext_api:
            built_prompt = build_prompt(
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
                round_mode=options.round_mode,
                target_chip=options.target_chip,
                enable_cann_ext_api=True,
            )
        else:
            built_prompt = build_prompt(
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
                round_mode=options.round_mode,
                target_chip=options.target_chip,
            )
    prompt = append_additional_user_instructions(
        built_prompt,
        options.prompt,
    )
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.OPTIMIZE,
        optimize_knowledge=options.optimize_knowledge,
        optimize_target=options.optimize_target,
        enable_cann_ext_api=options.enable_cann_ext_api,
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
        remote=options.remote,
        remote_workdir=options.remote_workdir,
        min_rounds=options.min_rounds,
        continue_optimize=resolution.resume_existing_session,
        no_agent_session=options.no_agent_session,
        round_mode=options.round_mode,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        optimize_role="worker" if options.round_mode != "continuous" else None,
        target_chip=options.target_chip,
        optimize_target=options.optimize_target,
        compiler_source_analysis=options.compiler_source_analysis,
        compiler_source_path=compiler_source.path if compiler_source is not None else None,
        compiler_source_commit=compiler_source.commit if compiler_source is not None else None,
        enable_agent_hooks=options.enable_agent_hooks,
        log_tools=options.log_tools,
    )


def run_optimize_request(
    request: AgentRequest,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> AgentResult:
    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        request.agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
        skill_sources=request.staged_skill_sources,
    )
    verbose_stream = stderr or sys.stderr
    if request.verbose:
        emit_verbose_lines(verbose_stream, "skills", manager.describe_prepare(links))
    try:
        runner = create_runner(request.agent_name)
        artifacts_manager = OptimizeSessionArtifactsManager()
        if request.round_mode == "continuous":
            return optimize_execution.execute_continuous_optimize(
                runner,
                artifacts_manager,
                request,
                stdout=stdout,
                stderr=stderr,
                verbose_stream=verbose_stream,
            )
        return optimize_execution.execute_multi_invocation_optimize(
            runner,
            artifacts_manager,
            request,
            stdout=stdout,
            stderr=stderr,
            verbose_stream=verbose_stream,
            )
    finally:
        if request.verbose:
            emit_verbose_lines(verbose_stream, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(verbose_stream, "skills", warning)
