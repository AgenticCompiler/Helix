from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.factory import create_runner
from triton_agent.generation.models import GenerationOptions
from triton_agent.generation.outputs import resolve_generation_output_path
from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.otel_trace import build_tool_trace_env, new_trace_run_id, trace_path_from_request, write_tool_trace_summary
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.remote_execution_env import merge_remote_execution_env
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager
from triton_agent.show_output_log import show_output_log_path
from triton_agent.verbose import emit_verbose, emit_verbose_lines


def build_generation_request(
    command_kind: CommandKind,
    input_path: Path,
    operator_path: Path,
    workdir: Path,
    options: GenerationOptions,
) -> AgentRequest:
    staged_skill_names, staged_skill_sources = resolve_staged_skills(command_kind)
    output_path = resolve_generation_output_path(
        command_kind,
        input_path,
        explicit_output=options.output,
        test_mode=options.test_mode,
    )
    prompt = append_additional_user_instructions(
        build_prompt(
            command_kind,
            input_path,
            operator_path,
            output_path,
            options.test_mode,
            options.bench_mode,
            options.force_overwrite,
            options.remote,
            options.remote_workdir,
            options.min_rounds,
            options.continue_optimize,
        ),
        options.prompt,
    )
    extra_env = None
    run_id = new_trace_run_id(prefix="generate")
    if options.log_tools:
        extra_env, _trace_path, _ = build_tool_trace_env(None, workdir=workdir, run_id=run_id)
    extra_env = merge_remote_execution_env(extra_env, options.remote, options.remote_workdir)

    return AgentRequest(
        command_kind=command_kind,
        input_path=input_path,
        operator_path=operator_path,
        output_path=output_path,
        test_mode=options.test_mode,
        bench_mode=options.bench_mode,
        interact=options.interact,
        verbose=options.verbose,
        show_output=options.show_output,
        force_overwrite=options.force_overwrite,
        agent_name=options.agent_name,
        skill_name=COMMAND_TO_SKILL[command_kind],
        prompt=prompt,
        workdir=workdir,
        remote=options.remote,
        remote_workdir=options.remote_workdir,
        extra_env=extra_env,
        run_id=run_id,
        min_rounds=options.min_rounds,
        continue_optimize=options.continue_optimize,
        no_agent_session=False,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        log_tools=options.log_tools,
    )


def run_generation_request(
    request: AgentRequest,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> AgentResult:
    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        request.agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
    )
    if request.verbose:
        emit_verbose_lines(stderr or sys.stderr, "skills", manager.describe_prepare(links))
    try:
        runner = create_runner(request.agent_name)
        if stdout is not None or stderr is not None:
            return cast(Any, runner).run(request, stdout=stdout, stderr=stderr)
        return runner.run(request)
    finally:
        _write_generation_trace_summary(request)
        if request.verbose:
            emit_verbose_lines(stderr or sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(stderr or sys.stderr, "skills", warning)


def _write_generation_trace_summary(request: AgentRequest) -> None:
    if not request.log_tools:
        return
    trace_path = trace_path_from_request(request)
    if trace_path is None:
        return
    warnings = write_tool_trace_summary(
        trace_path=trace_path,
        command_kind=request.command_kind.value,
        show_output_path=show_output_log_path(request),
    )
    if request.verbose and warnings:
        emit_verbose_lines(sys.stderr, "trace", warnings)
