from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.factory import create_runner
from triton_agent.convert.models import ConvertOptions
from triton_agent.convert.outputs import resolve_convert_output_path
from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.otel_trace import build_tool_trace_env, trace_path_from_request, write_tool_trace_summary
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager
from triton_agent.show_output_log import show_output_log_path
from triton_agent.verbose import emit_verbose, emit_verbose_lines


def build_convert_request(
    input_path: Path,
    operator_path: Path,
    workdir: Path,
    options: ConvertOptions,
) -> AgentRequest:
    output_path = resolve_convert_output_path(input_path, explicit_output=options.output)
    staged_skill_names, staged_skill_sources = resolve_staged_skills(CommandKind.CONVERT)
    prompt = append_additional_user_instructions(
        build_prompt(
            CommandKind.CONVERT,
            input_path,
            operator_path,
            output_path,
            options.test_mode,
            None,
            options.force_overwrite,
            options.remote,
            options.remote_workdir,
            None,
            False,
        ),
        options.prompt,
    )
    extra_env = None
    run_id = ""
    if options.log_tools:
        extra_env, _trace_path, run_id = build_tool_trace_env(None, workdir=workdir, run_id_prefix="convert")

    return AgentRequest(
        command_kind=CommandKind.CONVERT,
        input_path=input_path,
        operator_path=operator_path,
        output_path=output_path,
        test_mode=options.test_mode,
        bench_mode=None,
        interact=options.interact,
        verbose=options.verbose,
        show_output=options.show_output,
        force_overwrite=options.force_overwrite,
        agent_name=options.agent_name,
        skill_name=COMMAND_TO_SKILL[CommandKind.CONVERT],
        prompt=prompt,
        workdir=workdir,
        extra_env=extra_env,
        run_id=run_id,
        min_rounds=None,
        continue_optimize=False,
        no_agent_session=False,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        log_tools=options.log_tools,
    )


def run_convert_request(
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
        _write_convert_trace_summary(request)
        if request.verbose:
            emit_verbose_lines(stderr or sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(stderr or sys.stderr, "skills", warning)


def _write_convert_trace_summary(request: AgentRequest) -> None:
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
