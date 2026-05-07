from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.backends.factory import create_runner
from triton_agent.convert.models import ConvertOptions
from triton_agent.convert.outputs import resolve_convert_output_path
from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.prompts import append_additional_user_instructions, build_prompt
from triton_agent.resources import skills_root
from triton_agent.skills import SkillLinkManager
from triton_agent.verbose import emit_verbose, emit_verbose_lines


CONVERT_STAGED_SKILLS = (
    "triton-npu-convert-pytorch-operator",
    "triton-npu-gen-test",
    "triton-npu-run-eval",
    "triton-npu-repair-guide",
)


def build_convert_request(
    input_path: Path,
    operator_path: Path,
    workdir: Path,
    options: ConvertOptions,
) -> AgentRequest:
    output_path = resolve_convert_output_path(input_path, explicit_output=options.output)
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
        min_rounds=None,
        continue_optimize=False,
        no_agent_session=False,
        staged_skill_names=CONVERT_STAGED_SKILLS,
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
        if request.verbose:
            emit_verbose_lines(stderr or sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(stderr or sys.stderr, "skills", warning)
