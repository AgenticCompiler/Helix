from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TextIO, cast

from triton_agent.generation.models import GenerationOptions
from triton_agent.generation.outputs import resolve_generation_output_path
from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.prompts import build_prompt
from triton_agent.runner_factory import create_runner
from triton_agent.skills import SkillLinkManager
from triton_agent.verbose import emit_verbose, emit_verbose_lines


GEN_EVAL_STAGED_SKILLS = ("eval-gen", "test-gen", "bench-gen", "operator-eval")


def build_generation_request(
    command_kind: CommandKind,
    input_path: Path,
    operator_path: Path,
    workdir: Path,
    options: GenerationOptions,
) -> AgentRequest:
    staged_skill_names = GEN_EVAL_STAGED_SKILLS if command_kind == CommandKind.GEN_EVAL else None
    output_path = resolve_generation_output_path(
        command_kind,
        input_path,
        explicit_output=options.output,
        test_mode=options.test_mode,
    )
    prompt = build_prompt(
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
    )
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
        min_rounds=options.min_rounds,
        continue_optimize=options.continue_optimize,
        no_agent_session=False,
        staged_skill_names=staged_skill_names,
    )


def run_generation_request(
    request: AgentRequest,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> AgentResult:
    repo_root = Path(__file__).resolve().parents[3]
    manager = SkillLinkManager(repo_root / "skills")
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
