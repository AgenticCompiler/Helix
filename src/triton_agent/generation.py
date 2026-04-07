from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from triton_agent.models import AgentRequest, AgentResult, COMMAND_TO_SKILL, CommandKind
from triton_agent.paths import default_generated_output_path
from triton_agent.prompts import build_prompt
from triton_agent.runner_factory import create_runner
from triton_agent.skills import SkillLinkManager
from triton_agent.verbose import emit_verbose, emit_verbose_lines


@dataclass(frozen=True)
class GenerationOptions:
    interact: bool
    verbose: bool
    show_output: bool
    force_overwrite: bool
    agent_name: str
    remote: str | None
    remote_workdir: str | None
    min_rounds: int | None
    continue_optimize: bool
    output: str | None
    test_mode: str | None
    bench_mode: str | None


def resolve_generation_output_path(
    command_kind: CommandKind,
    input_path: Path,
    *,
    explicit_output: str | None,
    test_mode: str | None = None,
) -> Path | None:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH, CommandKind.OPTIMIZE}:
        return default_generated_output_path(command_kind, input_path, test_mode=test_mode)
    return None


def prepare_generation_target(
    command_kind: CommandKind,
    output_path: Path | None,
    force_overwrite: bool,
) -> list[str]:
    if output_path is None:
        return []
    if command_kind not in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
        return []
    if not output_path.exists():
        return []
    if output_path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory: {output_path}. Choose a file path instead."
        )
    if not force_overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force-overwrite to replace it."
        )
    output_path.unlink()
    return [f"removed existing output file {output_path}"]


def build_generation_request(
    command_kind: CommandKind,
    input_path: Path,
    operator_path: Path,
    workdir: Path,
    options: GenerationOptions,
) -> AgentRequest:
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
    )


def run_generation_request(request: AgentRequest) -> AgentResult:
    repo_root = Path(__file__).resolve().parents[2]
    manager = SkillLinkManager(repo_root / "skills")
    links = manager.prepare_skills(request.agent_name, request.workdir)
    if request.verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))
    try:
        runner = create_runner(request.agent_name)
        return runner.run(request)
    finally:
        if request.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        warnings = manager.cleanup(links)
        for warning in warnings:
            emit_verbose(sys.stderr, "skills", warning)
