from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.pattern_validation_loop.prompts import build_prepare_prompt
from triton_agent.prompts import append_additional_user_instructions
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager, staged_skill_dir
from triton_agent.skills_source_dir import OPTIMIZE_KNOWLEDGE_SKILL_NAME
from triton_agent.verbose import emit_verbose_lines

_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"


@dataclass(frozen=True)
class PrepareBatchParams:
    repo_root: Path
    synthesis_path: Path
    knowledge_path: Path
    batch_path: Path
    skills_workdir: Path
    skills_dir: str
    state_path: Path
    base_revision: str
    agent_name: str
    optimize_knowledge: Literal["v1", "v2", "v3"]
    verbose: bool
    show_output: bool
    user_prompt: str | None
    log_tag: str = "pattern-validation-loop"
    workflow: Literal["loop", "simulate"] = "loop"


def run_pattern_validation_prepare_agent(
    params: PrepareBatchParams,
    *,
    workspace_plan_path: Path | None,
) -> int:
    backend_skills = staged_skill_dir(params.agent_name)
    skill_root = backend_skills / _PATTERN_VALIDATION_SKILL
    knowledge_root = params.skills_workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    prompt = append_additional_user_instructions(
        build_prepare_prompt(
            repo_path=params.repo_root,
            synthesis_path=params.synthesis_path,
            knowledge_path=params.knowledge_path,
            batch_dir=params.batch_path,
            workspace_plan_path=workspace_plan_path,
            skills_workdir=params.skills_workdir,
            skills_dir=params.skills_dir,
            state_path=params.state_path,
            base_revision=params.base_revision,
            skill_root=skill_root,
            knowledge_root=knowledge_root,
            workflow=params.workflow,
        ),
        params.user_prompt,
    )
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.PATTERN_VALIDATION_LOOP,
        optimize_knowledge=params.optimize_knowledge,
    )
    request = AgentRequest(
        command_kind=CommandKind.PATTERN_VALIDATION_LOOP,
        input_path=params.repo_root,
        operator_path=None,
        output_path=params.batch_path,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=params.verbose,
        show_output=params.show_output,
        force_overwrite=False,
        agent_name=params.agent_name,
        skill_name=_PATTERN_VALIDATION_SKILL,
        prompt=prompt,
        workdir=params.repo_root,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
    )
    return _run_pattern_validation_agent(params, request)


def _run_pattern_validation_agent(params: PrepareBatchParams, request: AgentRequest) -> int:
    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        params.agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
        skill_sources=request.staged_skill_sources,
    )
    if params.verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))

    try:
        runner = create_runner(params.agent_name)
        result = runner.run(request)
    except FileNotFoundError as exc:
        print(
            f"[{params.log_tag}] agent executable not found: {exc}. "
            f"Make sure the '{params.agent_name}' CLI is installed and available in PATH.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if params.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        for warning in manager.cleanup(links):
            print(f"[{params.log_tag}] cleanup warning: {warning}", file=sys.stderr, flush=True)

    return result.return_code
