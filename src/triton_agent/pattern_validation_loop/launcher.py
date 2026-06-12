from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from triton_agent.models import AgentRequest, CommandKind
from triton_agent.pattern_validation_loop.git_worktree import resolve_git_worktree
from triton_agent.pattern_validation_loop.orchestration import run_pattern_validation_loop_orchestrated
from triton_agent.pattern_validation_loop.paths import (
    DEFAULT_BATCH_DIR,
    DEFAULT_STATE_FILE,
    DEFAULT_SYNTHESIS_FILE,
    resolve_repo_path,
)
from triton_agent.pattern_validation_loop.workspace_plan import (
    DEFAULT_KNOWLEDGE_FILE,
    resolve_knowledge_base_path,
)
from triton_agent.pattern_validation_loop.prompts import build_analyze_prompt, build_prepare_prompt
from triton_agent.pattern_validation_loop.seed_skills import (
    DEFAULT_SKILLS_DIR_NAME,
    seed_pattern_validation_skills_dir,
)
from triton_agent.prompts import append_additional_user_instructions
from triton_agent.resources import skills_root
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import staged_skill_dir
from triton_agent.skills_source_dir import OPTIMIZE_KNOWLEDGE_SKILL_NAME
OPTIMIZE_BATCH_ENV_PREFIX = "TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 "


def build_optimize_batch_extra_flags(
    *,
    target_chip: str | None = None,
    test_mode: str | None = None,
    bench_mode: str | None = None,
) -> str:
    parts: list[str] = []
    if target_chip is not None:
        parts.append(f"--target-chip {target_chip}")
    if test_mode is not None:
        parts.append(f"--test-mode {test_mode}")
    if bench_mode is not None:
        parts.append(f"--bench-mode {bench_mode}")
    if not parts:
        return ""
    return " " + " ".join(parts)


def build_optimize_batch_shell_command(
    *,
    batch_dir: Path | str,
    skills_dir: str,
    min_rounds: int,
    optimize_knowledge: str,
    agent_name: str,
    extra_flags: str = "",
    resume: Literal["fresh", "continue"],
    reset_optimize: bool = False,
) -> str:
    reset_flag = " --reset-optimize" if reset_optimize else ""
    return (
        f"{OPTIMIZE_BATCH_ENV_PREFIX}triton-agent optimize-batch -i {Path(batch_dir).as_posix()}"
        f" --resume {resume}{reset_flag}"
        f" --min-rounds {min_rounds} --concurrency 1 --show-output"
        f" --optimize-knowledge {optimize_knowledge} --skills-source-dir {skills_dir}"
        f"{extra_flags} --agent {agent_name}"
    )


def build_pattern_validation_loop_prompt(
    *,
    repo_path: Path,
    synthesis_path: Path,
    batch_dir: Path,
    skills_workdir: Path,
    skills_dir: str,
    state_path: Path,
    base_revision: str,
    min_rounds: int,
    max_iterations: int,
    agent_name: str,
    optimize_knowledge: str,
    target_chip: str | None = None,
    test_mode: str | None = None,
    bench_mode: str | None = None,
) -> str:
    """Legacy combined prompt; the CLI loop uses prepare/analyze prompts instead."""
    backend_skills = staged_skill_dir(agent_name)
    skill_root = backend_skills / "triton-npu-pattern-validation-loop"
    knowledge_root = skills_workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    prepare = build_prepare_prompt(
        repo_path=repo_path,
        synthesis_path=synthesis_path,
        knowledge_path=resolve_knowledge_base_path(repo_path, DEFAULT_KNOWLEDGE_FILE),
        batch_dir=batch_dir,
        workspace_plan_path=batch_dir / "workspace-plan.json",
        skills_workdir=skills_workdir,
        skills_dir=skills_dir,
        state_path=state_path,
        base_revision=base_revision,
        skill_root=skill_root,
        knowledge_root=knowledge_root,
    )
    analyze = build_analyze_prompt(
        repo_path=repo_path,
        batch_dir=batch_dir,
        skills_workdir=skills_workdir,
        state_path=state_path,
        audit_report_path=batch_dir / "audit-report.json",
        iteration=1,
        max_iterations=max_iterations,
        skill_root=skill_root,
        knowledge_root=knowledge_root,
    )
    return (
        f"{prepare}\n\n---\n\n"
        "After prepare, the CLI runs optimize-batch and collects evidence. "
        "Then an analyze agent runs with instructions like:\n\n"
        f"{analyze}"
    )


def build_pattern_validation_loop_request(
    *,
    target_path: Path,
    synthesis_output: str = DEFAULT_SYNTHESIS_FILE,
    batch_dir: str = DEFAULT_BATCH_DIR,
    skills_dir: str = DEFAULT_SKILLS_DIR_NAME,
    base_revision: str = "origin/main",
    min_rounds: int = 10,
    max_iterations: int = 5,
    optimize_knowledge: str = "v1",
    target_chip: str | None = None,
    test_mode: str | None = None,
    bench_mode: str | None = None,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    user_prompt: str | None = None,
) -> AgentRequest:
    repo_root = resolve_git_worktree(target_path)
    synthesis_path = resolve_repo_path(repo_root, synthesis_output)
    if not synthesis_path.is_file():
        raise ValueError(
            f"Synthesis report not found: {synthesis_path}. "
            "Pass --synthesis to an existing synthesis report or create PERF_PATTERN_SYNTHESIS.md in the repo.",
        )
    batch_path = resolve_repo_path(repo_root, batch_dir)
    state_path = repo_root / DEFAULT_STATE_FILE
    skills_workdir = seed_pattern_validation_skills_dir(
        repo_root,
        skills_dir,
        optimize_knowledge=optimize_knowledge,
    )
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.PATTERN_VALIDATION_LOOP,
        optimize_knowledge=optimize_knowledge,
    )
    prompt = append_additional_user_instructions(
        build_pattern_validation_loop_prompt(
            repo_path=repo_root,
            synthesis_path=synthesis_path,
            batch_dir=batch_path,
            skills_workdir=skills_workdir,
            skills_dir=skills_dir,
            state_path=state_path,
            base_revision=base_revision,
            min_rounds=min_rounds,
            max_iterations=max_iterations,
            agent_name=agent_name,
            optimize_knowledge=optimize_knowledge,
            target_chip=target_chip,
            test_mode=test_mode,
            bench_mode=bench_mode,
        ),
        user_prompt,
    )
    return AgentRequest(
        command_kind=CommandKind.PATTERN_VALIDATION_LOOP,
        input_path=repo_root,
        operator_path=None,
        output_path=batch_path,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=verbose,
        show_output=show_output,
        force_overwrite=False,
        agent_name=agent_name,
        skill_name="triton-npu-pattern-validation-loop",
        prompt=prompt,
        workdir=repo_root,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        min_rounds=min_rounds,
    )


def run_pattern_validation_loop(
    *,
    target_path: Path,
    synthesis_output: str = DEFAULT_SYNTHESIS_FILE,
    knowledge_base: str = "PERF_KNOWLEDGE_BASE.md",
    batch_dir: str = DEFAULT_BATCH_DIR,
    skills_dir: str = DEFAULT_SKILLS_DIR_NAME,
    base_revision: str = "origin/main",
    min_rounds: int = 10,
    max_iterations: int = 5,
    optimize_knowledge: Literal["v1", "v2", "v3"] = "v1",
    target_chip: Literal["A3", "A5"] | None = None,
    test_mode: Literal["standalone", "differential"] | None = None,
    bench_mode: Literal["standalone", "msprof"] | None = None,
    agent_name: str = "codex",
    verbose: bool = False,
    show_output: bool = True,
    user_prompt: str | None = None,
    skip_launch_functions: list[str] | None = None,
    pull_request_ids: list[str] | None = None,
) -> int:
    return run_pattern_validation_loop_orchestrated(
        target_path=target_path,
        synthesis_output=synthesis_output,
        knowledge_base=knowledge_base,
        batch_dir=batch_dir,
        skills_dir=skills_dir,
        base_revision=base_revision,
        min_rounds=min_rounds,
        max_iterations=max_iterations,
        optimize_knowledge=optimize_knowledge,
        target_chip=target_chip,
        test_mode=test_mode,
        bench_mode=bench_mode,
        agent_name=agent_name,
        verbose=verbose,
        show_output=show_output,
        user_prompt=user_prompt,
        skip_launch_functions=skip_launch_functions,
        pull_request_ids=pull_request_ids,
    )

