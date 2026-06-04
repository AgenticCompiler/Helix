from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.optimize.batch import run_optimize_batch
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.pattern_validation_loop.evidence import (
    collect_batch_evidence,
    reset_active_workspace_rounds,
)
from triton_agent.pattern_validation_loop.git_worktree import resolve_git_worktree
from triton_agent.pattern_validation_loop.paths import (
    DEFAULT_BATCH_DIR,
    DEFAULT_STATE_FILE,
    DEFAULT_SYNTHESIS_FILE,
    resolve_repo_path,
)
from triton_agent.pattern_validation_loop.workspace_plan import (
    DEFAULT_KNOWLEDGE_FILE,
    generate_workspace_plan_if_present,
    resolve_knowledge_base_path,
)
from triton_agent.pattern_validation_loop.prompts import build_analyze_prompt, build_prepare_prompt
from triton_agent.pattern_validation_loop.scaffold_verify import run_pattern_validation_verify
from triton_agent.pattern_validation_loop.seed_skills import (
    DEFAULT_SKILLS_DIR_NAME,
    seed_pattern_validation_skills_dir,
)
from triton_agent.pattern_validation_loop.reference_tests import (
    build_pattern_validation_optimize_reference_test_prompt,
)
from triton_agent.prompts import append_additional_user_instructions
from triton_agent.resources import skills_root
from triton_agent.skill_loader import load_skill_script_module
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager, staged_skill_dir
from triton_agent.skills_source_dir import OPTIMIZE_KNOWLEDGE_SKILL_NAME
from triton_agent.verbose import emit_verbose_lines

_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"
_STALL_TIMEOUT_ENV = "TRITON_AGENT_STALL_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class PatternValidationLoopConfig:
    repo_root: Path
    synthesis_path: Path
    knowledge_path: Path
    batch_path: Path
    skills_workdir: Path
    skills_dir: str
    state_path: Path
    base_revision: str
    min_rounds: int
    max_iterations: int
    optimize_knowledge: Literal["v1", "v2", "v3"]
    target_chip: Literal["A3", "A5"] | None
    test_mode: str | None
    bench_mode: str | None
    agent_name: str
    verbose: bool
    show_output: bool
    user_prompt: str | None


def run_pattern_validation_loop_orchestrated(
    *,
    target_path: Path,
    synthesis_output: str = DEFAULT_SYNTHESIS_FILE,
    knowledge_base: str = DEFAULT_KNOWLEDGE_FILE,
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
) -> int:
    try:
        config = _build_loop_config(
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
        )
    except ValueError as exc:
        print(f"[pattern-validation-loop] {exc}", file=sys.stderr, flush=True)
        return 2

    seed_pattern_validation_skills_dir(
        config.repo_root,
        config.skills_dir,
        optimize_knowledge=config.optimize_knowledge,
    )
    _ensure_loop_state(config)

    workspace_plan_path, plan_warnings = generate_workspace_plan_if_present(
        repo_root=config.repo_root,
        batch_root=config.batch_path,
        knowledge_output=_relative_to_repo(config.repo_root, config.knowledge_path),
        base_revision=config.base_revision,
        stream=sys.stderr,
    )
    for warning in plan_warnings:
        print(f"[pattern-validation-loop] {warning}", file=sys.stderr, flush=True)

    prepare_code = _run_prepare_agent(config, workspace_plan_path=workspace_plan_path)
    if prepare_code != 0:
        return prepare_code

    verify_code = run_pattern_validation_verify(config.batch_path, stream=sys.stderr)
    if verify_code != 0:
        print(
            "[pattern-validation-loop] scaffold verification failed; "
            "fix workspaces before optimize.",
            file=sys.stderr,
            flush=True,
        )
        return verify_code

    os.environ[_STALL_TIMEOUT_ENV] = "0"
    audit_report_path = config.batch_path / "audit-report.json"

    for iteration in range(1, config.max_iterations + 1):
        if verbose:
            print(
                f"[pattern-validation-loop] iteration {iteration}/{config.max_iterations}: optimize",
                file=sys.stderr,
                flush=True,
            )

        optimize_code = _run_optimize_batch(
            config,
            resume="fresh" if iteration == 1 else "continue",
            reset_optimize=iteration == 1,
        )
        if optimize_code != 0:
            print(
                "[pattern-validation-loop] optimize-batch reported failures; "
                "continuing with evidence collection and analyze agent.",
                file=sys.stderr,
                flush=True,
            )
            _record_loop_phase(
                config,
                phase="optimize",
                note=f"optimize-batch exit code {optimize_code} on iteration {iteration}",
            )

        collect_batch_evidence(config.batch_path, output_path=audit_report_path)

        if verbose:
            print(
                f"[pattern-validation-loop] iteration {iteration}/{config.max_iterations}: analyze",
                file=sys.stderr,
                flush=True,
            )

        analyze_code = _run_analyze_agent(
            config,
            audit_report_path=audit_report_path,
            iteration=iteration,
        )
        if analyze_code != 0:
            return analyze_code

        state = _load_loop_state(config.state_path)
        if state.get("status") == "complete":
            return 0
        if state.get("status") == "failed":
            return 1
        if iteration >= config.max_iterations:
            print(
                "[pattern-validation-loop] reached max_iterations without completion",
                file=sys.stderr,
                flush=True,
            )
            _mark_loop_failed(config, note="max_iterations exhausted")
            return 1

        if verbose:
            print(
                "[pattern-validation-loop] resetting active workspace rounds for next iteration",
                file=sys.stderr,
                flush=True,
            )
        reset_active_workspace_rounds(config.batch_path)

    return 1


def _build_loop_config(
    *,
    target_path: Path,
    synthesis_output: str,
    knowledge_base: str,
    batch_dir: str,
    skills_dir: str,
    base_revision: str,
    min_rounds: int,
    max_iterations: int,
    optimize_knowledge: Literal["v1", "v2", "v3"],
    target_chip: Literal["A3", "A5"] | None,
    test_mode: str | None,
    bench_mode: str | None,
    agent_name: str,
    verbose: bool,
    show_output: bool,
    user_prompt: str | None,
) -> PatternValidationLoopConfig:
    repo_root = resolve_git_worktree(target_path)
    synthesis_path = resolve_repo_path(repo_root, synthesis_output)
    if not synthesis_path.is_file():
        raise ValueError(
            f"Synthesis report not found: {synthesis_path}. "
            "Pass --synthesis to an existing synthesis report or create PERF_PATTERN_SYNTHESIS.md in the repo.",
        )
    batch_path = resolve_repo_path(repo_root, batch_dir)
    knowledge_path = resolve_knowledge_base_path(repo_root, knowledge_base)
    skills_workdir = repo_root / skills_dir
    state_path = repo_root / DEFAULT_STATE_FILE
    return PatternValidationLoopConfig(
        repo_root=repo_root,
        synthesis_path=synthesis_path,
        knowledge_path=knowledge_path,
        batch_path=batch_path,
        skills_workdir=skills_workdir,
        skills_dir=skills_dir,
        state_path=state_path,
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
    )


def _ensure_loop_state(config: PatternValidationLoopConfig) -> None:
    if config.state_path.is_file():
        return
    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, "init_loop_state")
    argv = [
        "--repo",
        config.repo_root.as_posix(),
        "--synthesis",
        _relative_to_repo(config.repo_root, config.synthesis_path),
        "--batch-dir",
        _relative_to_repo(config.repo_root, config.batch_path),
        "--skills-dir",
        config.skills_dir,
        "--base",
        config.base_revision,
        "--min-rounds",
        str(config.min_rounds),
        "--max-iterations",
        str(config.max_iterations),
        "--state",
        config.state_path.as_posix(),
    ]
    exit_code = int(module.main(argv))
    if exit_code != 0:
        raise RuntimeError(f"init_loop_state failed with exit code {exit_code}")


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _run_prepare_agent(
    config: PatternValidationLoopConfig,
    *,
    workspace_plan_path: Path | None,
) -> int:
    backend_skills = staged_skill_dir(config.agent_name)
    skill_root = backend_skills / _PATTERN_VALIDATION_SKILL
    knowledge_root = config.skills_workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    prompt = append_additional_user_instructions(
        build_prepare_prompt(
            repo_path=config.repo_root,
            synthesis_path=config.synthesis_path,
            knowledge_path=config.knowledge_path,
            batch_dir=config.batch_path,
            workspace_plan_path=workspace_plan_path,
            skills_workdir=config.skills_workdir,
            skills_dir=config.skills_dir,
            state_path=config.state_path,
            base_revision=config.base_revision,
            skill_root=skill_root,
            knowledge_root=knowledge_root,
        ),
        config.user_prompt,
    )
    request = _build_agent_request(config, prompt=prompt)
    return _run_agent_request(config, request)


def _run_analyze_agent(
    config: PatternValidationLoopConfig,
    *,
    audit_report_path: Path,
    iteration: int,
) -> int:
    backend_skills = staged_skill_dir(config.agent_name)
    skill_root = backend_skills / _PATTERN_VALIDATION_SKILL
    knowledge_root = config.skills_workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    state = _load_loop_state(config.state_path)
    current_iteration = int(state.get("iteration", iteration))
    prompt = build_analyze_prompt(
        repo_path=config.repo_root,
        batch_dir=config.batch_path,
        skills_workdir=config.skills_workdir,
        state_path=config.state_path,
        audit_report_path=audit_report_path,
        iteration=current_iteration,
        max_iterations=config.max_iterations,
        skill_root=skill_root,
        knowledge_root=knowledge_root,
    )
    request = _build_agent_request(config, prompt=prompt)
    return _run_agent_request(config, request)


def _build_agent_request(
    config: PatternValidationLoopConfig,
    *,
    prompt: str,
) -> AgentRequest:
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.PATTERN_VALIDATION_LOOP,
        optimize_knowledge=config.optimize_knowledge,
    )
    return AgentRequest(
        command_kind=CommandKind.PATTERN_VALIDATION_LOOP,
        input_path=config.repo_root,
        operator_path=None,
        output_path=config.batch_path,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=config.verbose,
        show_output=config.show_output,
        force_overwrite=False,
        agent_name=config.agent_name,
        skill_name=_PATTERN_VALIDATION_SKILL,
        prompt=prompt,
        workdir=config.repo_root,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
        min_rounds=config.min_rounds,
    )


def _run_agent_request(config: PatternValidationLoopConfig, request: AgentRequest) -> int:
    manager = SkillLinkManager(skills_root())
    links = manager.prepare_skills(
        config.agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
        skill_sources=request.staged_skill_sources,
    )
    if config.verbose:
        emit_verbose_lines(sys.stderr, "skills", manager.describe_prepare(links))

    try:
        runner = create_runner(config.agent_name)
        result = runner.run(request)
    except FileNotFoundError as exc:
        print(
            f"[pattern-validation-loop] agent executable not found: {exc}. "
            f"Make sure the '{config.agent_name}' CLI is installed and available in PATH.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if config.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        cleanup_warnings = manager.cleanup(links)
        for warning in cleanup_warnings:
            print(
                f"[pattern-validation-loop] cleanup warning: {warning}",
                file=sys.stderr,
                flush=True,
            )

    return result.return_code


def _build_pattern_validation_optimize_prompt(user_prompt: str | None) -> str:
    return append_additional_user_instructions(
        build_pattern_validation_optimize_reference_test_prompt(),
        user_prompt,
    )


def _run_optimize_batch(
    config: PatternValidationLoopConfig,
    *,
    resume: Literal["fresh", "continue"],
    reset_optimize: bool,
) -> int:
    options = OptimizeRunOptions(
        agent_name=config.agent_name,
        interact=False,
        verbose=config.verbose,
        show_output=True,
        remote=None,
        remote_workdir=None,
        min_rounds=config.min_rounds,
        resume_mode=resume,
        reset_optimize=reset_optimize,
        no_agent_session=True,
        round_mode="continuous",
        output=None,
        test_mode=config.test_mode,
        bench_mode=config.bench_mode,
        prompt=_build_pattern_validation_optimize_prompt(config.user_prompt),
        target_chip=config.target_chip or "A5",
        optimize_knowledge=config.optimize_knowledge,
        upload_enabled=False,
        report=False,
        skills_source_dir=config.skills_workdir.resolve(),
    )
    return run_optimize_batch(config.batch_path, options, max_concurrency=1)


def _load_loop_state(state_path: Path) -> dict[str, object]:
    return json.loads(state_path.read_text(encoding="utf-8"))


def _record_loop_phase(
    config: PatternValidationLoopConfig,
    *,
    phase: str,
    note: str,
    audit_report_path: Path | None = None,
) -> None:
    if not config.state_path.is_file():
        return
    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, "record_iteration")
    argv = [
        "--state",
        config.state_path.as_posix(),
        "--phase",
        phase,
        "--note",
        note,
    ]
    if audit_report_path is not None:
        argv.extend(["--audit-report", audit_report_path.as_posix()])
    module.main(argv)


def _mark_loop_failed(config: PatternValidationLoopConfig, *, note: str) -> None:
    _record_loop_phase(config, phase="failed", note=note)
