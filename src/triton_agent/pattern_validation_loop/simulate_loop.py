from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from triton_agent.backends.factory import create_runner
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.pattern_validation_loop.simulate_plan import (
    SimulatePlanConfig,
    bootstrap_simulate_batch,
    ensure_simulate_synthesis_ready,
    prepare_simulate_batch,
    run_simulate_workspace_agents,
    write_batch_simulate_report,
    _run_follow_up_optimize_batch,
)
from triton_agent.pattern_validation_loop.simulate_prompts import (
    BATCH_SIMULATE_REPORT_FILENAME,
    build_simulate_skill_audit_prompt,
)
from triton_agent.pattern_validation_loop.workspace_plan import (
    generate_workspace_plan_if_present,
)
from triton_agent.resources import skills_root
from triton_agent.skill_loader import load_skill_script_module
from triton_agent.skill_staging import resolve_staged_skills
from triton_agent.skills import SkillLinkManager, staged_skill_dir
from triton_agent.skills_source_dir import OPTIMIZE_KNOWLEDGE_SKILL_NAME
from triton_agent.verbose import emit_verbose_lines

_PATTERN_VALIDATION_SKILL = "triton-npu-pattern-validation-loop"
_DEFAULT_STATE_FILE = ".triton-agent/pattern-validation-simulate-state.json"
_RECORD_SCRIPT = "record_simulate_iteration"


@dataclass(frozen=True)
class SimulateLoopContext:
    config: SimulatePlanConfig
    state_path: Path


def run_pattern_validation_simulate_loop(config: SimulatePlanConfig) -> tuple[int, Path]:
    ctx = SimulateLoopContext(
        config=config,
        state_path=config.repo_root / _DEFAULT_STATE_FILE,
    )
    _ensure_simulate_state(ctx)

    extract_code = run_commit_perf_extraction_if_needed(config)
    if extract_code != 0:
        _record_simulate_phase(
            ctx,
            phase="failed",
            note=f"commit extraction failed: exit {extract_code}",
        )
        return extract_code, config.batch_path / BATCH_SIMULATE_REPORT_FILENAME

    try:
        ensure_simulate_synthesis_ready(config)
    except ValueError as exc:
        print(f"[pattern-validation-simulate] {exc}", file=sys.stderr, flush=True)
        _record_simulate_phase(ctx, phase="failed", note=str(exc))
        return 2, config.batch_path / BATCH_SIMULATE_REPORT_FILENAME

    _knowledge_rel = _relative_to_repo(config.repo_root, config.knowledge_path)
    workspace_plan_path, plan_warnings = generate_workspace_plan_if_present(
        repo_root=config.repo_root,
        batch_root=config.batch_path,
        knowledge_output=_knowledge_rel,
        base_revision=config.base_revision,
        skip_launch_functions=list(config.skip_launch_functions),
        pull_request_ids=list(config.pull_request_ids),
        stream=sys.stderr,
        log_tag="pattern-validation-simulate",
    )
    for warning in plan_warnings:
        print(f"[pattern-validation-simulate] {warning}", file=sys.stderr, flush=True)
    if config.verbose and workspace_plan_path is not None:
        print(
            f"[pattern-validation-simulate] workspace plan: {workspace_plan_path.as_posix()}",
            file=sys.stderr,
            flush=True,
        )

    bootstrap_code = bootstrap_simulate_batch(
        config,
        workspace_plan_path=workspace_plan_path,
        simulate_state_path=ctx.state_path,
        stream=sys.stderr,
    )
    if bootstrap_code != 0:
        _record_simulate_phase(
            ctx,
            phase="failed",
            note=f"batch bootstrap failed: exit {bootstrap_code}",
        )
        return bootstrap_code, config.batch_path / BATCH_SIMULATE_REPORT_FILENAME

    for iteration in range(1, config.max_iterations + 1):
        _set_state_iteration(ctx, iteration)
        if config.verbose:
            print(
                f"[pattern-validation-simulate] iteration {iteration}/{config.max_iterations}",
                file=sys.stderr,
                flush=True,
            )

        prep_code = 0
        if iteration > 1:
            prep_code = prepare_simulate_batch(config)
        elif config.verbose:
            print(
                "[pattern-validation-simulate] skipping redundant deps sync "
                "(bootstrap already synced and verified)",
                file=sys.stderr,
                flush=True,
            )
        if prep_code != 0:
            _record_simulate_phase(ctx, phase="failed", note=f"sync failed: exit {prep_code}")
            return prep_code, config.batch_path / BATCH_SIMULATE_REPORT_FILENAME

        results, agent_code = run_simulate_workspace_agents(config)
        report_path = write_batch_simulate_report(config.batch_path, results)
        _record_simulate_phase(
            ctx,
            phase="simulate",
            note=f"simulate agents exit {agent_code}",
            simulate_report_path=report_path,
            increment_iteration=False,
        )
        if agent_code != 0:
            _record_simulate_phase(ctx, phase="failed", note="simulate agent failures")
            return agent_code, report_path

        if config.verbose:
            print(
                "[pattern-validation-simulate] running simulate-analyze & skill-audit "
                "(independent review of simulate reports, expected patterns, and pattern cards)",
                file=sys.stderr,
                flush=True,
            )
        audit_code = _run_skill_audit_agent(ctx, report_path=report_path, iteration=iteration)
        if audit_code != 0:
            _record_simulate_phase(ctx, phase="failed", note=f"simulate-analyze & skill-audit agent exit {audit_code}")
            return audit_code, report_path

        state = _load_simulate_state(ctx.state_path)
        if state.get("status") == "complete":
            _record_simulate_phase(
                ctx,
                phase="complete",
                note="simulate-analyze confirmed simulate loop complete",
            )
            return _finish_simulate_loop(ctx, report_path)

        if iteration >= config.max_iterations:
            print(
                "[pattern-validation-simulate] reached max_iterations without completion",
                file=sys.stderr,
            )
            _record_simulate_phase(ctx, phase="failed", note="max_iterations exhausted")
            return 1, report_path

    report_path = config.batch_path / BATCH_SIMULATE_REPORT_FILENAME
    return 1, report_path


def _finish_simulate_loop(ctx: SimulateLoopContext, report_path: Path) -> tuple[int, Path]:
    if ctx.config.run_optimize_after:
        return _run_follow_up_optimize_batch(ctx.config, stream=sys.stderr), report_path
    from triton_agent.pattern_validation_loop.simulate_plan import print_batch_summary

    print_batch_summary(ctx.config, report_path, failed_count=0, stream=sys.stderr)
    return 0, report_path


def _ensure_simulate_state(ctx: SimulateLoopContext) -> None:
    if ctx.state_path.is_file():
        return
    ctx.state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "running",
        "iteration": 1,
        "max_iterations": ctx.config.max_iterations,
        "batch_dir": ctx.config.batch_path.name,
        "skills_dir": ctx.config.skills_workdir.name,
        "synthesis_path": _relative_to_repo(ctx.config.repo_root, ctx.config.synthesis_path),
        "knowledge_path": _relative_to_repo(ctx.config.repo_root, ctx.config.knowledge_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "history": [],
    }
    ctx.state_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_simulate_state(state_path: Path) -> dict[str, Any]:
    return json.loads(state_path.read_text(encoding="utf-8"))


def _set_state_iteration(ctx: SimulateLoopContext, iteration: int) -> None:
    state = _load_simulate_state(ctx.state_path)
    state["iteration"] = iteration
    ctx.state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _record_simulate_phase(
    ctx: SimulateLoopContext,
    *,
    phase: str,
    note: str,
    simulate_report_path: Path | None = None,
    increment_iteration: bool = False,
) -> None:
    module = load_skill_script_module(_PATTERN_VALIDATION_SKILL, _RECORD_SCRIPT)
    argv = ["--state", ctx.state_path.as_posix(), "--phase", phase, "--note", note]
    if simulate_report_path is not None:
        argv.extend(["--simulate-report", simulate_report_path.as_posix()])
    if increment_iteration:
        argv.append("--increment-iteration")
    code = int(module.main(argv))
    if code != 0:
        raise RuntimeError(f"record_simulate_iteration failed with exit code {code}")


def _run_skill_audit_agent(
    ctx: SimulateLoopContext,
    *,
    report_path: Path,
    iteration: int,
) -> int:
    backend_skills = staged_skill_dir(ctx.config.agent_name)
    skill_root = backend_skills / _PATTERN_VALIDATION_SKILL
    knowledge_root = ctx.config.skills_workdir / OPTIMIZE_KNOWLEDGE_SKILL_NAME
    record_script = skill_root / "scripts" / f"{_RECORD_SCRIPT}.py"
    prompt = build_simulate_skill_audit_prompt(
        repo_path=ctx.config.repo_root,
        batch_dir=ctx.config.batch_path,
        skills_workdir=ctx.config.skills_workdir,
        state_path=ctx.state_path,
        simulate_report_path=report_path,
        iteration=iteration,
        max_iterations=ctx.config.max_iterations,
        skill_root=skill_root,
        knowledge_root=knowledge_root,
        record_script=record_script,
    )
    staged_skill_names, staged_skill_sources = resolve_staged_skills(
        CommandKind.PATTERN_VALIDATION_LOOP,
        optimize_knowledge=ctx.config.optimize_knowledge,
    )
    request = AgentRequest(
        command_kind=CommandKind.PATTERN_VALIDATION_LOOP,
        input_path=ctx.config.repo_root,
        operator_path=None,
        output_path=ctx.config.batch_path,
        test_mode=None,
        bench_mode=None,
        interact=False,
        verbose=ctx.config.verbose,
        show_output=ctx.config.show_output,
        force_overwrite=False,
        agent_name=ctx.config.agent_name,
        skill_name=_PATTERN_VALIDATION_SKILL,
        prompt=prompt,
        workdir=ctx.config.repo_root,
        no_agent_session=True,
        staged_skill_names=staged_skill_names,
        staged_skill_sources=staged_skill_sources,
    )
    return _run_agent_request(ctx.config, request)


def _relative_to_repo(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _run_agent_request(config: SimulatePlanConfig, request: AgentRequest) -> int:
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
            f"[pattern-validation-simulate] agent executable not found: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if config.verbose:
            emit_verbose_lines(sys.stderr, "skills", manager.describe_cleanup(links))
        for warning in manager.cleanup(links):
            print(f"[pattern-validation-simulate] cleanup warning: {warning}", file=sys.stderr)
    return result.return_code


def run_commit_perf_extraction_if_needed(config: SimulatePlanConfig) -> int:
    """Run analyze-commit-perf when PERF reports are missing or --force is set."""
    if config.skip_extract:
        if config.verbose:
            print(
                "[pattern-validation-simulate] skipping commit extraction (--skip-extract)",
                file=sys.stderr,
                flush=True,
            )
        return 0

    synthesis_exists = config.synthesis_path.is_file()
    knowledge_exists = config.knowledge_path.is_file()
    if synthesis_exists and knowledge_exists and not config.force_extract:
        if config.verbose:
            print(
                "[pattern-validation-simulate] reusing existing PERF reports "
                f"(synthesis={_relative_to_repo(config.repo_root, config.synthesis_path)}, "
                f"knowledge={_relative_to_repo(config.repo_root, config.knowledge_path)}); "
                "pass --force to re-extract",
                file=sys.stderr,
                flush=True,
            )
        return 0

    from triton_agent.commit_perf_analysis.launcher import run_commit_perf_analysis

    print(
        "[pattern-validation-simulate] running commit performance extraction "
        "before simulate loop",
        file=sys.stderr,
        flush=True,
    )
    knowledge_output = _relative_to_repo(config.repo_root, config.knowledge_path)
    synthesis_output = _relative_to_repo(config.repo_root, config.synthesis_path)
    return run_commit_perf_analysis(
        target_path=config.repo_root,
        output=knowledge_output,
        synthesis_output=synthesis_output,
        base_revision=config.base_revision,
        target_chip=config.target_chip,
        include_ir=config.include_ir,
        force=config.force_extract,
        pull_requests=list(config.pull_request_ids),
        agent_name=config.agent_name,
        verbose=config.verbose,
        show_output=config.show_output,
        user_prompt=config.user_prompt,
    )
