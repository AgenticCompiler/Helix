from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TextIO

from triton_agent.backends.factory import create_runner
from triton_agent.batch_utils import (
    NO_CANDIDATE_OPERATOR_FILE,
    PrefixedTextStream,
    discover_batch_workspaces,
)
from triton_agent.models import AgentRequest, CommandKind
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.naming import resolve_batch_optimize_operator_file
from triton_agent.optimize.orchestration import build_optimize_request
from triton_agent.optimize.pattern_reminders import resolve_generic_optimize_knowledge_skill_name
from triton_agent.optimize.session_artifacts import OptimizeSessionArtifactsManager
from triton_agent.pattern_validation_loop.git_worktree import resolve_git_worktree
from triton_agent.pattern_validation_loop.paths import resolve_repo_path
from triton_agent.pattern_validation_loop.scaffold_verify import run_pattern_validation_verify
from triton_agent.pattern_validation_loop.seed_skills import (
    DEFAULT_SKILLS_DIR_NAME,
    seed_pattern_validation_skills_dir,
)
from triton_agent.pattern_validation_loop.simulate_prompts import (
    BATCH_SIMULATE_REPORT_FILENAME,
    SIMULATE_PLAN_DIR,
    SIMULATE_REPORT_FILENAME,
    build_simulate_plan_prompt,
)
from triton_agent.pattern_validation_loop.reference_tests import (
    build_pattern_validation_optimize_reference_test_prompt,
)
from triton_agent.pattern_validation_loop.workspace_sync import sync_batch_workspace_dependencies
from triton_agent.prompts import append_additional_user_instructions
from triton_agent.resources import skills_root
from triton_agent.skills import SkillLinkManager
from triton_agent.skills_source_dir import build_skills_source_overrides
from triton_agent.verbose import emit_verbose_lines

_DEFAULT_BATCH_DIR = "pattern-validation-batch"


@dataclass(frozen=True)
class SimulatePlanConfig:
    repo_root: Path
    batch_path: Path
    skills_workdir: Path
    agent_name: str
    optimize_knowledge: Literal["v1", "v2", "v3"]
    target_chip: Literal["A3", "A5"]
    test_mode: str | None
    bench_mode: str | None
    user_prompt: str | None
    verbose: bool
    show_output: bool
    skip_verify: bool
    run_optimize_after: bool


@dataclass(frozen=True)
class WorkspaceSimulateResult:
    workspace: Path
    status: Literal["ok", "failed", "skipped"]
    message: str
    report_path: Path | None = None


def build_simulate_plan_config(
    *,
    target_path: Path,
    batch_dir: str = _DEFAULT_BATCH_DIR,
    skills_dir: str = DEFAULT_SKILLS_DIR_NAME,
    agent_name: str = "codex",
    optimize_knowledge: Literal["v1", "v2", "v3"] = "v1",
    target_chip: Literal["A3", "A5"] = "A5",
    test_mode: str | None = None,
    bench_mode: str | None = None,
    user_prompt: str | None = None,
    verbose: bool = False,
    show_output: bool = True,
    skip_verify: bool = False,
    run_optimize_after: bool = False,
) -> SimulatePlanConfig:
    repo_root = resolve_git_worktree(target_path)
    batch_path = resolve_repo_path(repo_root, batch_dir)
    skills_workdir = seed_pattern_validation_skills_dir(
        repo_root,
        skills_dir,
        optimize_knowledge=optimize_knowledge,
    )
    return SimulatePlanConfig(
        repo_root=repo_root,
        batch_path=batch_path,
        skills_workdir=skills_workdir,
        agent_name=agent_name,
        optimize_knowledge=optimize_knowledge,
        target_chip=target_chip,
        test_mode=test_mode,
        bench_mode=bench_mode,
        user_prompt=user_prompt,
        verbose=verbose,
        show_output=show_output,
        skip_verify=skip_verify,
        run_optimize_after=run_optimize_after,
    )


def build_simulate_optimize_options(config: SimulatePlanConfig) -> OptimizeRunOptions:
    return OptimizeRunOptions(
        agent_name=config.agent_name,
        interact=False,
        verbose=config.verbose,
        show_output=config.show_output,
        remote=None,
        remote_workdir=None,
        min_rounds=1,
        resume_mode="fresh",
        reset_optimize=False,
        no_agent_session=True,
        round_mode="continuous",
        output=None,
        test_mode=config.test_mode,
        bench_mode=config.bench_mode,
        prompt=None,
        target_chip=config.target_chip,
        optimize_knowledge=config.optimize_knowledge,
        upload_enabled=False,
        report=False,
        skills_source_dir=config.skills_workdir.resolve(),
    )


def build_simulate_plan_request(
    operator_file: Path,
    workspace: Path,
    config: SimulatePlanConfig,
) -> AgentRequest:
    options = build_simulate_optimize_options(config)
    base = build_optimize_request(operator_file, workspace, options)
    meta = _load_validation_meta(workspace)
    simulate_prompt = build_simulate_plan_prompt(
        operator_path=operator_file,
        workdir=workspace,
        test_mode=base.test_mode,
        bench_mode=base.bench_mode,
        target_chip=base.target_chip,
        optimize_target=base.optimize_target,
        validation_meta=meta,
        compiler_source_path=base.compiler_source_path,
        compiler_source_commit=base.compiler_source_commit,
        enable_cann_ext_api=options.enable_cann_ext_api,
        user_prompt=config.user_prompt,
    )
    return replace(
        base,
        prompt=simulate_prompt,
        min_rounds=None,
        continue_optimize=False,
        show_output_label="simulate-plan",
    )


def run_simulate_plan_request(
    request: AgentRequest,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    manager = SkillLinkManager(skills_root())
    knowledge_overrides = build_skills_source_overrides(
        request.workdir,
        request.agent_name,
        request.skills_source_dir,
        request.staged_skill_names,
    )
    links = manager.prepare_skills(
        request.agent_name,
        request.workdir,
        skill_names=request.staged_skill_names,
        skill_sources=request.staged_skill_sources,
        skill_dir_overrides=knowledge_overrides,
    )
    verbose_stream = stderr or sys.stderr
    if request.verbose:
        emit_verbose_lines(verbose_stream, "skills", manager.describe_prepare(links))
    try:
        runner = create_runner(request.agent_name)
        artifacts_manager = OptimizeSessionArtifactsManager()
        session_state = artifacts_manager.prepare_continuous_session(
            request.workdir,
            operator_path=request.input_path,
            test_mode=request.test_mode or "differential",
            bench_mode=request.bench_mode or "standalone",
            agent_name=request.agent_name,
            optimize_target=request.optimize_target,
            compiler_source_path=request.compiler_source_path,
            compiler_source_commit=request.compiler_source_commit,
            enable_cann_ext_api=_staged_cann_ext_api_enabled(request),
            optimize_knowledge_skill_name=resolve_generic_optimize_knowledge_skill_name(
                request.staged_skill_names,
                request.staged_skill_sources,
            ),
        )
        if request.verbose:
            emit_verbose_lines(
                verbose_stream,
                "agents",
                artifacts_manager.describe_prepare_continuous_session(session_state),
            )
        try:
            result = runner.run(request, stdout=stdout, stderr=stderr)
            return result.return_code
        finally:
            if request.verbose:
                emit_verbose_lines(
                    verbose_stream,
                    "agents",
                    artifacts_manager.describe_cleanup_continuous_session(session_state),
                )
            for warning in artifacts_manager.cleanup_continuous_session(session_state):
                emit_verbose(verbose_stream, "agents", warning)
    finally:
        if request.verbose:
            emit_verbose_lines(verbose_stream, "skills", manager.describe_cleanup(links))
        for warning in manager.cleanup(links):
            emit_verbose(verbose_stream, "skills", warning)
    return 1


def run_simulate_plan_batch(
    config: SimulatePlanConfig,
    *,
    stream: TextIO | None = None,
) -> tuple[int, Path]:
    out = stream or sys.stderr
    batch_path = config.batch_path
    if not batch_path.is_dir():
        print(
            f"[pattern-validation-simulate] batch directory not found: {batch_path}",
            file=sys.stderr,
        )
        return 2, batch_path / BATCH_SIMULATE_REPORT_FILENAME

    sync_code = sync_batch_workspace_dependencies(
        batch_path,
        config.repo_root,
        stream=out,
    )
    if sync_code != 0:
        print(
            "[pattern-validation-simulate] dependency sync failed; fix imports before simulate.",
            file=sys.stderr,
        )
        return sync_code, batch_path / BATCH_SIMULATE_REPORT_FILENAME

    if not config.skip_verify:
        verify_code = run_pattern_validation_verify(batch_path, stream=out)
        if verify_code != 0:
            print(
                "[pattern-validation-simulate] scaffold verify failed; fix workspaces first.",
                file=sys.stderr,
            )
            return verify_code, batch_path / BATCH_SIMULATE_REPORT_FILENAME

    discovered, failures = discover_batch_workspaces(
        batch_path,
        resolve_operator_file=resolve_batch_optimize_operator_file,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
    results: list[WorkspaceSimulateResult] = []
    for workspace, message in failures:
        results.append(
            WorkspaceSimulateResult(workspace=workspace, status="failed", message=message),
        )

    for workspace, operator_file in discovered:
        print(f"[pattern-validation-simulate] {workspace.name}", file=out, flush=True)
        request = build_simulate_plan_request(operator_file, workspace, config)
        prefix = f"[{workspace.name}] "
        stdout = PrefixedTextStream(sys.stdout, prefix) if config.show_output else None
        stderr = PrefixedTextStream(sys.stderr, prefix) if config.show_output else None
        code = run_simulate_plan_request(request, stdout=stdout, stderr=stderr)
        report_path = workspace / SIMULATE_PLAN_DIR / SIMULATE_REPORT_FILENAME
        if code == 0 and report_path.is_file():
            results.append(
                WorkspaceSimulateResult(
                    workspace=workspace,
                    status="ok",
                    message="simulate plan report written",
                    report_path=report_path,
                ),
            )
        elif code == 0:
            results.append(
                WorkspaceSimulateResult(
                    workspace=workspace,
                    status="failed",
                    message=f"agent exited 0 but missing {report_path.as_posix()}",
                ),
            )
        else:
            results.append(
                WorkspaceSimulateResult(
                    workspace=workspace,
                    status="failed",
                    message=f"simulate agent exit code {code}",
                    report_path=report_path if report_path.is_file() else None,
                ),
            )

    batch_report_path = write_batch_simulate_report(batch_path, results)
    failed = [item for item in results if item.status != "ok"]
    print_batch_summary(config, batch_report_path, failed_count=len(failed), stream=out)

    exit_code = 1 if failed else 0
    if config.run_optimize_after and exit_code == 0:
        optimize_code = _run_follow_up_optimize_batch(config, stream=out)
        return optimize_code, batch_report_path
    return exit_code, batch_report_path


def write_batch_simulate_report(
    batch_path: Path,
    results: list[WorkspaceSimulateResult],
) -> Path:
    workspace_reports: list[dict[str, Any]] = []
    for item in results:
        entry: dict[str, Any] = {
            "workspace": item.workspace.name,
            "status": item.status,
            "message": item.message,
            "report_path": item.report_path.as_posix() if item.report_path else None,
        }
        if item.report_path is not None and item.report_path.is_file():
            try:
                payload = json.loads(item.report_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    entry["simulate_report"] = payload
            except json.JSONDecodeError:
                entry["simulate_report_error"] = "invalid JSON in workspace report"
        workspace_reports.append(entry)

    payload = {
        "schema_version": 1,
        "batch_root": batch_path.as_posix(),
        "workspace_count": len(results),
        "ok_count": sum(1 for item in results if item.status == "ok"),
        "failed_count": sum(1 for item in results if item.status != "ok"),
        "workspaces": workspace_reports,
        "next_step_manual_optimize": build_manual_optimize_command_hint(batch_path),
    }
    report_path = batch_path / BATCH_SIMULATE_REPORT_FILENAME
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def build_manual_optimize_command_hint(batch_path: Path) -> str:
    return (
        "TRITON_AGENT_STALL_TIMEOUT_SECONDS=0 uv run triton-agent optimize-batch "
        f"-i {batch_path.as_posix()} --resume fresh --reset-optimize "
        "--min-rounds 10 --concurrency 1 --show-output "
        "--skills-source-dir pattern-validation-skills --agent <backend>"
    )


def print_batch_summary(
    config: SimulatePlanConfig,
    batch_report_path: Path,
    *,
    failed_count: int,
    stream: TextIO,
) -> None:
    print(f"[pattern-validation-simulate] batch report: {batch_report_path.as_posix()}", file=stream)
    if failed_count:
        print(
            f"[pattern-validation-simulate] {failed_count} workspace(s) failed; "
            "fix reports before running optimize-batch.",
            file=stream,
        )
    else:
        print(
            "[pattern-validation-simulate] all workspace simulate plans succeeded.",
            file=stream,
        )
    if not config.run_optimize_after:
        print(
            "[pattern-validation-simulate] To run real optimize after manual review, use:\n  "
            + build_manual_optimize_command_hint(config.batch_path),
            file=stream,
        )


def _run_follow_up_optimize_batch(config: SimulatePlanConfig, *, stream: TextIO) -> int:
    from triton_agent.optimize.batch import run_optimize_batch

    print("[pattern-validation-simulate] running optimize-batch (--run-optimize)", file=stream, flush=True)
    optimize_prompt = append_additional_user_instructions(
        build_pattern_validation_optimize_reference_test_prompt(),
        config.user_prompt,
    )
    options = OptimizeRunOptions(
        agent_name=config.agent_name,
        interact=False,
        verbose=config.verbose,
        show_output=config.show_output,
        remote=None,
        remote_workdir=None,
        min_rounds=10,
        resume_mode="fresh",
        reset_optimize=True,
        no_agent_session=True,
        round_mode="continuous",
        output=None,
        test_mode=config.test_mode,
        bench_mode=config.bench_mode,
        prompt=optimize_prompt,
        target_chip=config.target_chip,
        optimize_knowledge=config.optimize_knowledge,
        upload_enabled=False,
        report=False,
        skills_source_dir=config.skills_workdir.resolve(),
    )
    return run_optimize_batch(config.batch_path, options, max_concurrency=1)


def _staged_cann_ext_api_enabled(request: AgentRequest) -> bool:
    return (
        request.staged_skill_names is not None
        and "triton-npu-cann-ext-api-patterns" in request.staged_skill_names
    )


def _load_validation_meta(workspace: Path) -> dict[str, Any] | None:
    meta_path = workspace / "validation-meta.json"
    if not meta_path.is_file():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None
