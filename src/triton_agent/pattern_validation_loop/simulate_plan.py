from __future__ import annotations

import json
import shutil
import sys
import threading
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
from triton_agent.pattern_validation_loop.paths import (
    DEFAULT_SYNTHESIS_FILE,
    resolve_repo_path,
)
from triton_agent.pattern_validation_loop.workspace_plan import (
    DEFAULT_KNOWLEDGE_FILE,
    resolve_knowledge_base_path,
)
from triton_agent.pattern_validation_loop.prepare_agent import (
    PrepareBatchParams,
    run_pattern_validation_prepare_agent,
)
from triton_agent.pattern_validation_loop.scaffold_verify import run_pattern_validation_verify
from triton_agent.pattern_validation_loop.simulate_isolation import isolate_workspace_for_simulate
from triton_agent.skill_loader import load_skill_script_module
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
    skip_prepare: bool
    run_optimize_after: bool
    skills_dir: str
    synthesis_path: Path
    knowledge_path: Path
    base_revision: str = "origin/main"
    skip_launch_functions: tuple[str, ...] = ()
    pull_request_ids: tuple[str, ...] = ()
    max_iterations: int = 5


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
    skip_prepare: bool = False,
    run_optimize_after: bool = False,
    max_iterations: int = 5,
    synthesis_output: str = DEFAULT_SYNTHESIS_FILE,
    knowledge_base: str = DEFAULT_KNOWLEDGE_FILE,
    base_revision: str = "origin/main",
    skip_launch_functions: list[str] | None = None,
    pull_request_ids: list[str] | None = None,
) -> SimulatePlanConfig:
    repo_root = resolve_git_worktree(target_path)
    batch_path = resolve_repo_path(repo_root, batch_dir)
    synthesis_path = resolve_repo_path(repo_root, synthesis_output)
    if not synthesis_path.is_file():
        raise ValueError(
            f"Synthesis report not found: {synthesis_path}. "
            f"Pass --synthesis or create {DEFAULT_SYNTHESIS_FILE} in the repo.",
        )
    knowledge_path = resolve_knowledge_base_path(repo_root, knowledge_base)
    skills_workdir = seed_pattern_validation_skills_dir(
        repo_root,
        skills_dir,
        optimize_knowledge=optimize_knowledge,
    )
    skip_launch = tuple(
        name
        for raw in (skip_launch_functions or [])
        for name in raw.replace(",", " ").split()
        if name.strip()
    )
    pull_requests = tuple(
        name
        for raw in (pull_request_ids or [])
        for name in raw.replace(",", " ").split()
        if name.strip()
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
        skip_prepare=skip_prepare,
        run_optimize_after=run_optimize_after,
        skills_dir=skills_dir,
        max_iterations=max(1, max_iterations),
        synthesis_path=synthesis_path,
        knowledge_path=knowledge_path,
        base_revision=base_revision.strip() or "origin/main",
        skip_launch_functions=skip_launch,
        pull_request_ids=pull_requests,
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
    simulate_prompt = build_simulate_plan_prompt(
        operator_path=operator_file,
        workdir=workspace,
        test_mode=base.test_mode,
        bench_mode=base.bench_mode,
        target_chip=base.target_chip,
        optimize_target=base.optimize_target,
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


def run_simulate_workspace_agents(
    config: SimulatePlanConfig,
    *,
    stream: TextIO | None = None,
) -> tuple[list[WorkspaceSimulateResult], int]:
    out = stream or sys.stderr
    discovered, failures = discover_batch_workspaces(
        config.batch_path,
        resolve_operator_file=resolve_batch_optimize_operator_file,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
    results: list[WorkspaceSimulateResult] = []
    for workspace, message in failures:
        results.append(
            WorkspaceSimulateResult(workspace=workspace, status="failed", message=message),
        )

    output_lock = threading.Lock()
    for workspace, operator_file in discovered:
        print(f"[pattern-validation-simulate] {workspace.name}", file=out, flush=True)
        request = build_simulate_plan_request(operator_file, workspace, config)
        prefix = f"[{workspace.name}] "
        stdout = (
            PrefixedTextStream(sys.stdout, prefix, output_lock) if config.show_output else None
        )
        stderr = (
            PrefixedTextStream(sys.stderr, prefix, output_lock) if config.show_output else None
        )
        with isolate_workspace_for_simulate(workspace):
            code = run_simulate_plan_request(request, stdout=stdout, stderr=stderr)
        report_path = workspace / SIMULATE_PLAN_DIR / SIMULATE_REPORT_FILENAME
        if code == 0 and report_path.is_file():
            _payload, validation_errors = _load_simulate_report_payload(report_path)
            if validation_errors:
                results.append(
                    WorkspaceSimulateResult(
                        workspace=workspace,
                        status="failed",
                        message="simulate report failed CLI validation: "
                        + "; ".join(validation_errors),
                        report_path=report_path,
                    ),
                )
            else:
                results.append(
                    WorkspaceSimulateResult(
                        workspace=workspace,
                        status="ok",
                        message="simulate plan report passed CLI validation",
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

    failed = [item for item in results if item.status != "ok"]
    return results, 1 if failed else 0


def bootstrap_simulate_batch(
    config: SimulatePlanConfig,
    *,
    workspace_plan_path: Path | None,
    simulate_state_path: Path,
    stream: TextIO | None = None,
) -> int:
    """Plan → prepare (if needed) → sync deps → verify before simulate iterations."""
    out = stream or sys.stderr
    if not config.batch_path.is_dir():
        config.batch_path.mkdir(parents=True, exist_ok=True)

    _maybe_migrate_legacy_batch_evaluation(config.batch_path, stream=out)
    active_count = _active_validation_workspace_count(config.batch_path)
    if active_count == 0:
        if config.skip_prepare:
            print(
                "[pattern-validation-simulate] no active workspaces and --skip-prepare set; "
                "cannot continue.",
                file=sys.stderr,
            )
            return 1
        print(
            "[pattern-validation-simulate] no workspaces yet — launching prepare agent "
            "(plan → scaffold → verify)",
            file=out,
            flush=True,
        )
        prepare_code = run_pattern_validation_prepare_agent(
            PrepareBatchParams(
                repo_root=config.repo_root,
                synthesis_path=config.synthesis_path,
                knowledge_path=config.knowledge_path,
                batch_path=config.batch_path,
                skills_workdir=config.skills_workdir,
                skills_dir=config.skills_dir,
                state_path=simulate_state_path,
                base_revision=config.base_revision,
                agent_name=config.agent_name,
                optimize_knowledge=config.optimize_knowledge,
                verbose=config.verbose,
                show_output=config.show_output,
                user_prompt=config.user_prompt,
                log_tag="pattern-validation-simulate",
                workflow="simulate",
            ),
            workspace_plan_path=workspace_plan_path,
        )
        if prepare_code != 0:
            print(
                "[pattern-validation-simulate] prepare agent failed.",
                file=sys.stderr,
            )
            return prepare_code
    elif config.verbose:
        print(
            f"[pattern-validation-simulate] reusing {active_count} existing workspace(s)",
            file=out,
            flush=True,
        )

    return prepare_simulate_batch(config, stream=out, run_verify=not config.skip_verify)


def prepare_simulate_batch(
    config: SimulatePlanConfig,
    *,
    stream: TextIO | None = None,
    run_verify: bool = False,
) -> int:
    out = stream or sys.stderr
    if not config.batch_path.is_dir():
        print(
            f"[pattern-validation-simulate] batch directory not found: {config.batch_path}",
            file=sys.stderr,
        )
        return 2

    sync_code = sync_batch_workspace_dependencies(
        config.batch_path,
        config.repo_root,
        stream=out,
    )
    if sync_code != 0:
        print(
            "[pattern-validation-simulate] dependency sync failed; fix imports before simulate.",
            file=sys.stderr,
        )
        return sync_code

    if run_verify and not config.skip_verify:
        verify_code = run_pattern_validation_verify(config.batch_path, stream=out)
        if verify_code != 0:
            print(
                "[pattern-validation-simulate] scaffold verify failed; fix workspaces first.",
                file=sys.stderr,
            )
            return verify_code
        print(
            "[pattern-validation-simulate] bootstrap complete: deps synced, scaffold verified",
            file=out,
            flush=True,
        )
    elif run_verify:
        print(
            "[pattern-validation-simulate] bootstrap complete: deps synced (--skip-verify)",
            file=out,
            flush=True,
        )
    return 0


def run_simulate_plan_batch(
    config: SimulatePlanConfig,
    *,
    stream: TextIO | None = None,
    run_prepare: bool = True,
) -> tuple[int, Path]:
    out = stream or sys.stderr
    batch_report_path = config.batch_path / BATCH_SIMULATE_REPORT_FILENAME
    if run_prepare:
        prep_code = prepare_simulate_batch(config, stream=out)
        if prep_code != 0:
            return prep_code, batch_report_path

    results, agent_code = run_simulate_workspace_agents(config, stream=out)
    batch_report_path = write_batch_simulate_report(config.batch_path, results)
    failed = [item for item in results if item.status != "ok"]
    print_batch_summary(config, batch_report_path, failed_count=len(failed), stream=out)

    exit_code = 1 if agent_code != 0 else 0
    return exit_code, batch_report_path


def write_batch_simulate_report(
    batch_path: Path,
    results: list[WorkspaceSimulateResult],
) -> Path:
    try:
        batch_eval_mod = load_skill_script_module("triton-npu-pattern-validation-loop", "batch_evaluation")
    except Exception:
        batch_eval_mod = None

    workspace_reports: list[dict[str, Any]] = []
    for item in results:
        entry: dict[str, Any] = {
            "workspace": item.workspace.name,
            "status": item.status,
            "message": item.message,
            "report_path": item.report_path.as_posix() if item.report_path else None,
        }
        expected_patterns: list[str] = []
        if batch_eval_mod is not None:
            try:
                meta = batch_eval_mod.resolve_workspace_meta(item.workspace, batch_root=batch_path)
                expected_patterns = [str(p) for p in meta.get("expected_patterns", [])]
            except Exception:
                pass

        simulate_pattern_hits: list[str] = []
        if item.report_path is not None and item.report_path.is_file():
            try:
                payload = json.loads(item.report_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    entry["simulate_report"] = payload
                    validation_errors = validate_simulate_report(payload)
                    if validation_errors:
                        entry["validation_errors"] = validation_errors

                    ranked = payload.get("ranked_patterns", [])
                    if isinstance(ranked, list):
                        for pat in ranked:
                            if isinstance(pat, dict) and pat.get("hit") is True:
                                pat_id = pat.get("pattern_id")
                                if pat_id:
                                    simulate_pattern_hits.append(str(pat_id))
            except json.JSONDecodeError:
                entry["simulate_report_error"] = "invalid JSON in workspace report"

        matched = [p for p in expected_patterns if p in simulate_pattern_hits]
        missing = [p for p in expected_patterns if p not in matched]

        heuristic_suggested_pass = False
        if expected_patterns:
            heuristic_suggested_pass = (not missing) and item.status == "ok" and not entry.get("validation_errors")
        else:
            heuristic_suggested_pass = item.status == "ok" and not entry.get("validation_errors")

        entry["expected_patterns"] = expected_patterns
        entry["heuristic_pattern_hits"] = matched
        entry["heuristic_missing_patterns"] = missing
        entry["heuristic_suggested_pass"] = heuristic_suggested_pass
        workspace_reports.append(entry)

    active_remaining = [
        entry["workspace"] for entry in workspace_reports if not entry.get("heuristic_suggested_pass")
    ]

    payload = {
        "schema_version": 1,
        "batch_root": batch_path.as_posix(),
        "workspace_count": len(results),
        "ok_count": sum(1 for item in results if item.status == "ok"),
        "failed_count": sum(1 for item in results if item.status != "ok"),
        "workspaces": workspace_reports,
        "reports": workspace_reports,  # Alias for consistency with audit-report.json
        "active_remaining": active_remaining,
        "completed_total": len(results) - len(active_remaining),
        "next_step_manual_optimize": build_manual_optimize_command_hint(batch_path),
    }
    report_path = batch_path / BATCH_SIMULATE_REPORT_FILENAME
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_path


def remove_batch_workspace_simulate_plans(batch_path: Path) -> list[str]:
    """Delete per-workspace simulate-plan dirs after analyze, before the next optimize round."""
    batch_path = batch_path.expanduser().resolve()
    discovered, _failures = discover_batch_workspaces(
        batch_path,
        resolve_operator_file=resolve_batch_optimize_operator_file,
    )
    removed: list[str] = []
    for workspace, _operator in discovered:
        plan_dir = workspace / SIMULATE_PLAN_DIR
        if not plan_dir.is_dir():
            continue
        shutil.rmtree(plan_dir)
        removed.append(workspace.name)
    return removed


def load_batch_simulate_report(batch_path: Path) -> dict[str, Any] | None:
    report_path = batch_path / BATCH_SIMULATE_REPORT_FILENAME
    if not report_path.is_file():
        return None
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def validate_simulate_report(simulate_report: dict[str, Any]) -> list[str]:
    """CLI structural checks — do not trust agent self-assessment alone."""
    errors: list[str] = []
    ranked = simulate_report.get("ranked_patterns")
    if not isinstance(ranked, list) or not ranked:
        errors.append("ranked_patterns missing or empty")

    proposed = simulate_report.get("proposed_code_changes")
    if not isinstance(proposed, dict):
        errors.append("proposed_code_changes missing")
        proposed = {}

    diff = str(proposed.get("unified_diff", "")).strip()
    if not (diff.startswith("---") and "+++" in diff):
        errors.append("proposed_code_changes.unified_diff missing or not a unified diff")

    edits = proposed.get("edits_by_pattern")
    if not isinstance(edits, list) or not edits:
        errors.append("proposed_code_changes.edits_by_pattern missing or empty")

    if isinstance(ranked, list):
        hit_ids: list[str] = []
        for entry in ranked:
            if isinstance(entry, dict) and entry.get("hit") is True:
                pattern_id = str(entry.get("pattern_id", "")).strip()
                if pattern_id:
                    hit_ids.append(pattern_id)
        edit_ids = {
            str(entry.get("pattern_id", "")).strip()
            for entry in (edits if isinstance(edits, list) else [])
            if isinstance(entry, dict) and str(entry.get("pattern_id", "")).strip()
        }
        for pattern_id in hit_ids:
            if pattern_id not in edit_ids:
                errors.append(f"hit pattern {pattern_id!r} has no edits_by_pattern entry")
            else:
                edit = next(
                    item
                    for item in edits
                    if isinstance(item, dict) and str(item.get("pattern_id", "")).strip() == pattern_id
                )
                before = str(edit.get("before_excerpt", "")).strip()
                after = str(edit.get("after_excerpt", "")).strip()
                if not before or not after:
                    errors.append(
                        f"edits_by_pattern for {pattern_id!r} missing before_excerpt or after_excerpt",
                    )

    alignment = str(simulate_report.get("skills_alignment", "")).strip().lower()
    if alignment not in {"aligned", "partial", "mismatch"}:
        errors.append("skills_alignment must be aligned | partial | mismatch")

    quality = str(simulate_report.get("code_plan_quality", "")).strip().lower()
    if quality not in {"concrete", "vague", "missing"}:
        errors.append("code_plan_quality must be concrete | vague | missing")

    return errors


def _load_simulate_report_payload(report_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not report_path.is_file():
        return None, [f"missing report file: {report_path.as_posix()}"]
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, ["invalid JSON in workspace report"]
    if not isinstance(payload, dict):
        return None, ["workspace report must be a JSON object"]
    return payload, validate_simulate_report(payload)


def _simulate_report_code_plan_concrete(simulate_report: dict[str, Any]) -> bool:
    quality = str(simulate_report.get("code_plan_quality", "")).strip().lower()
    if quality == "concrete":
        return True
    if quality in {"vague", "missing"}:
        return False
    proposed = simulate_report.get("proposed_code_changes")
    if isinstance(proposed, dict):
        diff = str(proposed.get("unified_diff", "")).strip()
        if diff.startswith("---") and "+++" in diff:
            return True
    legacy = str(simulate_report.get("proposed_changes", "")).strip()
    return legacy.startswith("---") and "+++" in legacy


def batch_report_skills_aligned(payload: dict[str, Any]) -> bool:
    workspaces = payload.get("workspaces")
    if not isinstance(workspaces, list) or not workspaces:
        return False
    for entry in workspaces:
        if not isinstance(entry, dict):
            return False
        if entry.get("status") != "ok":
            return False
        validation_errors = entry.get("validation_errors")
        if isinstance(validation_errors, list) and validation_errors:
            return False
        simulate_report = entry.get("simulate_report")
        if not isinstance(simulate_report, dict):
            return False
        if validate_simulate_report(simulate_report):
            return False
        alignment = str(simulate_report.get("skills_alignment", "")).strip().lower()
        if alignment != "aligned":
            return False
        if not _simulate_report_code_plan_concrete(simulate_report):
            return False
    return True


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
            "[pattern-validation-simulate] simulate loop complete after skill-audit review "
            "(all workspaces structurally valid and skills_alignment=aligned).",
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


def _maybe_migrate_legacy_batch_evaluation(batch_path: Path, *, stream: TextIO) -> None:
    module = load_skill_script_module(
        "triton-npu-pattern-validation-loop",
        "batch_evaluation",
    )
    migrated = module.migrate_legacy_workspace_meta(batch_path)
    if migrated and stream:
        print(
            f"[pattern-validation-simulate] migrated {len(migrated)} legacy validation-meta "
            f"entries into {module.BATCH_EVALUATION_FILENAME}",
            file=stream,
            flush=True,
        )


def _active_validation_workspace_count(batch_path: Path) -> int:
    module = load_skill_script_module(
        "triton-npu-pattern-validation-loop",
        "batch_layout",
    )
    return len(module.list_active_validation_workspaces(batch_path))


def _load_validation_meta(workspace: Path) -> dict[str, Any] | None:
    meta_path = workspace / "validation-meta.json"
    if not meta_path.is_file():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None
