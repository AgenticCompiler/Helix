from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import TextIO, cast

from triton_agent.distill.agent import run_distill_agent
from triton_agent.distill.discovery import discover_operator_pairs
from triton_agent.distill.models import (
    DiffAgentOutput,
    DistillConfig,
    IterationReport,
    OperatorPair,
    PairRunResult,
    SkipRecord,
    Status,
)
from triton_agent.distill.prompts import (
    build_analysis_prompt,
    build_distill_prompt,
    build_simulate_prompt,
)
from triton_agent.distill.reports import (
    read_json_file,
    report_path_for_pair,
    write_pair_report,
    write_skip_report,
)
from triton_agent.distill.knowledge_workspace import (
    ensure_editable_knowledge_skill,
    export_changed_pattern_cards,
    find_pattern_card,
    promote_editable_knowledge_skill,
    rebuild_pattern_index,
    snapshot_pattern_card_texts,
)
from triton_agent.distill.git_repo_workspaces import (
    DEFAULT_OPERATOR_WORKSPACES_DIR,
    DEFAULT_WORKSPACE_PLAN_NAME,
    GIT_REPO_PLAN_SKILL_NAME,
    build_workspace_plan_prompt,
    compute_fork_point,
    detect_default_base_branch,
    detect_git_worktree,
    operator_workspaces_created,
    scaffold_operator_workspaces,
)
from triton_agent.models import AgentResult

AgentRunner = Callable[..., AgentResult]


def run_distill(
    config: DistillConfig,
    *,
    agent_runner: AgentRunner = run_distill_agent,
    stream: TextIO | None = None,
) -> list[PairRunResult]:
    output_stream = stream or sys.stderr

    # ── Phase 1 (git-repo): Agent → workspace-plan.json → scaffold ─────
    discovery_root = config.input_root
    if config.source == "git-repo":
        git_info = detect_git_worktree(config.input_root)
        if git_info is None:
            print(
                "[git-repo] Input is not inside a Git work tree. "
                "Use --source code-diff for pre-organized operator directories.",
                file=output_stream,
            )
            return []
        repo_root, _head_sha = git_info

        # Resolve the base branch: use explicit --base, or auto-detect from remote
        base_branch = config.base_revision or detect_default_base_branch(repo_root=repo_root)
        if config.base_revision:
            print(
                f"[git-repo] Using base branch: {base_branch}",
                file=output_stream,
            )
        else:
            print(
                f"[git-repo] Auto-detected base branch: {base_branch}",
                file=output_stream,
            )

        # Deterministically compute the fork point before calling the agent
        fork_revision = compute_fork_point(
            repo_root=repo_root, base_branch=base_branch
        )
        if fork_revision is None:
            print(
                f"[git-repo] Failed to compute merge-base "
                f"({base_branch}..HEAD). "
                f"Ensure the base branch ref exists (e.g. `git fetch` first).",
                file=output_stream,
            )
            return []
        print(
            f"[git-repo] Fork point (merge-base {base_branch}..HEAD): "
            f"{fork_revision[:12]}...",
            file=output_stream,
        )

        plan_path = config.input_root / ".triton-agent" / DEFAULT_WORKSPACE_PLAN_NAME
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        organize_prompt = build_workspace_plan_prompt(
            repo_root=repo_root,
            language=config.language,
            base_revision=base_branch,
            fork_revision=fork_revision,
            plan_path=plan_path,
        )
        print(
            "[git-repo] Running agent to produce workspace plan...",
            file=output_stream,
        )
        plan_result = agent_runner(
            agent_name=config.agent_name,
            workdir=config.input_root,
            prompt=organize_prompt,
            stream_output=config.stream_output,
            verbose=config.verbose,
            language=config.language,
            skills_root=config.skills_dir,
            repository_skill_names=(GIT_REPO_PLAN_SKILL_NAME,),
            stage_editable_knowledge=False,
            output_label="[git-repo]",
        )
        if plan_result.return_code != 0 or not plan_path.is_file():
            print(
                "[git-repo] Agent failed to produce workspace plan.",
                file=output_stream,
            )
            return []
        print(
            f"[git-repo] Plan written to {plan_path.as_posix()}",
            file=output_stream,
        )

        organized_dir = config.input_root / DEFAULT_OPERATOR_WORKSPACES_DIR
        print(
            "[git-repo] Running scaffold script to create operator workspaces...",
            file=output_stream,
        )
        scaffold_rc = scaffold_operator_workspaces(
            plan_path=plan_path,
            output_root=organized_dir,
            base_revision=base_branch,
            fork_revision=fork_revision,
            stream=output_stream,
        )
        if scaffold_rc != 0 or not operator_workspaces_created(organized_dir):
            print(
                "[git-repo] Scaffold script failed to create operator workspaces.",
                file=output_stream,
            )
            return []
        discovery_root = organized_dir
        print(
            f"[git-repo] Workspaces created in {organized_dir.as_posix()}",
            file=output_stream,
        )

        # Clean up intermediate .triton-agent/ directory — no longer needed
        triton_agent_dir = config.input_root / ".triton-agent"
        if triton_agent_dir.is_dir():
            shutil.rmtree(triton_agent_dir)
            print(
                f"[git-repo] Cleaned up intermediate {triton_agent_dir.as_posix()}",
                file=output_stream,
            )

    # ── Phase 2: Operator Pair Discovery ────────────────────────────────
    discovery = discover_operator_pairs(
        discovery_root,
        source=config.source,
        stream=output_stream,
        exclude_dirs={config.skills_dir, config.update_skills_dir},
    )
    knowledge_dir = ensure_editable_knowledge_skill(config.skills_dir, language=config.language)
    pattern_snapshot = snapshot_pattern_card_texts(knowledge_dir)
    for skip in discovery.skips:
        _write_skip_report(skip)
    if not discovery.pairs:
        print("No valid operator pairs found.", file=output_stream)
        return []

    # ── Phase 3: Validate operator pairs ───────────────────────────────
    validated_pairs: list[OperatorPair] = []
    for pair in discovery.pairs:
        if not pair.baseline_path.is_file():
            print(
                f"skip {pair.operator_dir}: baseline file not found: {pair.baseline_path}",
                file=output_stream,
            )
            continue
        if not pair.expected_path.is_file():
            print(
                f"skip {pair.operator_dir}: expected file not found: {pair.expected_path}",
                file=output_stream,
            )
            continue
        validated_pairs.append(pair)
    if not validated_pairs:
        print(
            f"All {len(discovery.pairs)} discovered pair(s) failed validation.",
            file=output_stream,
        )
        return []
    if len(validated_pairs) < len(discovery.pairs):
        print(
            f"Validated {len(validated_pairs)}/{len(discovery.pairs)} operator pairs.",
            file=output_stream,
        )

    # ── Phase 4: Simulate→Analyze per operator ─────────────────────────
    pair_counts = Counter(pair.operator_dir for pair in validated_pairs)
    skills_lock = Lock()

    if config.concurrency <= 1:
        results = [
            _run_pair(
                pair,
                config=config,
                knowledge_dir=knowledge_dir,
                pair_count_in_dir=pair_counts[pair.operator_dir],
                agent_runner=agent_runner,
                skills_lock=skills_lock,
                stream=output_stream,
            )
            for pair in validated_pairs
        ]
    else:
        results: list[PairRunResult] = []
        with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
            futures = {
                executor.submit(
                    _run_pair,
                    pair,
                    config=config,
                    knowledge_dir=knowledge_dir,
                    pair_count_in_dir=pair_counts[pair.operator_dir],
                    agent_runner=agent_runner,
                    skills_lock=skills_lock,
                    stream=output_stream,
                ): pair
                for pair in validated_pairs
            }
            for future in as_completed(futures):
                results.append(future.result())

    # ── Phase 5: Export updated skills summary ─────────────────────────

    updated_pattern_names = _merge_unique(
        [],
        [name for result in results for name in result.updated_patterns],
    )
    exported = export_changed_pattern_cards(
        knowledge_dir,
        config.update_skills_dir,
        language=config.language,
        pattern_snapshot=pattern_snapshot,
        updated_pattern_names=updated_pattern_names,
    )
    if exported:
        print(
            f"exported updated patterns: {', '.join(exported)} -> {config.update_skills_dir}",
            file=output_stream,
        )
    else:
        print("no pattern cards were changed.", file=output_stream)

    _write_enriched_manifest(
        config=config,
        knowledge_dir=knowledge_dir,
        exported=exported,
        updated_pattern_names=updated_pattern_names,
        results=results,
    )

    # ── Final skills summary ─────────────────────────────────────────
    aligned = sum(1 for r in results if r.status == "aligned")
    not_aligned = sum(1 for r in results if r.status == "not_aligned")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    print(
        f"\n[distill] summary: "
        f"{aligned} aligned, {not_aligned} not-aligned, "
        f"{failed} failed, {skipped} skipped "
        f"(total {len(results)} pairs)",
        file=output_stream,
    )
    return results


def _run_pair(
    pair: OperatorPair,
    *,
    config: DistillConfig,
    knowledge_dir: Path,
    pair_count_in_dir: int,
    agent_runner: AgentRunner,
    skills_lock: Lock,
    stream: TextIO,
) -> PairRunResult:
    simulate_dir = pair.operator_dir / "simulate"
    simulate_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_path_for_pair(pair.stem, simulate_dir, pair_count_in_dir=pair_count_in_dir)
    if config.skip_existing and not config.force:
        existing = read_json_file(report_path)
        if existing.get("status") == "aligned":
            print(f"skip {pair.operator_dir}: existing aligned report {report_path}", file=stream)
            return PairRunResult(
                pair=pair,
                status="skipped",
                matched_patterns=list(_string_list(existing.get("matched_patterns"))),
                updated_patterns=list(_string_list(existing.get("updated_patterns"))),
                iterations=[],
                report_path=report_path,
                message="existing aligned report",
            )

    baseline_copy = simulate_dir / pair.baseline_path.name
    shutil.copy2(pair.baseline_path, baseline_copy)

    distill_output = simulate_dir / f"distill-{pair.stem}.json"
    distill_prompt = build_distill_prompt(pair, skills_dir=config.skills_dir, output_json=distill_output, language=config.language)
    with skills_lock:
        distill_result = agent_runner(
            agent_name=config.agent_name,
            workdir=pair.operator_dir,
            prompt=distill_prompt,
            stream_output=config.stream_output,
            verbose=config.verbose,
            language=config.language,
            skills_root=config.skills_dir,
            output_label=f"[{pair.operator_dir.name}] [distill]",
        )
        if distill_result.return_code == 0:
            _regenerate_if_possible(knowledge_dir)
    if distill_result.return_code != 0:
        result = PairRunResult(
            pair=pair,
            status="failed",
            matched_patterns=[],
            updated_patterns=[],
            iterations=[],
            report_path=report_path,
            message="distill agent failed",
        )
        write_pair_report(result)
        return result

    distill_output_data = _read_distill_output(distill_output)
    matched_patterns = _merge_unique(
        distill_output_data.matched_patterns,
        distill_output_data.updated_patterns,
    )
    updated_patterns = list(distill_output_data.updated_patterns)

    # optimize-process: skip simulate-analyze if all optimizations already covered
    if pair.source_kind == "optimize-process" and distill_output_data.aligned:
        result = PairRunResult(
            pair=pair,
            status="aligned",
            matched_patterns=matched_patterns,
            updated_patterns=updated_patterns,
            iterations=[],
            report_path=report_path,
            message="all optimizations already covered by existing skills",
        )
        write_pair_report(result)
        return result

    iterations: list[IterationReport] = []
    status: Status = "failed"
    message = "max iterations reached"

    for iteration in range(1, config.max_iterations + 1):
        candidate_path = simulate_dir / f"generated_{pair.stem}.py"
        simulate_output = simulate_dir / f"simulate-{pair.stem}-{iteration}.json"
        simulate_prompt = build_simulate_prompt(
            baseline_filename=baseline_copy.name,
            candidate_filename=candidate_path.name,
            matched_patterns=matched_patterns,
            output_json=simulate_output,
        )
        print(
            f"[{pair.operator_dir.name}] [simulate-iter-{iteration}/{config.max_iterations}]: "
            f"{pair.baseline_path.name}",
            file=stream,
        )
        simulate_result = agent_runner(
            agent_name=config.agent_name,
            workdir=simulate_dir,
            prompt=simulate_prompt,
            stream_output=config.stream_output,
            verbose=config.verbose,
            language=config.language,
            skills_root=config.skills_dir,
            output_label=f"[{pair.operator_dir.name}] [simulate-iter-{iteration}/{config.max_iterations}]",
        )
        if simulate_result.return_code != 0:
            iterations.append(
                IterationReport(
                    iteration=iteration,
                    status="failed",
                    candidate_path=candidate_path,
                    simulate_return_code=simulate_result.return_code,
                    analysis_return_code=0,
                    analysis_summary="simulate agent failed",
                    updated_patterns=[],
                )
            )
            status = "failed"
            message = "simulate agent failed"
            break

        analysis_output = simulate_dir / f"analysis-{pair.stem}-{iteration}.json"
        analysis_prompt = build_analysis_prompt(
            pair=pair,
            candidate_path=candidate_path,
            skills_dir=config.skills_dir,
            output_json=analysis_output,
        )
        print(
            f"[{pair.operator_dir.name}] [analyze-iter-{iteration}/{config.max_iterations}]: "
            f"{pair.baseline_path.name}",
            file=stream,
        )
        with skills_lock:
            analysis_result = agent_runner(
                agent_name=config.agent_name,
                workdir=pair.operator_dir,
                prompt=analysis_prompt,
                stream_output=config.stream_output,
                verbose=config.verbose,
                language=config.language,
                skills_root=config.skills_dir,
                output_label=f"[{pair.operator_dir.name}] [analyze-iter-{iteration}/{config.max_iterations}]",
            )
            if analysis_result.return_code == 0:
                _regenerate_if_possible(knowledge_dir)
            analysis_data = read_json_file(analysis_output)
            aligned = bool(analysis_data.get("aligned"))
            if analysis_result.return_code == 0 and aligned and config.promote_converged_skills:
                promoted_dir = promote_editable_knowledge_skill(
                    knowledge_dir,
                    language=config.language,
                )
                print(f"[{pair.operator_dir.name}] promote-aligned: {promoted_dir}", file=stream)
        if analysis_result.return_code != 0:
            aligned = False
        analysis_summary = str(analysis_data.get("summary") or "")
        iteration_updated_patterns = _updated_patterns_from_analysis(analysis_data)
        updated_patterns = _merge_unique(updated_patterns, iteration_updated_patterns)
        matched_patterns = _merge_unique(matched_patterns, iteration_updated_patterns)
        current_status: Status = "aligned" if aligned else "not_aligned"
        if analysis_result.return_code != 0:
            current_status = "failed"
            analysis_summary = analysis_summary or "analysis agent failed"
        iterations.append(
            IterationReport(
                iteration=iteration,
                status=current_status,
                candidate_path=candidate_path,
                simulate_return_code=simulate_result.return_code,
                analysis_return_code=analysis_result.return_code,
                analysis_summary=analysis_summary,
                updated_patterns=iteration_updated_patterns,
            )
        )
        if analysis_result.return_code != 0:
            status = "failed"
            message = "analysis agent failed"
            break
        if aligned:
            status = "aligned"
            message = "candidate aligned with expected answer"
            break
        status = "not_aligned"
        message = "candidate not aligned"
        _delete_unaligned_candidate(candidate_path)

    result = PairRunResult(
        pair=pair,
        status=status,
        matched_patterns=matched_patterns,
        updated_patterns=updated_patterns,
        iterations=iterations,
        report_path=report_path,
        message=message,
    )
    write_pair_report(result)
    return result


def _read_distill_output(path: Path) -> DiffAgentOutput:
    data = read_json_file(path)
    return DiffAgentOutput(
        matched_patterns=list(_string_list(data.get("matched_patterns"))),
        updated_patterns=list(_string_list(data.get("updated_patterns"))),
        summary=str(data.get("summary") or ""),
        aligned=bool(data.get("aligned")),
    )


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in cast(list[object], value) if isinstance(item, str))


def _updated_patterns_from_analysis(data: dict[str, object]) -> list[str]:
    updated = list(_string_list(data.get("updated_patterns")))
    if updated:
        return updated
    return list(_string_list(data.get("skill_updates")))


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


def _write_enriched_manifest(
    *,
    config: DistillConfig,
    knowledge_dir: Path,
    exported: list[str],
    updated_pattern_names: list[str],
    results: list[PairRunResult],
) -> None:
    """Write updated_patterns.json with per-operator exported pattern mapping."""
    operator_records: list[dict[str, object]] = []
    for result in results:
        operator_exported: list[str] = []
        for name in result.updated_patterns:
            resolved = find_pattern_card(knowledge_dir, name)
            if resolved is not None and resolved.name in exported:
                if resolved.name not in operator_exported:
                    operator_exported.append(resolved.name)
        operator_records.append({
            "operator_dir": str(result.pair.operator_dir),
            "status": result.status,
            "matched_patterns": result.matched_patterns,
            "updated_patterns": result.updated_patterns,
            "exported_patterns": operator_exported,
        })
    manifest = {
        "exported_patterns": exported,
        "updated_pattern_names": updated_pattern_names,
        "operators": operator_records,
    }
    config.update_skills_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.update_skills_dir / "updated_patterns.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_skip_report(record: SkipRecord) -> None:
    simulate_dir = record.operator_dir / "simulate"
    if record.opt_path is None:
        report_path = simulate_dir / "report.json"
    else:
        stem = record.opt_path.name.removeprefix("opt_").removesuffix(".py")
        report_path = simulate_dir / f"report_{stem}.json"
    write_skip_report(record, report_path)


def _regenerate_if_possible(knowledge_dir: Path) -> None:
    rebuild_pattern_index(knowledge_dir)


def _delete_unaligned_candidate(candidate_path: Path) -> None:
    if candidate_path.exists():
        candidate_path.unlink()
