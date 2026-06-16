from __future__ import annotations

import shutil
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import TextIO

from triton_agent.diff_skills_update.agent import run_diff_skills_agent
from triton_agent.diff_skills_update.discovery import discover_operator_pairs
from triton_agent.diff_skills_update.models import (
    DiffAgentOutput,
    DiffSkillsUpdateConfig,
    IterationReport,
    OperatorPair,
    PairRunResult,
    SkipRecord,
    Status,
)
from triton_agent.diff_skills_update.prompts import (
    build_analysis_prompt,
    build_diff_to_skill_prompt,
    build_simulate_prompt,
)
from triton_agent.diff_skills_update.reports import (
    read_json_file,
    report_path_for_pair,
    write_pair_report,
    write_skip_report,
)
from triton_agent.diff_skills_update.skills_workspace import (
    ensure_skills_workspace,
    promote_converged_knowledge_workspace,
    regenerate_pattern_index,
)
from triton_agent.models import AgentResult

AgentRunner = Callable[..., AgentResult]


def run_diff_skills_update(
    config: DiffSkillsUpdateConfig,
    *,
    agent_runner: AgentRunner = run_diff_skills_agent,
    stream: TextIO | None = None,
) -> list[PairRunResult]:
    output_stream = stream or sys.stderr
    discovery = discover_operator_pairs(
        config.input_root,
        stream=output_stream,
        exclude_dirs={config.skills_dir},
    )
    knowledge_dir = ensure_skills_workspace(config.skills_dir)
    pair_counts = Counter(pair.operator_dir for pair in discovery.pairs)
    skills_lock = Lock()
    for skip in discovery.skips:
        _write_skip_report(skip)
    if not discovery.pairs:
        print("No valid operator pairs found.", file=output_stream)
        return []

    if config.concurrency <= 1:
        return [
            _run_pair(
                pair,
                config=config,
                knowledge_dir=knowledge_dir,
                pair_count_in_dir=pair_counts[pair.operator_dir],
                agent_runner=agent_runner,
                skills_lock=skills_lock,
                stream=output_stream,
            )
            for pair in discovery.pairs
        ]

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
            for pair in discovery.pairs
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_pair(
    pair: OperatorPair,
    *,
    config: DiffSkillsUpdateConfig,
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

    diff_output = simulate_dir / f"diff-skills-{pair.stem}.json"
    diff_prompt = build_diff_to_skill_prompt(pair, skills_dir=config.skills_dir, output_json=diff_output)
    with skills_lock:
        diff_result = agent_runner(
            agent_name=config.agent_name,
            workdir=pair.operator_dir,
            prompt=diff_prompt,
            show_output=config.show_output,
            verbose=config.verbose,
            output_label=f"[{pair.operator_dir.name}] [diff-skills]",
        )
        if diff_result.return_code == 0:
            _regenerate_if_possible(knowledge_dir)
    if diff_result.return_code != 0:
        result = PairRunResult(
            pair=pair,
            status="failed",
            matched_patterns=[],
            updated_patterns=[],
            iterations=[],
            report_path=report_path,
            message="diff-to-skill agent failed",
        )
        write_pair_report(result)
        return result

    diff_output_data = _read_diff_output(diff_output)
    matched_patterns = diff_output_data.matched_patterns
    updated_patterns = list(diff_output_data.updated_patterns)
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
            show_output=config.show_output,
            verbose=config.verbose,
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
                show_output=config.show_output,
                verbose=config.verbose,
                output_label=f"[{pair.operator_dir.name}] [analyze-iter-{iteration}/{config.max_iterations}]",
            )
            if analysis_result.return_code == 0:
                _regenerate_if_possible(knowledge_dir)
            analysis_data = read_json_file(analysis_output)
            aligned = bool(analysis_data.get("aligned"))
            if analysis_result.return_code == 0 and aligned and config.promote_converged_skills:
                promoted_dir = promote_converged_knowledge_workspace(knowledge_dir)
                print(f"[{pair.operator_dir.name}] promote-converged-skills: {promoted_dir}", file=stream)
        if analysis_result.return_code != 0:
            aligned = False
        analysis_summary = str(analysis_data.get("summary") or "")
        iteration_updated_patterns = _updated_patterns_from_analysis(analysis_data)
        updated_patterns = _merge_unique(updated_patterns, iteration_updated_patterns)
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


def _read_diff_output(path: Path) -> DiffAgentOutput:
    data = read_json_file(path)
    return DiffAgentOutput(
        matched_patterns=list(_string_list(data.get("matched_patterns"))),
        updated_patterns=list(_string_list(data.get("updated_patterns"))),
        summary=str(data.get("summary") or ""),
    )


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


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


def _write_skip_report(record: SkipRecord) -> None:
    simulate_dir = record.operator_dir / "simulate"
    if record.opt_path is None:
        report_path = simulate_dir / "report.json"
    else:
        stem = record.opt_path.name.removeprefix("opt_").removesuffix(".py")
        report_path = simulate_dir / f"report_{stem}.json"
    write_skip_report(record, report_path)


def _regenerate_if_possible(knowledge_dir: Path) -> None:
    try:
        regenerate_pattern_index(knowledge_dir)
    except Exception as exc:
        print(f"Warning: pattern index regeneration failed: {exc}", file=sys.stderr)


def _delete_unaligned_candidate(candidate_path: Path) -> None:
    if candidate_path.exists():
        candidate_path.unlink()
