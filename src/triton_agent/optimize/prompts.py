from __future__ import annotations

from pathlib import Path
from typing import Literal

from triton_agent.optimize.subagents import optimize_subagent_recommendation_lines
from triton_agent.optimize.contract import baseline_state_contract_lines


def _display_path(path: Path) -> str:
    return path.as_posix()


def strict_learned_lessons_lines() -> list[str]:
    return [
        "`learned_lessons.md` is only for reusable, evidence-backed optimization or profiling rules that can transfer to related Triton Ascend NPU operators.",
        "Do not put round narrative, command failures, or operator-specific details in `learned_lessons.md`; keep those in `attempts.md`, `summary.md`, or `opt-note.md`.",
    ]


def sequential_round_execution_lines() -> list[str]:
    return [
        "Complete optimize rounds strictly one at a time in sequence.",
        "Only one optimize round may be active at a time.",
        "Do not use subagents to implement or advance multiple optimize rounds in parallel.",
        "Subagents may help with supporting analysis, but they must not create, repair, or progress a different optimize round while the current round is still in flight.",
    ]


def next_round_reflection_lines() -> list[str]:
    return [
        "Use the staged `triton-npu-optimize-start-round` skill before opening the next round.",
        "Before editing code for the next round, stop and reflect on the best entrypoint.",
        "Choose which operator, kernel path, or wrapper bottleneck should anchor the round before making the next code change.",
        "Decide whether existing benchmark and compare-perf evidence is already sufficient or whether profiling is needed first.",
        "Escalate to IR only after profiler evidence narrows the bottleneck but still does not explain it.",
        "Use compiler-source analysis only after profiler and IR evidence have narrowed a concrete compiler-side question.",
        "Do not use agents or subagents to optimize multiple rounds in parallel; keep the optimize session one round at a time.",
        "Do not treat the next round as a parameter-only tuning sweep; make a bottleneck-backed change instead. "
"Do not use a script to create multiple optimize rounds where each round only adjusts parameters in order to speed up the optimization process. "
"This is cheating behavior and is strictly prohibited.",
    ]


def layered_analysis_lines(*, round_scope: str) -> list[str]:
    return [
        f"Choose the analysis level for {round_scope} before editing code.",
        "Record the round's primary analysis level separately from its supporting evidence.",
        "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
        "Use pattern triage only to decide whether a strong pattern-backed hypothesis already exists.",
        "Use the staged `triton-npu-optimize-knowledge` skill for generic pattern and symptom references.",
        "When pattern triage is used, record candidate patterns, the selected pattern if one is chosen, and why that pattern looks plausible in `opt-round-N/attempts.md`.",
        "When a named pattern guides the round, record the final selected pattern direction in `opt-round-N/summary.md`.",
        "Read the staged `triton-npu-optimize-knowledge` skill's generated `references/pattern_index.md` before detailed pattern references.",
        "Inspect the operator file directly when code structure is still unclear at pattern triage.",
        "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
        "Use the staged `triton-npu-optimize-knowledge` skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists.",
        "Use IR attribution only after profiler-backed symptoms need explanation.",
        "Use compiler-source escalation only when profiler and IR evidence have already narrowed the issue.",
        "When starting from a deeper level, cite the reused evidence path and explain why the shallower level is already established or insufficient.",
        "Do not begin with blind tiling or launch-parameter search.",
    ]


def compiler_source_analysis_lines(
    *,
    compiler_source_path: Path | None,
    compiler_source_commit: str | None,
) -> list[str]:
    if compiler_source_path is None or compiler_source_commit is None:
        return []
    return [
        "Compiler source analysis is enabled for this optimize run.",
        f"Compiler source path: {_display_path(compiler_source_path)}",
        f"Compiler source commit: {compiler_source_commit}.",
        "Treat the compiler source checkout as read-only.",
        "Do not run git clone, git fetch, git pull, or modify files in the compiler source checkout.",
        "Use the staged `triton-npu-analyze-compiler-source` skill only when compiler source evidence is needed.",
        "Prefer the evidence ladder first: benchmark and correctness results, then profiler evidence, then IR evidence, then compiler source.",
    ]


def cann_ext_api_lines(*, enabled: bool) -> list[str]:
    if not enabled:
        return []
    return [
        "CANN Triton extension API pattern access is enabled for this optimize run.",
        "Use the staged `triton-npu-cann-ext-api-patterns` skill for the specialized A5-only pattern guidance.",
        "Treat these extension APIs and patterns as a high-value optimization direction when the kernel structure matches.",
        "Give serious attention to whether CANN extension API patterns can improve this kernel instead of treating them as an edge-case option.",
    ]


def optimize_target_lines(*, optimize_target: str) -> list[str]:
    if optimize_target == "operator":
        return [
            "Target optimization scope for this optimize session: operator.",
            "Optimize end-to-end operator latency.",
            "You may optimize wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code together.",
            "Preserve a real Triton Ascend NPU computation path.",
            "A pure PyTorch rewrite that bypasses the Triton Ascend NPU path does not count as a successful optimize round.",
            "When comparing round performance, run `compare-perf` so both kernel and total-op views are visible.",
            "Use the total-op view as the canonical round conclusion and record `effective_metric_source=total-op` in `round-state.json`.",
            "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
        ]
    return [
        "Target optimization scope for this optimize session: kernel.",
        "PyTorch-facing public API may remain as a wrapper when that is the intended operator entrypoint.",
        "You must continue optimizing the Triton Ascend NPU kernel path itself.",
        "Do not replace the core computation with a pure PyTorch implementation just to improve final outputs or benchmark numbers.",
        "A round that bypasses the Triton kernel path with pure PyTorch code does not count as a successful optimize round.",
        "When comparing round performance, prefer the kernel-oriented `compare-perf` view.",
        "If the comparison falls back to total-op or mixes kernel and total-op across cases, record the resolved `effective_metric_source` and surface that mismatch as a warning.",
    ]


def _shared_optimize_prompt_lines(
    *,
    target_chip: str,
    optimize_check_line: str,
    optimize_target: str,
    enable_subagent: bool = False,
) -> list[str]:
    return [
        *optimize_target_lines(optimize_target=optimize_target),
        "Use the staged `triton-npu-prepare-optimize-baseline` skill when baseline artifacts are missing or invalid.",
        optimize_check_line,
        *(
            optimize_subagent_recommendation_lines()
            if enable_subagent
            else []
        ),
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "If baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill and continue only after it has repaired the baseline through `triton-npu-optimize-submit-baseline`.",
        "For each round, write the optimized operator snapshot as `opt_<original-operator>.py` inside `opt-round-N/`.",
        "For each round, keep the benchmark artifact as `opt_<original-operator>_perf.txt` inside `opt-round-N/`, ensure that file is generated by the `triton-npu-run-eval` skill's `run-bench` flow, and record that exact filename in `round-state.json`.",
        "Use `baseline/<operator>_perf.txt` for canonical performance comparisons.",
        "Use `compare-perf` as the only authority for claimed speedups or benchmark deltas.",
        "Record exactly one resolved comparison basis in `round-state.json` as `effective_metric_source` using one of: `kernel`, `total-op`, or `mixed`.",
        "Do not read staged skill implementation files under skills/*/scripts/ unless debugging, patching, or verifying that helper behavior.",
        "Prefer SKILL.md and references/*.md for workflow guidance.",
        "Use the staged `triton-npu-analyze-round-performance` skill when a round needs deeper diagnosis from profile or IR evidence.",
        "When you use that analysis flow, write `opt-round-N/perf-analysis.md` as the standalone analysis artifact.",
        "Use `triton-npu-analyze-ir` as the IR evidence companion when IR attribution is needed, while `triton-npu-analyze-round-performance` remains the owner of `opt-round-N/perf-analysis.md`.",
        "Reuse existing correctness tests and benchmark cases when they already exist; generate them only when required artifacts are missing.",
        "State the optimization hypothesis and why it may help before editing code for each round.",
        "Explain what evidence supports the change, using benchmark behavior, profiling, IR inspection, code structure, or a combination of them.",
        "If you skip profiling or IR capture for a round, explain why the existing evidence is already sufficient.",
        *layered_analysis_lines(round_scope="the round"),
        *strict_learned_lessons_lines(),
        f"Target chip for this optimize session: {target_chip}.",
        f"When ranking optimization points, prefer changes that fit {target_chip} unless the round artifacts prove a different chip target.",
    ]


def _finalize_optimize_prompt_lines(
    *,
    lines: list[str],
    resume_existing_session: bool,
    compiler_source_path: Path | None,
    compiler_source_commit: str | None,
    enable_cann_ext_api: bool,
) -> str:
    lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )
    )
    lines.extend(cann_ext_api_lines(enabled=enable_cann_ext_api))
    lines.extend(baseline_state_contract_lines())
    if resume_existing_session:
        lines.extend(
            [
                "Continue the existing optimization session instead of restarting from scratch.",
                "Read `opt-note.md`, existing `opt-round-*` directories, and existing round logs before making changes.",
            ]
        )
    return "\n".join(lines)


def _extract_additional_user_instructions(base_prompt: str | None) -> list[str]:
    if not base_prompt:
        return []
    extracted: list[str] = []
    capture_additional_user_instructions = False
    for line in base_prompt.strip().splitlines():
        if capture_additional_user_instructions:
            extracted.append(line)
            continue
        if line == "Additional user instructions:":
            extracted.append(line)
            capture_additional_user_instructions = True
    if not extracted:
        stripped = base_prompt.strip()
        return ["Additional user instructions:", stripped] if stripped else []
    while extracted and extracted[-1] == "":
        extracted.pop()
    return extracted


def build_optimize_supervisor_prompt(
    workdir: Path,
    *,
    latest_round_dir: Path | None = None,
    optimize_target: str = "kernel",
    cli_followup_summary: str | None = None,
    workflow_phase_summary: str | None = None,
) -> str:
    lines = [
        "This invocation is the optimize supervisor pass.",
        "This invocation is an audit and handoff pass, not a new optimization round.",
        f"Target optimization scope for this optimize session: {optimize_target}.",
        f"Read `{_display_path(workdir / 'opt-note.md')}` before acting.",
    ]
    if workflow_phase_summary is not None:
        lines.extend(["Workflow phase summary:", workflow_phase_summary])
    if latest_round_dir is not None:
        lines.append(f"Read `{_display_path(latest_round_dir)}` before acting.")
        if cli_followup_summary is not None:
            lines.extend(
                [
                    "Read this CLI round follow-up summary before auditing the round:",
                    cli_followup_summary,
                ]
            )
        lines.extend(
            [
                "Apply only metadata repairs derived from existing facts.",
                "Use only existing `compare-perf` results when auditing or restating performance conclusions.",
                "Read the staged `triton-npu-optimize`, `triton-npu-prepare-optimize-baseline`, `triton-npu-optimize-submit-baseline`, `triton-npu-optimize-submit-round`, and `triton-npu-optimize-start-round` skills as the workflow contract that the worker round was supposed to follow.",
                "Audit the worker against this analysis ladder: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
                "Require the recorded analysis level, escalation reason, and cited evidence path to agree with the round artifacts.",
                "Read the latest `opt-round-N/attempts.md`, `summary.md`, and `round-state.json` before deciding anything.",
                "Read existing benchmark, profiler, and IR artifacts only when they already exist and are needed to verify the worker's recorded claims.",
                (
                    "Accept valid whole-operator restructuring when the optimize target is operator."
                    if optimize_target == "operator"
                    else "Keep kernel-path-focused intent when the optimize target is kernel."
                ),
                (
                    "Audit operator-target rounds against the total-op conclusion while still reading the kernel view as supporting diagnosis."
                    if optimize_target == "operator"
                    else "Allow fallback-driven rounds to pass when their recorded `effective_metric_source` is `total-op` or `mixed`, but require that mismatch to be called out as a warning."
                ),
                "Reject rounds that preserve only the public API shape but replace the Triton kernel path with pure PyTorch computation.",
                "Write `supervisor-report.md` with a `Status:` line and a `Blocking issues:` line.",
                "Use only these supervisor statuses: `pass` or `fail`.",
                "The CLI will read that supervisor report and pass the relevant continuation context into any later worker invocation.",
                "The CLI decides whether another round is required from the round-loop policy; do not encode stop-versus-continue in the supervisor report.",
                "Do not edit the operator implementation.",
                "Do not perform open-ended optimization work.",
                "Do not fabricate missing correctness, benchmark, profiler, or IR evidence.",
                "Do not launch new profiler or IR collection from the supervisor pass.",
                "Do not silently promote an invalid round to current best.",
            ]
        )
    return "\n".join(lines)


def build_optimize_resume_prompt(
    summary: str,
    *,
    base_prompt: str | None = None,
    round_mode: Literal["checked", "supervised"] = "checked",
    optimize_target: str = "kernel",
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_subagent: bool = False,
) -> str:
    lines: list[str] = []
    if base_prompt:
        lines.extend([base_prompt.strip(), ""])
    if not base_prompt:
        del round_mode
        lines.extend(
            [
                "This invocation continues the optimize task.",
                "This invocation owns exactly one round.",
                "Treat this as a long-running task.",
                "Keep making progress until the current round is complete.",
                "Do not self-approve whether the optimize session should continue.",
                "",
            ]
        )
    continuation_lines = [
        f"Target optimization scope for this optimize session: {optimize_target}.",
        (
            "Optimize end-to-end operator latency."
            if optimize_target == "operator"
            else "Continue optimizing the Triton Ascend NPU kernel path itself."
        ),
        "Continue the existing optimize task instead of restarting from scratch.",
        "Read `opt-note.md`, existing `opt-round-*` directories, and any round summaries or attempt logs before making the next change.",
        "Reuse the established `baseline/` directory instead of redefining the canonical baseline.",
        "Keep the optimize workflow hypothesis-driven: explain why each next change may help and what evidence supports it.",
        "Use `compare-perf` output as the only source for performance deltas and speedup metrics.",
        *next_round_reflection_lines(),
        *sequential_round_execution_lines(),
        *(
            [
                "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
            ]
            if optimize_target == "operator"
            else []
        ),
        *(
            optimize_subagent_recommendation_lines()
            if enable_subagent
            else []
        ),
        *layered_analysis_lines(round_scope="the round"),
        *strict_learned_lessons_lines(),
    ]
    lines.extend(continuation_lines)
    lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )
    )
    lines.extend(["", f"Progress summary:\n{summary}"])
    return "\n".join(lines)


def build_optimize_round_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
    round_mode: Literal["checked", "supervised"],
    baseline_ready: bool = True,
    current_round: int = 1,
    final_round: int = 1,
    round_batch_size: int = 5,
    workflow_phase_summary: str | None = None,
) -> str:
    del input_path, output_path, test_mode, bench_mode, round_batch_size, round_mode
    lines = [
        f"This invocation owns rounds {current_round} through {final_round}.",
        "Execute those rounds strictly one at a time.",
        "Do not pre-plan the full batch before acting.",
        "Produce all required round artifacts before stopping.",
        "The CLI will validate the completed batch after the invocation exits.",
    ]
    if baseline_ready:
        lines.append("The baseline has already been validated before this worker batch.")
        lines.append(
            "If a round needs repairs or continuation, a later invocation will return with direct CLI guidance in the prompt."
        )
    else:
        lines.append(
            "In this interactive session, repair or establish `baseline/` before `opt-round-1` if it is missing or invalid."
        )
        lines.append(
            "Do not rely on a separate baseline-preflight invocation or a later worker batch to do that setup for you."
        )
    if workflow_phase_summary is not None:
        lines.extend(["", "Workflow phase summary:", workflow_phase_summary])
    lines.extend(
        _shared_optimize_prompt_lines(
            target_chip=target_chip,
            optimize_check_line="You must run the staged `triton-npu-optimize-submit-round` skill after each completed round.",
            optimize_target=optimize_target,
            enable_subagent=enable_subagent,
        )
    )
    lines.append("Do not self-approve whether the optimize session should continue.")
    lines.append("Before each round, re-evaluate the next bottleneck and choose the right analysis depth from the current evidence.")
    return _finalize_optimize_prompt_lines(
        lines=lines,
        resume_existing_session=resume_existing_session,
        compiler_source_path=compiler_source_path,
        compiler_source_commit=compiler_source_commit,
        enable_cann_ext_api=enable_cann_ext_api,
    )


def build_optimize_baseline_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    baseline_state: str,
    base_prompt: str | None = None,
    remote: str | None = None,
    remote_workdir: str | None = None,
    workflow_phase_summary: str | None = None,
) -> str:
    lines = [
        "This invocation repairs the optimize baseline before the round loop begins.",
        f"Baseline preflight result: {baseline_state}.",
    ]
    context_lines = [
        f"Operator input: {_display_path(input_path)}",
    ]
    if test_mode is not None:
        context_lines.append(f"Requested test mode: {test_mode}")
    if bench_mode is not None:
        context_lines.append(f"Requested bench mode: {bench_mode}")
    if remote is not None:
        context_lines.append(f"Remote execution target: {remote}")
        if remote_workdir is not None:
            context_lines.append(f"Remote execution root: {remote_workdir}")
    context_lines.extend(
        [
            f"Target optimization scope for this optimize session: {optimize_target}.",
            f"Target chip for this optimize session: {target_chip}.",
        ]
    )
    context_lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )
    )
    context_lines.extend(cann_ext_api_lines(enabled=enable_cann_ext_api))
    additional_user_instructions = _extract_additional_user_instructions(base_prompt)
    if additional_user_instructions:
        context_lines.extend(["", *additional_user_instructions])
    if context_lines:
        lines.extend(["", *context_lines])
    if workflow_phase_summary is not None:
        lines.extend(["", "Workflow phase summary:", workflow_phase_summary])
    lines.extend(
        [
            "",
            "Repair or establish `baseline/` before the round loop begins.",
            "Use the staged `triton-npu-optimize-submit-baseline` skill to submit the baseline until it passes.",
            "Do not open a new optimization round yet.",
        ]
    )
    return "\n".join(lines)
