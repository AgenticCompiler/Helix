from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.contract import baseline_state_contract_lines


def _display_path(path: Path) -> str:
    return path.as_posix()


def strict_learned_lessons_lines() -> list[str]:
    return [
        "`learned_lessons.md` is only for reusable, evidence-backed optimization or profiling rules that can transfer to related Triton Ascend NPU operators.",
        "Do not put round narrative, command failures, or operator-specific details in `learned_lessons.md`; keep those in `attempts.md`, `summary.md`, or `opt-note.md`.",
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


def torch_npu_operator_knowledge_lines(*, optimize_target: str) -> list[str]:
    if optimize_target != "operator":
        return []
    return [
        "Use the staged `torch-npu-optimize-knowledge` skill for Torch NPU and operator-level pattern references.",
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
        *torch_npu_operator_knowledge_lines(optimize_target=optimize_target),
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
) -> list[str]:
    return [
        *optimize_target_lines(optimize_target=optimize_target),
        "Use the staged `triton-npu-prepare-optimize-baseline` skill when baseline artifacts are missing or invalid.",
        optimize_check_line,
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "If baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill and continue only after it has repaired the baseline through `check-baseline`.",
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


def build_optimize_worker_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
) -> str:
    del input_path, output_path, test_mode, bench_mode, min_rounds
    lines = [
        "This invocation is the optimize worker role.",
        "This invocation owns exactly one round.",
        "Read `.triton-agent/round-brief.md` before acting.",
        "Treat this as a long-running task.",
        "Keep making progress until the current round is complete.",
        *_shared_optimize_prompt_lines(
            target_chip=target_chip,
            optimize_check_line="Use the staged `triton-npu-optimize-check` skill to validate the completed round.",
            optimize_target=optimize_target,
        ),
        "Produce all required round artifacts before stopping.",
        "After finishing the round, use the staged `triton-npu-optimize-check` skill to run `check-round` and repair the round until it passes.",
        "The current round must pass `check-round` through `triton-npu-optimize-check` before the invocation ends.",
        "Do not self-approve whether the optimize session should continue.",
    ]
    return _finalize_optimize_prompt_lines(
        lines=lines,
        resume_existing_session=resume_existing_session,
        compiler_source_path=compiler_source_path,
        compiler_source_commit=compiler_source_commit,
        enable_cann_ext_api=enable_cann_ext_api,
    )


def build_optimize_unsupervised_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    optimize_target: str = "kernel",
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
) -> str:
    del input_path, output_path, test_mode, bench_mode
    lines = [
        "This invocation is an unsupervised optimize run.",
        "Own the end-to-end optimize session and continue optimizing until the session should stop.",
    ]
    if min_rounds is not None:
        lines.extend(
            [
                f"Complete at least {min_rounds} optimization rounds before deciding the session should stop.",
                f"Once {min_rounds} optimization rounds are complete, stop the session after the current round passes `check-round` through `triton-npu-optimize-check` unless there is a concrete reason to continue.",
            ]
        )
    lines.append("Treat this as a long-running task.")
    lines.extend(
        _shared_optimize_prompt_lines(
            target_chip=target_chip,
            optimize_check_line="Use the staged `triton-npu-optimize-check` skill to validate every completed round.",
            optimize_target=optimize_target,
        )
    )
    if min_rounds is not None:
        lines.append(
            f"After finishing each round, use the staged `triton-npu-optimize-check` skill to run "
            f"`check-round --round-dir opt-round-N --min-rounds {min_rounds}` and repair the round "
            f"until it passes. Read the summary for the exit signal."
        )
    else:
        lines.append(
            "After finishing each round, use the staged `triton-npu-optimize-check` skill to run "
            "`check-round` and repair the round until it passes."
        )
    lines.extend(
        [
            "Do not begin the next round until the current round passes `check-round` through `triton-npu-optimize-check`.",
            "Record round outcomes and keep optimize artifacts up to date before stopping.",
        ]
    )
    return _finalize_optimize_prompt_lines(
        lines=lines,
        resume_existing_session=resume_existing_session,
        compiler_source_path=compiler_source_path,
        compiler_source_commit=compiler_source_commit,
        enable_cann_ext_api=enable_cann_ext_api,
    )


def build_optimize_supervisor_prompt(
    workdir: Path,
    *,
    latest_round_dir: Path | None = None,
    optimize_target: str = "kernel",
) -> str:
    lines = [
        "This invocation is the optimize supervisor role.",
        "This invocation is an audit and handoff pass, not a new optimization round.",
        f"Target optimization scope for this optimize session: {optimize_target}.",
        f"Read `{_display_path(workdir / 'opt-note.md')}` before acting.",
    ]
    if latest_round_dir is not None:
        lines.append(f"Read `{_display_path(latest_round_dir)}` before acting.")
        lines.extend(
            [
                "Apply only metadata repairs derived from existing facts.",
                "Use only existing `compare-perf` results when auditing or restating performance conclusions.",
                "Read the staged `triton-npu-optimize`, `triton-npu-prepare-optimize-baseline`, and `triton-npu-optimize-check` skills as the workflow contract that the worker round was supposed to follow.",
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
                "Write `.triton-agent/supervisor-report.md` with a `Decision:` line and a `Blocking issues:` line.",
                "Write `.triton-agent/round-brief.md` with the next-worker handoff; when continuation is not allowed, record the stop or repair reason there.",
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
    supervise: str = "off",
    optimize_target: str = "kernel",
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
) -> str:
    lines: list[str] = []
    if base_prompt:
        lines.extend([base_prompt.strip(), ""])
    if not base_prompt:
        if supervise == "on":
            lines.extend(
                [
                    "This invocation is the optimize worker role.",
                    "This invocation owns exactly one round.",
                    "Read `.triton-agent/round-brief.md` before acting.",
                    "Treat this as a long-running task.",
                    "Keep making progress until the current round is complete.",
                    "Do not self-approve whether the optimize session should continue.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "This invocation continues an unsupervised optimize task.",
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
        *torch_npu_operator_knowledge_lines(optimize_target=optimize_target),
        *layered_analysis_lines(round_scope="the round"),
        *strict_learned_lessons_lines(),
    ]
    if supervise == "off" and base_prompt:
        continuation_lines.insert(0, "This invocation continues an unsupervised optimize task.")
    lines.extend(continuation_lines)
    lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )
    )
    lines.extend(["", f"Progress summary:\n{summary}"])
    return "\n".join(lines)
