from __future__ import annotations

from pathlib import Path
from typing import Literal

from triton_agent.models import COMMAND_TO_SKILL, CommandKind
from triton_agent.optimize.contract import baseline_state_contract_lines
from triton_agent.paths import default_generated_output_path


PROMPT_INTROS = {
    CommandKind.GEN_EVAL: "Repair the operator when needed, then generate correctness tests and a benchmark.",
    CommandKind.GEN_TEST: "Generate correctness tests for the operator file.",
    CommandKind.RUN_TEST: "Run the generated correctness tests for the operator file.",
    CommandKind.GEN_BENCH: "Generate a benchmark for the operator file.",
    CommandKind.RUN_BENCH: "Run the generated benchmark for the operator file.",
    CommandKind.OPTIMIZE: "Optimize the operator implementation.",
}


def strict_learned_lessons_lines() -> list[str]:
    return [
        "`learned_lessons.md` is only for reusable, evidence-backed optimization or profiling rules that can transfer to related Triton Ascend NPU operators.",
        "Do not put round narrative, command failures, or operator-specific details in `learned_lessons.md`; keep those in `attempts.md`, `summary.md`, or `opt-note.md`.",
    ]


def layered_analysis_lines(*, round_scope: str) -> list[str]:
    return [
        f"Choose the analysis level for {round_scope} before editing code.",
        "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
        "Use pattern triage only to decide whether a strong pattern-backed hypothesis already exists.",
        "Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.",
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
        f"Compiler source path: {compiler_source_path}",
        f"Compiler source commit: {compiler_source_commit}.",
        "Treat the compiler source checkout as read-only.",
        "Do not run git clone, git fetch, git pull, or modify files in the compiler source checkout.",
        "Use the staged `triton-npu-analyze-compiler-source` skill only when compiler source evidence is needed.",
        "Prefer the evidence ladder first: benchmark and correctness results, then profiler evidence, then IR evidence, then compiler source.",
    ]


def append_additional_user_instructions(prompt: str, user_prompt: str | None) -> str:
    if user_prompt is None:
        return prompt
    stripped_prompt = user_prompt.strip()
    if not stripped_prompt:
        return prompt
    return f"{prompt}\n\nAdditional user instructions:\n{stripped_prompt}"


def build_optimize_worker_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
) -> str:
    del input_path, output_path, test_mode, bench_mode, min_rounds
    lines = [
        "This invocation is the optimize worker role.",
        "This invocation owns exactly one round.",
        "Read `.triton-agent/round-brief.md` before acting.",
        "Treat this as a long-running task.",
        "Keep making progress until the current round is complete.",
        "PyTorch-facing public API may remain as a wrapper when that is the intended operator entrypoint.",
        "You must continue optimizing the Triton Ascend NPU kernel path itself.",
        "Do not replace the core computation with a pure PyTorch implementation just to improve final outputs or benchmark numbers.",
        "A round that bypasses the Triton kernel path with pure PyTorch code does not count as a successful optimize round.",
        "Use the staged `triton-npu-prepare-optimize-baseline` skill when baseline artifacts are missing or invalid.",
        "Use the staged `triton-npu-optimize-check` skill to validate the completed round.",
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "If baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill and continue only after it has repaired the baseline through `check-baseline`.",
        "Use `baseline/perf.txt` for canonical performance comparisons.",
        "Use `compare-perf` as the only authority for claimed speedups or benchmark deltas.",
        "Use the staged `triton-npu-analyze-round-performance` skill when a round needs deeper diagnosis from profile or IR evidence.",
        "When you use that analysis flow, write `opt-round-N/perf-analysis.md` as the standalone analysis artifact.",
        "Reuse existing correctness tests and benchmark cases when they already exist; generate them only when required artifacts are missing.",
        "State the optimization hypothesis and why it may help before editing code for each round.",
        "Explain what evidence supports the change, using benchmark behavior, profiling, IR inspection, code structure, or a combination of them.",
        "If you skip profiling or IR capture for a round, explain why the existing evidence is already sufficient.",
        *layered_analysis_lines(round_scope="the round"),
        *strict_learned_lessons_lines(),
        f"Target chip for this optimize session: {target_chip}.",
        f"When ranking optimization points, prefer changes that fit {target_chip} unless the round artifacts prove a different chip target.",
        "Produce all required round artifacts before stopping.",
        "After finishing the round, use the staged `triton-npu-optimize-check` skill to run `check-round` and repair the round until it passes.",
        "The current round must pass `check-round` through `triton-npu-optimize-check` before the invocation ends.",
        "Do not self-approve whether the optimize session should continue.",
    ]
    lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )
    )
    lines.extend(baseline_state_contract_lines())
    if resume_existing_session:
        lines.extend(
            [
                "Continue the existing optimization session instead of restarting from scratch.",
                "Read `opt-note.md`, existing `opt-round-*` directories, and existing round logs before making changes.",
            ]
        )
    return "\n".join(lines)


def build_optimize_unsupervised_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    target_chip: str = "A5",
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
) -> str:
    del input_path, output_path, test_mode, bench_mode
    lines = [
        "This invocation is an unsupervised optimize run.",
        "Own the end-to-end optimize session and continue optimizing until the session should stop.",
        "Treat this as a long-running task.",
        "PyTorch-facing public API may remain as a wrapper when that is the intended operator entrypoint.",
        "You must continue optimizing the Triton Ascend NPU kernel path itself.",
        "Do not replace the core computation with a pure PyTorch implementation just to improve final outputs or benchmark numbers.",
        "A round that bypasses the Triton kernel path with pure PyTorch code does not count as a successful optimize round.",
        "Use the staged `triton-npu-prepare-optimize-baseline` skill when baseline artifacts are missing or invalid.",
        "Use the staged `triton-npu-optimize-check` skill to validate every completed round.",
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "If baseline preparation is needed, use the staged `triton-npu-prepare-optimize-baseline` skill and continue only after it has repaired the baseline through `check-baseline`.",
        "Use `baseline/perf.txt` for canonical performance comparisons.",
        "Use `compare-perf` as the only authority for claimed speedups or benchmark deltas.",
        "Use the staged `triton-npu-analyze-round-performance` skill when a round needs deeper diagnosis from profile or IR evidence.",
        "When you use that analysis flow, write `opt-round-N/perf-analysis.md` as the standalone analysis artifact.",
        "Reuse existing correctness tests and benchmark cases when they already exist; generate them only when required artifacts are missing.",
        "State the optimization hypothesis and why it may help before editing code for each round.",
        "Explain what evidence supports the change, using benchmark behavior, profiling, IR inspection, code structure, or a combination of them.",
        "If you skip profiling or IR capture for a round, explain why the existing evidence is already sufficient.",
        *layered_analysis_lines(round_scope="the round"),
        *strict_learned_lessons_lines(),
        f"Target chip for this optimize session: {target_chip}.",
        f"When ranking optimization points, prefer changes that fit {target_chip} unless the round artifacts prove a different chip target.",
        "After finishing each round, use the staged `triton-npu-optimize-check` skill to run `check-round` and repair the round until it passes.",
        "Do not begin the next round until the current round passes `check-round` through `triton-npu-optimize-check`.",
        "Record round outcomes and keep optimize artifacts up to date before stopping.",
    ]
    lines.extend(
        compiler_source_analysis_lines(
            compiler_source_path=compiler_source_path,
            compiler_source_commit=compiler_source_commit,
        )
    )
    if min_rounds is not None:
        lines.insert(
            2,
            f"Complete at least {min_rounds} optimization rounds before deciding the session should stop.",
        )
        lines.insert(
            3,
            f"Once {min_rounds} optimization rounds are complete, stop the session after the current round passes `check-round` through `triton-npu-optimize-check` unless there is a concrete reason to continue.",
        )
    lines.extend(baseline_state_contract_lines())
    if resume_existing_session:
        lines.extend(
            [
                "Continue the existing optimization session instead of restarting from scratch.",
                "Read `opt-note.md`, existing `opt-round-*` directories, and existing round logs before making changes.",
            ]
        )
    return "\n".join(lines)


def build_optimize_supervisor_prompt(
    workdir: Path,
    *,
    latest_round_dir: Path | None = None,
) -> str:
    lines = [
        "This invocation is the optimize supervisor role.",
        "This invocation is an audit and handoff pass, not a new optimization round.",
        f"Read `{workdir / 'opt-note.md'}` before acting.",
    ]
    if latest_round_dir is not None:
        lines.append(f"Read `{latest_round_dir}` before acting.")
        lines.extend(
            [
                "Apply only metadata repairs derived from existing facts.",
                "Use only existing `compare-perf` results when auditing or restating performance conclusions.",
                "Read the staged `triton-npu-optimize`, `triton-npu-prepare-optimize-baseline`, and `triton-npu-optimize-check` skills as the workflow contract that the worker round was supposed to follow.",
                "Audit the worker against this analysis ladder: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
                "Require the recorded analysis level, escalation reason, and cited evidence path to agree with the round artifacts.",
                "Read the latest `opt-round-N/attempts.md`, `summary.md`, and `round-state.json` before deciding anything.",
                "Read existing benchmark, profiler, and IR artifacts only when they already exist and are needed to verify the worker's recorded claims.",
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


def build_prompt(
    command_kind: CommandKind,
    input_path: Path,
    operator_path: Path | None,
    output_path: Path | None,
    test_mode: str | None,
    bench_mode: str | None,
    force_overwrite: bool,
    remote: str | None = None,
    remote_workdir: str | None = None,
    min_rounds: int | None = None,
    continue_optimize: bool = False,
    resume_existing_session: bool | None = None,
    supervise: Literal["on", "off"] = "off",
    target_chip: str | None = None,
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
) -> str:
    should_resume_existing_session = (
        continue_optimize if resume_existing_session is None else resume_existing_session
    )
    skill_name = COMMAND_TO_SKILL[command_kind]
    lines = [PROMPT_INTROS[command_kind]]
    if skill_name:
        lines.extend(
            [
                f"Use the local skill `{skill_name}` from the workspace skills directory as the primary workflow contract.",
                "Treat any helper scripts or subcommands mentioned later as implementation details inside that skill, not as a replacement for reading the skill.",
            ]
        )
    if command_kind == CommandKind.RUN_TEST:
        lines.append(f"Operator file: {operator_path}")
        lines.append(f"Test file: {input_path}")
    elif command_kind == CommandKind.RUN_BENCH:
        lines.append(f"Operator file: {operator_path}")
        lines.append(f"Benchmark file: {input_path}")
    else:
        lines.append(f"Operator input: {input_path}")
    if command_kind == CommandKind.GEN_EVAL:
        test_output = default_generated_output_path(CommandKind.GEN_TEST, input_path, test_mode=test_mode)
        bench_output = default_generated_output_path(CommandKind.GEN_BENCH, input_path)
        lines.extend(
            [
                f"Requested test output: {test_output}",
                f"Requested benchmark output: {bench_output}",
            ]
        )
    if output_path is not None and command_kind != CommandKind.GEN_EVAL:
        lines.append(f"Requested output: {output_path}")
    if test_mode is not None:
        lines.append(f"Requested test mode: {test_mode}")
    if bench_mode is not None:
        lines.append(f"Requested bench mode: {bench_mode}")
    if force_overwrite:
        if command_kind == CommandKind.GEN_EVAL:
            lines.append(
                "Overwrite any existing generated test, benchmark, or archived execution output files before starting."
            )
        else:
            lines.append("Overwrite the requested output file if it already exists.")
    if remote is not None:
        lines.append(f"Remote execution target: {remote}")
        if remote_workdir is not None:
            lines.append(f"Remote execution root: {remote_workdir}")
        lines.append(
            "When you execute generated test cases or benchmark cases in this task, include the "
            "same `--remote` setting and reuse `--remote-workdir` when provided."
        )
    if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
        lines.append(
            "After generating the artifact, execute the generated test or benchmark case. "
            "If execution fails, repair the generated artifact and retry automatically."
        )
    if command_kind == CommandKind.GEN_EVAL:
        lines.extend(
            [
                "You may edit the original operator file directly when the operator implementation is at fault.",
                "Generate both the test harness and the benchmark harness in this task.",
                "After generating them, both generated artifacts must be executed before the task finishes.",
                "If validation fails, repair the generated harness when the harness is at fault, or repair the original operator file when the operator is at fault, then retry.",
            ]
        )

    if command_kind == CommandKind.OPTIMIZE:
        if supervise == "on":
            lines.extend(
                build_optimize_worker_prompt(
                    input_path,
                    output_path,
                    test_mode=test_mode,
                    bench_mode=bench_mode,
                    target_chip=target_chip or "A5",
                    min_rounds=min_rounds,
                    resume_existing_session=should_resume_existing_session,
                    compiler_source_path=compiler_source_path,
                    compiler_source_commit=compiler_source_commit,
                ).splitlines()
            )
        else:
            lines.extend(
                build_optimize_unsupervised_prompt(
                    input_path,
                    output_path,
                    test_mode=test_mode,
                    bench_mode=bench_mode,
                    target_chip=target_chip or "A5",
                    min_rounds=min_rounds,
                    resume_existing_session=should_resume_existing_session,
                    compiler_source_path=compiler_source_path,
                    compiler_source_commit=compiler_source_commit,
                ).splitlines()
            )
    else:
        lines.append("Complete the requested task and summarize assumptions briefly.")
    return "\n".join(lines)


def build_optimize_resume_prompt(
    summary: str,
    *,
    base_prompt: str | None = None,
    supervise: Literal["on", "off"] = "off",
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
        "Continue the existing optimize task instead of restarting from scratch.",
        "Read `opt-note.md`, existing `opt-round-*` directories, and any round summaries or attempt logs before making the next change.",
        "Reuse the established `baseline/` directory instead of redefining the canonical baseline.",
        "Keep the optimize workflow hypothesis-driven: explain why each next change may help and what evidence supports it.",
        "Use `compare-perf` output as the only source for performance deltas and speedup metrics.",
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
