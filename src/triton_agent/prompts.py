from __future__ import annotations

from pathlib import Path
from typing import Literal

from triton_agent.models import COMMAND_TO_SKILL, CommandKind
from triton_agent.optimize_contract import baseline_state_contract_lines
from triton_agent.paths import default_generated_output_path


PROMPT_INTROS = {
    CommandKind.GEN_EVAL: "Repair the operator when needed, then generate correctness tests and a benchmark.",
    CommandKind.GEN_TEST: "Generate correctness tests for the operator file.",
    CommandKind.RUN_TEST: "Run the generated correctness tests for the operator file.",
    CommandKind.GEN_BENCH: "Generate a benchmark for the operator file.",
    CommandKind.RUN_BENCH: "Run the generated benchmark for the operator file.",
    CommandKind.OPTIMIZE: "Optimize the operator implementation.",
}


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
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    require_analysis: bool = False,
) -> str:
    del input_path, output_path, test_mode, bench_mode, min_rounds
    lines = [
        "This invocation is the optimize worker role.",
        "This invocation owns exactly one round.",
        "Read `.triton-agent/round-brief.md` before acting.",
        "Treat this as a long-running task.",
        "Keep making progress until the current round is complete.",
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "Use `baseline/perf.txt` for canonical performance comparisons.",
        "Use `compare-perf` as the only authority for claimed speedups or benchmark deltas.",
        "Reuse existing correctness tests and benchmark cases when they already exist; generate them only when required artifacts are missing.",
        "State the optimization hypothesis and why it may help before editing code for each round.",
        "Explain what evidence supports the change, using benchmark behavior, profiling, IR inspection, code structure, or a combination of them.",
        "If you skip profiling or IR capture for a round, explain why the existing evidence is already sufficient.",
        "Produce all required round artifacts before stopping.",
        "Do not self-approve whether the optimize session should continue.",
    ]
    lines.extend(baseline_state_contract_lines())
    if resume_existing_session:
        lines.extend(
            [
                "Continue the existing optimization session instead of restarting from scratch.",
                "Read `opt-note.md`, existing `opt-round-*` directories, and existing round logs before making changes.",
            ]
        )
    if require_analysis:
        lines.extend(
            [
                "Before the first code-changing round, gather profiling or IR-backed evidence, or record a concrete reason why one analysis path is unavailable and the remaining evidence is sufficient.",
                "Do not begin with blind tiling or launch-parameter search.",
            ]
        )
    return "\n".join(lines)


def build_optimize_unsupervised_prompt(
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    bench_mode: str | None,
    min_rounds: int | None = None,
    resume_existing_session: bool = False,
    require_analysis: bool = False,
) -> str:
    del input_path, output_path, test_mode, bench_mode, min_rounds
    lines = [
        "This invocation is an unsupervised optimize run.",
        "Own the end-to-end optimize session and keep making progress until the run should stop.",
        "Treat this as a long-running task.",
        "Establish or reuse `baseline/` before creating `opt-round-1`.",
        "Use `baseline/perf.txt` for canonical performance comparisons.",
        "Use `compare-perf` as the only authority for claimed speedups or benchmark deltas.",
        "Reuse existing correctness tests and benchmark cases when they already exist; generate them only when required artifacts are missing.",
        "State the optimization hypothesis and why it may help before editing code for each round.",
        "Explain what evidence supports the change, using benchmark behavior, profiling, IR inspection, code structure, or a combination of them.",
        "If you skip profiling or IR capture for a round, explain why the existing evidence is already sufficient.",
        "Record round outcomes and keep optimize artifacts up to date before stopping.",
    ]
    lines.extend(baseline_state_contract_lines())
    if resume_existing_session:
        lines.extend(
            [
                "Continue the existing optimization session instead of restarting from scratch.",
                "Read `opt-note.md`, existing `opt-round-*` directories, and existing round logs before making changes.",
            ]
        )
    if require_analysis:
        lines.extend(
            [
                "Before the first code-changing round, gather profiling or IR-backed evidence, or record a concrete reason why one analysis path is unavailable and the remaining evidence is sufficient.",
                "Do not begin with blind tiling or launch-parameter search.",
            ]
        )
    return "\n".join(lines)


def build_optimize_supervisor_prompt(
    workdir: Path,
    *,
    latest_round_dir: Path | None = None,
    require_analysis: bool = False,
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
            "Emit a structured gate result and next-round brief when continuation is allowed.",
            "Do not perform open-ended optimization work.",
        ]
    )
    if require_analysis:
        lines.append(
            "Require existing profiling or IR-backed evidence, or require the next worker round to record why the remaining evidence is sufficient."
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
    require_analysis: bool = False,
    supervise: Literal["on", "off"] = "off",
) -> str:
    should_resume_existing_session = (
        continue_optimize if resume_existing_session is None else resume_existing_session
    )
    skill_name = COMMAND_TO_SKILL[command_kind]
    lines = [
        PROMPT_INTROS[command_kind],
        f"Use the local skill `{skill_name}` from the workspace skills directory.",
    ]
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
                    min_rounds=min_rounds,
                    resume_existing_session=should_resume_existing_session,
                    require_analysis=require_analysis,
                ).splitlines()
            )
        else:
            lines.extend(
                build_optimize_unsupervised_prompt(
                    input_path,
                    output_path,
                    test_mode=test_mode,
                    bench_mode=bench_mode,
                    min_rounds=min_rounds,
                    resume_existing_session=should_resume_existing_session,
                    require_analysis=require_analysis,
                ).splitlines()
            )
    else:
        lines.append("Complete the requested task and summarize assumptions briefly.")
    return "\n".join(lines)


def build_optimize_resume_prompt(
    summary: str,
    *,
    base_prompt: str | None = None,
    require_analysis: bool = False,
    supervise: Literal["on", "off"] = "off",
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
    ]
    if supervise == "off" and base_prompt:
        continuation_lines.insert(0, "This invocation continues an unsupervised optimize task.")
    lines.extend(continuation_lines)
    if require_analysis:
        lines.append(
            "Before the next code-changing round, gather profiling or IR-backed evidence, or record why the existing evidence is already sufficient."
        )
    lines.extend(["", f"Progress summary:\n{summary}"])
    return "\n".join(lines)
