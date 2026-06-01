from __future__ import annotations

from pathlib import Path
from typing import Literal

from triton_agent.models import COMMAND_TO_SKILL, CommandKind
from triton_agent.optimize.prompts import (
    build_optimize_baseline_prompt,
    build_optimize_continuous_prompt,
    build_optimize_resume_prompt,
    build_optimize_round_prompt,
    build_optimize_supervisor_prompt,
)
from triton_agent.paths import default_generated_output_path

__all__ = [
    "PROMPT_INTROS",
    "append_additional_user_instructions",
    "build_optimize_baseline_prompt",
    "build_optimize_continuous_prompt",
    "build_optimize_resume_prompt",
    "build_optimize_round_prompt",
    "build_optimize_supervisor_prompt",
    "build_prompt",
]


PROMPT_INTROS = {
    CommandKind.GEN_EVAL: "Repair the operator when needed, then generate correctness tests and a benchmark.",
    CommandKind.CONVERT: "Convert the PyTorch operator into a Triton NPU-backed PyTorch operator and validate it with differential correctness testing.",
    CommandKind.GEN_TEST: "Generate correctness tests for the operator file.",
    CommandKind.RUN_TEST: "Run the generated correctness tests for the operator file.",
    CommandKind.GEN_BENCH: "Generate a benchmark for the operator file.",
    CommandKind.RUN_BENCH: "Run the generated benchmark for the operator file.",
    CommandKind.OPTIMIZE: "Optimize the operator implementation.",
    CommandKind.REPORT: "Read the optimize workspace and generate a Chinese optimization report.",
}


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    return path.as_posix()


def append_additional_user_instructions(prompt: str, user_prompt: str | None) -> str:
    if user_prompt is None:
        return prompt
    stripped_prompt = user_prompt.strip()
    if not stripped_prompt:
        return prompt
    return f"{prompt}\n\nAdditional user instructions:\n{stripped_prompt}"


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
    min_rounds: int | None = 5,
    continue_optimize: bool = False,
    resume_existing_session: bool | None = None,
    round_mode: Literal["continuous", "checked", "supervised"] = "continuous",
    target_chip: str | None = None,
    optimize_target: str = "kernel",
    compiler_source_path: Path | None = None,
    compiler_source_commit: str | None = None,
    enable_cann_ext_api: bool = False,
    enable_subagent: bool = False,
) -> str:
    should_resume_existing_session = (
        continue_optimize if resume_existing_session is None else resume_existing_session
    )
    resolved_min_rounds = 5 if min_rounds is None else min_rounds
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
        lines.append(f"Operator file: {_display_path(operator_path)}")
        lines.append(f"Test file: {_display_path(input_path)}")
    elif command_kind == CommandKind.RUN_BENCH:
        lines.append(f"Operator file: {_display_path(operator_path)}")
        lines.append(f"Benchmark file: {_display_path(input_path)}")
    else:
        lines.append(f"Operator input: {_display_path(input_path)}")
    if command_kind == CommandKind.GEN_EVAL:
        test_output = default_generated_output_path(CommandKind.GEN_TEST, input_path, test_mode=test_mode)
        bench_output = default_generated_output_path(CommandKind.GEN_BENCH, input_path)
        lines.extend(
            [
                f"Requested test output: {_display_path(test_output)}",
                f"Requested benchmark output: {_display_path(bench_output)}",
            ]
        )
    if output_path is not None and command_kind != CommandKind.GEN_EVAL:
        lines.append(f"Requested output: {_display_path(output_path)}")
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
        if command_kind in {CommandKind.GEN_EVAL, CommandKind.GEN_EVAL_BATCH, CommandKind.OPTIMIZE}:
            lines.append(
                "When you execute generated test cases or benchmark cases in this task, include the "
                "same `--remote` setting and reuse `--remote-workdir` when provided."
            )
        elif command_kind in {CommandKind.GEN_TEST, CommandKind.CONVERT}:
            lines.append(
                "When you execute generated test cases in this task, include the same `--remote` "
                "setting and reuse `--remote-workdir` when provided."
            )
        elif command_kind == CommandKind.GEN_BENCH:
            lines.append(
                "When you execute generated benchmark cases in this task, include the same "
                "`--remote` setting and reuse `--remote-workdir` when provided."
            )
    if command_kind == CommandKind.GEN_TEST:
        lines.append(
            "When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function "
            "and that wrapper launches the Triton kernel, prefer that module class as the "
            "public entrypoint rather than selecting the intermediate wrapper function."
        )
        lines.append(
            "If the generated harness uses randomized inputs, explicitly fix the seed during "
            "case construction so repeated runs of the same harness produce identical inputs."
        )
        lines.append(
            "After generating the artifact, execute the generated test case. "
            "If execution fails, repair the generated artifact and retry automatically."
        )
    if command_kind == CommandKind.GEN_BENCH:
        lines.append(
            "When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function "
            "and that wrapper launches the Triton kernel, prefer that module class as the "
            "public entrypoint rather than selecting the intermediate wrapper function."
        )
        lines.append(
            "If the generated harness uses randomized inputs, explicitly fix the seed during "
            "case construction so repeated runs of the same harness produce identical inputs."
        )
        lines.append(
            "After generating the artifact, execute the generated benchmark case. "
            "If execution fails, repair the generated artifact and retry automatically."
        )
    if command_kind == CommandKind.GEN_EVAL:
        lines.extend(
            [
                "When a `class Model` (or equivalent `torch.nn.Module`) calls a wrapper function "
                "and that wrapper launches the Triton kernel, prefer that module class as the "
                "public entrypoint rather than selecting the intermediate wrapper function.",
                "If either generated harness uses randomized inputs, explicitly fix the seed during "
                "case construction so repeated runs of the same harness produce identical inputs.",
                "You may edit the original operator file directly when the operator implementation is at fault.",
                "Generate both the test harness and the benchmark harness in this task.",
                "After generating them, both generated artifacts must be executed before the task finishes.",
                "If validation fails, repair the generated harness when the harness is at fault, or repair the original operator file when the operator is at fault, then retry.",
            ]
        )
    if command_kind == CommandKind.CONVERT:
        lines.extend(
            [
                "Treat the input operator file as source material only.",
                "Do not execute the original input operator file.",
                "Treat the input operator file as source material and the differential correctness oracle.",
                "Write the converted operator to the requested output path and keep the original input file unchanged.",
                "Preserve the trailing input-helper block from the input file in the converted output so later harnesses can reuse it.",
                "When generating or validating harnesses, you may add broader coverage and do not need to limit yourself to only the preserved trailing helpers.",
                "Do not introduce unnecessary wrappers, compatibility branches, helper layers, or scaffolding.",
                "Keep the converted artifact as a PyTorch-facing operator backed by a real Triton Ascend NPU kernel path.",
                "A PyTorch-facing wrapper or module API may remain when that is the intended public interface.",
                "A pure PyTorch rewrite does not satisfy this convert task, even if differential validation passes.",
                "Target Ascend NPU only for this conversion flow and do not add CUDA, CPU, MPS, or generic multi-backend fallback logic unless the source file already requires shared import structure around the public API.",
                "Do not inline correctness or benchmark harness code into the converted operator file.",
                "Do not benchmark this workflow.",
                "Do not create `baseline/`.",
                "Generate a differential test for the converted output and execute it.",
                "Validate the converted output by comparing it against the original operator behavior.",
            ]
        )

    if command_kind == CommandKind.OPTIMIZE:
        if round_mode == "continuous":
            lines.extend(
                build_optimize_continuous_prompt(
                    input_path,
                    output_path,
                    test_mode=test_mode,
                    bench_mode=bench_mode,
                    target_chip=target_chip or "A5",
                    optimize_target=optimize_target,
                    min_rounds=resolved_min_rounds,
                    resume_existing_session=should_resume_existing_session,
                    compiler_source_path=compiler_source_path,
                    compiler_source_commit=compiler_source_commit,
                    enable_cann_ext_api=enable_cann_ext_api,
                    enable_subagent=enable_subagent,
                ).splitlines()
            )
        else:
            lines.extend(
                build_optimize_round_prompt(
                    input_path,
                    output_path,
                    test_mode=test_mode,
                    bench_mode=bench_mode,
                    target_chip=target_chip or "A5",
                    optimize_target=optimize_target,
                    resume_existing_session=should_resume_existing_session,
                    compiler_source_path=compiler_source_path,
                    compiler_source_commit=compiler_source_commit,
                    enable_cann_ext_api=enable_cann_ext_api,
                    enable_subagent=enable_subagent,
                    round_mode=round_mode,
                    baseline_ready=True,
                ).splitlines()
            )
    else:
        lines.append("Complete the requested task and summarize assumptions briefly.")
    return "\n".join(lines)
