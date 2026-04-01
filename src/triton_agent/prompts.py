from __future__ import annotations

from pathlib import Path

from triton_agent.models import COMMAND_TO_SKILL, CommandKind


PROMPT_INTROS = {
    CommandKind.GEN_TEST: "Generate correctness tests for the operator file.",
    CommandKind.RUN_TEST: "Run the generated correctness tests for the operator file.",
    CommandKind.GEN_BENCH: "Generate a benchmark for the operator file.",
    CommandKind.RUN_BENCH: "Run the generated benchmark for the operator file.",
    CommandKind.OPTIMIZE: "Optimize the operator implementation.",
}


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
) -> str:
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
    if output_path is not None:
        lines.append(f"Requested output: {output_path}")
    if test_mode is not None:
        lines.append(f"Requested test mode: {test_mode}")
    if bench_mode is not None:
        lines.append(f"Requested bench mode: {bench_mode}")
    if force_overwrite:
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

    if command_kind == CommandKind.OPTIMIZE:
        lines.extend(
            [
                "Treat this as a long-running task.",
                "Keep making progress until the optimized operator is complete.",
            ]
        )
    else:
        lines.append("Complete the requested task and summarize assumptions briefly.")
    return "\n".join(lines)
