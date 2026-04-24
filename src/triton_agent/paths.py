from __future__ import annotations

from pathlib import Path

from triton_agent.models import CommandKind


def default_generated_output_path(
    command_kind: CommandKind,
    input_path: Path,
    test_mode: str | None = None,
) -> Path:
    stem = input_path.stem
    if command_kind == CommandKind.GEN_TEST:
        if test_mode == "differential":
            return input_path.with_name(f"differential_test_{stem}.py")
        return input_path.with_name(f"test_{stem}.py")
    if command_kind == CommandKind.GEN_BENCH:
        return input_path.with_name(f"bench_{stem}.py")
    if command_kind == CommandKind.OPTIMIZE:
        return input_path.with_name(f"opt_{stem}.py")
    raise ValueError(f"Command {command_kind.value} does not generate a default output file")


def resolve_execution_target(command_kind: CommandKind, input_path: Path) -> Path:
    if command_kind == CommandKind.RUN_TEST:
        target = input_path.with_name(f"test_{input_path.stem}.py")
    elif command_kind == CommandKind.RUN_BENCH:
        target = input_path.with_name(f"bench_{input_path.stem}.py")
    else:
        raise ValueError(f"Command {command_kind.value} does not execute a derived artifact")

    if not target.exists():
        raise FileNotFoundError(f"Expected generated artifact does not exist: {target}")
    return target
