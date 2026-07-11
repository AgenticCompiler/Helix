from __future__ import annotations

import sys
from pathlib import Path

from helix.models import CommandKind


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            return Path(str(bundle_root)).resolve()
        return Path(sys.executable).resolve().parent
    root = Path(__file__).resolve().parents[2]
    if (root / "skills").is_dir():
        return root
    return root


def skills_root() -> Path:
    return application_root() / "skills"


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
