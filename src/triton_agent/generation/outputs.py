from __future__ import annotations

from pathlib import Path

from triton_agent.models import CommandKind
from triton_agent.paths import default_generated_output_path


def resolve_generation_output_path(
    command_kind: CommandKind,
    input_path: Path,
    *,
    explicit_output: str | None,
    test_mode: str | None = None,
) -> Path | None:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    if command_kind in {
        CommandKind.GEN_TEST,
        CommandKind.GEN_BENCH,
        CommandKind.OPTIMIZE,
    }:
        return default_generated_output_path(command_kind, input_path, test_mode=test_mode)
    return None


def prepare_generation_target(
    command_kind: CommandKind,
    output_path: Path | None,
    force_overwrite: bool,
) -> list[str]:
    if output_path is None:
        return []
    if command_kind not in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
        return []
    if not output_path.exists():
        return []
    if output_path.is_dir():
        raise IsADirectoryError(
            f"Output path is a directory: {output_path}. Choose a file path instead."
        )
    if not force_overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Use --force-overwrite to replace it."
        )
    output_path.unlink()
    return [f"removed existing output file {output_path}"]


def prepare_generation_targets(
    command_kind: CommandKind,
    input_path: Path,
    output_path: Path | None,
    *,
    test_mode: str | None,
    force_overwrite: bool,
) -> list[str]:
    if command_kind == CommandKind.GEN_EVAL:
        protected_paths = [
            default_generated_output_path(CommandKind.GEN_TEST, input_path, test_mode=test_mode),
            default_generated_output_path(CommandKind.GEN_BENCH, input_path),
        ]
        cleanup_only_paths = [
            input_path.with_name(f"{input_path.stem}_result.pt"),
            input_path.with_name(f"{input_path.stem}_perf.txt"),
        ]
        messages: list[str] = []
        for target_path in protected_paths:
            if not target_path.exists():
                continue
            if target_path.is_dir():
                raise IsADirectoryError(
                    f"Output path is a directory: {target_path}. Choose a file path instead."
                )
            if not force_overwrite:
                raise FileExistsError(
                    f"Output file already exists: {target_path}. Use --force-overwrite to replace it."
                )
            target_path.unlink()
            messages.append(f"removed existing output file {target_path}")
        if force_overwrite:
            for target_path in cleanup_only_paths:
                if not target_path.exists():
                    continue
                if target_path.is_dir():
                    raise IsADirectoryError(
                        f"Output path is a directory: {target_path}. Choose a file path instead."
                    )
                target_path.unlink()
                messages.append(f"removed existing output file {target_path}")
        return messages
    return prepare_generation_target(command_kind, output_path, force_overwrite)
