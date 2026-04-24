from __future__ import annotations

from pathlib import Path


def resolve_convert_output_path(
    input_path: Path,
    *,
    explicit_output: str | None,
) -> Path:
    if explicit_output:
        return Path(explicit_output).expanduser().resolve()
    return input_path.with_name(f"triton_{input_path.stem}.py")


def prepare_convert_target(
    output_path: Path,
    *,
    force_overwrite: bool,
) -> list[str]:
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
