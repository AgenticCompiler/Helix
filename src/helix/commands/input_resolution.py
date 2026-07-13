from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def resolve_single_operator_input(
    input_path: Path,
    *,
    resolve_operator_file: Callable[[Path], Path],
) -> tuple[Path, Path]:
    if input_path.is_dir():
        operator_path = resolve_operator_file(input_path)
        return operator_path, input_path
    return input_path, input_path.parent
