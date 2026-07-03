from __future__ import annotations

from pathlib import Path


_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}


def resolve_workspace_operator_file(workspace: Path) -> Path:
    candidates = [
        path for path in sorted(workspace.iterdir()) if is_workspace_operator_candidate(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"no candidate operator file found in workspace: {workspace}")
    raise ValueError(f"multiple candidate operator files found in workspace: {workspace}")


def expected_round_operator_name(workspace: Path) -> str:
    operator_file = resolve_workspace_operator_file(workspace)
    return f"opt_{operator_file.name}"


def expected_round_perf_name(workspace: Path) -> str:
    operator_file = resolve_workspace_operator_file(workspace)
    return f"opt_{operator_file.stem}_perf.txt"


def is_workspace_operator_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_OPTIMIZE_EXCLUDED_NAMES:
        return False
    return not path.name.startswith(_BATCH_OPTIMIZE_EXCLUDED_PREFIXES)
