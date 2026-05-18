from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.skill_contract import optimize_check_module

_OPTIMIZE_CHECK_MODULE = optimize_check_module()

expected_round_operator_name = _OPTIMIZE_CHECK_MODULE.expected_round_operator_name  # type: ignore[reportUnknownVariableType]
expected_round_perf_name = _OPTIMIZE_CHECK_MODULE.expected_round_perf_name  # type: ignore[reportUnknownVariableType]
resolve_round_perf_file = _OPTIMIZE_CHECK_MODULE.resolve_round_perf_file  # type: ignore[reportUnknownVariableType]
resolve_round_operator_file = _OPTIMIZE_CHECK_MODULE.resolve_round_operator_file  # type: ignore[reportUnknownVariableType]


def is_batch_optimize_operator_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix != ".py":
        return False
    if path.name == "__init__.py":
        return False
    return not path.name.startswith(("test_", "differential_test_", "bench_", "opt_"))


def resolve_batch_optimize_operator_file(workspace: Path) -> Path:
    candidates = [
        path for path in sorted(workspace.iterdir()) if is_batch_optimize_operator_candidate(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"no candidate operator file found in workspace: {workspace}")
    raise ValueError(f"multiple candidate operator files found in workspace: {workspace}")
