from __future__ import annotations

from pathlib import Path

from triton_agent.batch_utils import (
    NO_CANDIDATE_OPERATOR_FILE,
    is_batch_operator_candidate,
    resolve_batch_operator_file,
)
from triton_agent.optimize.skill_contract import optimize_check_module

_OPTIMIZE_CHECK_MODULE = optimize_check_module()
_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}

expected_round_operator_name = _OPTIMIZE_CHECK_MODULE.expected_round_operator_name  # type: ignore[reportUnknownVariableType]
expected_round_perf_name = _OPTIMIZE_CHECK_MODULE.expected_round_perf_name  # type: ignore[reportUnknownVariableType]
resolve_round_perf_file = _OPTIMIZE_CHECK_MODULE.resolve_round_perf_file  # type: ignore[reportUnknownVariableType]
resolve_round_operator_file = _OPTIMIZE_CHECK_MODULE.resolve_round_operator_file  # type: ignore[reportUnknownVariableType]


def is_batch_optimize_operator_candidate(path: Path) -> bool:
    return is_batch_operator_candidate(
        path,
        excluded_names=_BATCH_OPTIMIZE_EXCLUDED_NAMES,
        excluded_prefixes=_BATCH_OPTIMIZE_EXCLUDED_PREFIXES,
    )


def resolve_batch_optimize_operator_file(workspace: Path) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_optimize_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
    )
