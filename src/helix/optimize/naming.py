from __future__ import annotations

from pathlib import Path

from helix.batch.discovery import (
    NO_CANDIDATE_OPERATOR_FILE,
    is_batch_operator_candidate,
    resolve_batch_operator_file,
)

_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}

def is_batch_optimize_operator_candidate(path: Path) -> bool:
    return is_batch_operator_candidate(
        path,
        excluded_names=_BATCH_OPTIMIZE_EXCLUDED_NAMES,
        excluded_prefixes=_BATCH_OPTIMIZE_EXCLUDED_PREFIXES,
    )


def resolve_batch_optimize_operator_file(
    workspace: Path,
    *,
    operator_filter: str | None = None,
) -> Path:
    return resolve_batch_operator_file(
        workspace,
        is_operator_candidate=is_batch_optimize_operator_candidate,
        no_candidate_message=NO_CANDIDATE_OPERATOR_FILE,
        operator_filter=operator_filter,
    )
