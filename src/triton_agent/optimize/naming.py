from __future__ import annotations

from pathlib import Path

from triton_agent.batch_utils import (
    NO_CANDIDATE_OPERATOR_FILE,
    is_batch_operator_candidate,
    resolve_batch_operator_file,
)

_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}
_ROUND_METADATA_FILENAMES = {
    "attempts.md",
    "summary.md",
    "perf.txt",
    "perf-analysis.md",
    "round-state.json",
}


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


def expected_round_operator_name(workspace: Path) -> str:
    operator_file = resolve_batch_optimize_operator_file(workspace)
    return f"opt_{operator_file.name}"


def expected_round_perf_name(workspace: Path) -> str:
    operator_file = resolve_batch_optimize_operator_file(workspace)
    return f"opt_{operator_file.stem}_perf.txt"


def resolve_round_perf_file(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    try:
        perf_name = expected_round_perf_name(workspace)
    except ValueError:
        perf_name = None
    if perf_name is not None:
        perf_path = round_dir / perf_name
        if perf_path.is_file():
            return perf_path

    perf_txt = round_dir / "perf.txt"
    if perf_txt.is_file():
        return perf_txt

    perf_files = sorted(round_dir.glob("*_perf.txt"))
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def resolve_round_operator_file(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    try:
        operator_name = expected_round_operator_name(workspace)
    except ValueError:
        operator_name = None
    if operator_name is not None:
        operator_path = round_dir / operator_name
        if operator_path.is_file():
            return operator_path

    try:
        legacy_operator_name = resolve_batch_optimize_operator_file(workspace).name
    except ValueError:
        legacy_operator_name = None
    if legacy_operator_name is not None:
        legacy_operator_path = round_dir / legacy_operator_name
        if legacy_operator_path.is_file():
            return legacy_operator_path

    candidates = [
        path
        for path in sorted(round_dir.iterdir())
        if path.is_file()
        and path.name not in _ROUND_METADATA_FILENAMES
        and not path.name.endswith("_perf.txt")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        preferred_python = [path for path in candidates if path.suffix == ".py"]
        if len(preferred_python) == 1:
            return preferred_python[0]
        return candidates[0]
    return None
