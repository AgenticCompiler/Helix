from __future__ import annotations

import os
from pathlib import Path
from env_registry import HELIX_PROFILE_TIMEOUT_SECONDS, TRITON_ALWAYS_COMPILE
import run_bench_execution
from run_runtime import (
    ResultPayload,
    env_int,
)


def profile_timeout() -> int:
    return env_int(HELIX_PROFILE_TIMEOUT_SECONDS, 900)


def execute_local_profile(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    prev = os.environ.get(TRITON_ALWAYS_COMPILE)
    os.environ[TRITON_ALWAYS_COMPILE] = "1"
    try:
        return profile_local_torch_npu_profiler_case(bench_file, operator_file, case_id)
    finally:
        if prev is None:
            del os.environ[TRITON_ALWAYS_COMPILE]
        else:
            os.environ[TRITON_ALWAYS_COMPILE] = prev


def resolve_local_profile_dir(search_root: Path) -> Path:
    candidates = [
        candidate
        for candidate in search_root.rglob("PROF_*")
        if candidate.is_dir() and _is_valid_profile_dir(candidate)
    ]
    if not candidates:
        raise FileNotFoundError(f"No PROF_* directory found under {search_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    return latest


def validate_profile_dir(profile_dir: Path) -> None:
    output_dir = profile_dir / "mindstudio_profiler_output"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Profiler output is incomplete: missing {output_dir}")
    if not list(output_dir.glob("op_statistic_*.csv")):
        raise FileNotFoundError(f"Profiler output is incomplete: no op_statistic_*.csv under {output_dir}")


def _is_valid_profile_dir(profile_dir: Path) -> bool:
    try:
        validate_profile_dir(profile_dir)
    except FileNotFoundError:
        return False
    return True


def profile_local_torch_npu_profiler_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    return run_bench_execution.profile_bench_case_quick(bench_file, operator_file, case_id)
