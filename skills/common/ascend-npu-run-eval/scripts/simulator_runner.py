from __future__ import annotations

import os
from pathlib import Path

from bench_contract import resolve_bench_kernel_resolution
from env_registry import (
    HELIX_BENCH_TIMEOUT_SECONDS,
    HELIX_SIMULATOR_SOC_VERSION,
    TRITON_ALWAYS_COMPILE,
)
from result_payload import ResultPayload
import run_bench_execution
from run_runtime import env_int, local_python_executable, run_streaming_process


def _simulator_timeout() -> int:
    return env_int(HELIX_BENCH_TIMEOUT_SECONDS, 900)


def _simulator_soc_version() -> str:
    return os.environ.get(HELIX_SIMULATOR_SOC_VERSION, "Ascend950PR_9599")


def _run_bench_execution_script_path() -> Path:
    return Path(__file__).resolve().with_name("run_bench_execution.py")


def _resolve_selected_case_id(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None,
) -> str:
    cases, _resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
    case = run_bench_execution.select_bench_case(cases, case_id)
    return str(case.case_id)


def _resolve_selected_kernel_name(
    bench_file: Path,
    operator_file: Path,
    kernel_name: str | None,
) -> str:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    kernel_names = resolution.kernel_names
    if kernel_name is not None:
        if kernel_name not in kernel_names:
            available = ", ".join(kernel_names)
            raise ValueError(f"Unknown simulator kernel '{kernel_name}'. Available kernel names: {available}")
        return kernel_name
    if len(kernel_names) == 1:
        return kernel_names[0]
    available = ", ".join(kernel_names)
    raise ValueError(
        "run-simulator requires --kernel-name when multiple kernels resolve. "
        f"Available kernel names: {available}"
    )


def run_local_simulator(
    bench_file: Path,
    operator_file: Path,
    *,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> ResultPayload:
    selected_case = _resolve_selected_case_id(bench_file, operator_file, case_id)
    selected_kernel = _resolve_selected_kernel_name(bench_file, operator_file, kernel_name)
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    command = [
        "msprof",
        "op",
        "simulator",
        f"--soc-version={_simulator_soc_version()}",
        f"--kernel-name={selected_kernel}",
        local_python_executable(),
        str(_run_bench_execution_script_path()),
        "run-one",
        "--bench-file",
        bench_file.name,
        "--operator-file",
        operator_arg,
        "--case-id",
        selected_case,
    ]
    return run_streaming_process(
        command,
        str(bench_file.parent),
        stall_timeout_seconds=_simulator_timeout(),
        extra_env={TRITON_ALWAYS_COMPILE: "1"},
    )
