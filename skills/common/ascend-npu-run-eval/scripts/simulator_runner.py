from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from bench_contract import resolve_bench_kernel_resolution
from result_payload import ResultPayload
from run_runtime import env_int, local_python_executable, run_streaming_process


def _simulator_timeout() -> int:
    return env_int("TRITON_AGENT_BENCH_TIMEOUT_SECONDS", 900)


def _simulator_soc_version() -> str:
    return os.environ.get("TRITON_AGENT_SIMULATOR_SOC_VERSION", "Ascend950PR_9599")


def _bench_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("bench_runtime.py")


def _load_bench_runtime_module():
    script_path = _bench_runtime_script_path()
    module_name = f"triton_agent_simulator_runtime_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load bench runtime helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    script_dir = str(script_path.parent)
    added = False
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        added = True
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        if added:
            sys.path.remove(script_dir)
    return module


def _resolve_selected_case_id(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None,
) -> str:
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
    case = runtime.select_bench_case(cases, case_id)
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
        str(_bench_runtime_script_path()),
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
        extra_env={"TRITON_ALWAYS_COMPILE": "1"},
    )
