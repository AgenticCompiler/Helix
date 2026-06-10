from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import TextIO

from run_runtime import (
    RemoteSpec,
    ResultPayload,
    env_int,
    cleanup_remote_workspace,
    copy_directory_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    local_python_executable,
    result_succeeded,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)


def _profile_timeout() -> int:
    return env_int("TRITON_AGENT_PROFILE_TIMEOUT_SECONDS", 900)


def _normalize_bench_mode(bench_mode: str) -> str:
    return "torch-npu-profiler" if bench_mode == "standalone" else bench_mode


def run_local_profile_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    bench_mode = _normalize_bench_mode(bench_mode)
    del kernel_name
    if bench_mode == "msprof":
        result = _run_local_profile_msprof(bench_file, operator_file, case_id)
    else:
        if case_id is None:
            raise ValueError("torch-npu-profiler benchmark profiling requires --case-id <id>.")
        result = _run_local_profile_torch_npu_profiler(
            bench_file, operator_file, case_id,
        )
    if not result_succeeded(result):
        return result, None
    profile_dir = _resolve_local_profile_dir(bench_file.parent)
    return result, profile_dir


def run_remote_profile_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    case_id: str | None = None,
    kernel_name: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    bench_mode = _normalize_bench_mode(bench_mode)
    del kernel_name
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    try:
        copy_file_to_remote(
            spec, bench_file, f"{remote_workspace}/{bench_file.name}", verbose=verbose, stderr=stderr
        )
        copy_file_to_remote(
            spec,
            operator_file,
            f"{remote_workspace}/{operator_file.name}",
            verbose=verbose,
            stderr=stderr,
        )
        if bench_mode == "msprof":
            if case_id is not None:
                result = _run_remote_profile_msprof(
                    spec,
                    remote_workspace,
                    bench_file,
                    operator_file,
                    case_id,
                    verbose=verbose,
                    stderr=stderr,
                )
            else:
                result = _run_remote_profile_msprof(
                    spec,
                    remote_workspace,
                    bench_file,
                    operator_file,
                    None,
                    verbose=verbose,
                    stderr=stderr,
                )
        else:
            if case_id is None:
                raise ValueError("torch-npu-profiler benchmark profiling requires --case-id <id>.")
            result = _run_remote_profile_torch_npu_profiler(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
                case_id,
                verbose=verbose,
                stderr=stderr,
            )
        if not result_succeeded(result):
            return result, None, remote_workspace
        remote_profile_name = _resolve_remote_profile_name(
            spec,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        local_profile_dir = operator_file.parent / remote_profile_name
        if local_profile_dir.exists():
            raise FileExistsError(f"Local profile directory already exists: {local_profile_dir}")
        copy_directory_from_remote(
            spec,
            f"{remote_workspace}/{remote_profile_name}",
            local_profile_dir,
            verbose=verbose,
            stderr=stderr,
        )
        _validate_profile_dir(local_profile_dir)
        return result, local_profile_dir, remote_workspace
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _run_local_profile_torch_npu_profiler(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    prev = os.environ.get("TRITON_ALWAYS_COMPILE")
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    try:
        return profile_local_torch_npu_profiler_case(bench_file, operator_file, case_id)
    finally:
        if prev is None:
            del os.environ["TRITON_ALWAYS_COMPILE"]
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev


def _run_local_profile_msprof(
    bench_file: Path,
    operator_file: Path,
    case_id: str | None,
) -> ResultPayload:
    selected_case = _resolve_bench_case(bench_file, operator_file, case_id)
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    return run_streaming_process(
        [
            "msprof",
            local_python_executable(),
            "bench_runtime.py",
            "run-one",
            "--bench-file",
            bench_file.name,
            "--operator-file",
            operator_arg,
            "--case-id",
            selected_case,
        ],
        str(bench_file.parent),
        stall_timeout_seconds=_profile_timeout(),
        extra_env={"TRITON_ALWAYS_COMPILE": "1"},
    )


def _run_remote_profile_torch_npu_profiler(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> ResultPayload:
    for support_path in _bench_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    extra_env = {"TRITON_ALWAYS_COMPILE": "1"}
    return run_remote_command_streaming(
        spec,
        remote_workspace,
        [
            "python3",
            "-c",
            _build_remote_torch_npu_profiler_profile_script(),
            bench_file.name,
            operator_file.name,
            case_id,
        ],
        stall_timeout_seconds=_profile_timeout(),
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
    )


def _run_remote_profile_msprof(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    case_id: str | None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> ResultPayload:
    selected_case = _resolve_bench_case(bench_file, operator_file, case_id)
    for support_path in _bench_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    extra_env = {"TRITON_ALWAYS_COMPILE": "1"}
    return run_remote_command_streaming(
        spec,
        remote_workspace,
        [
            "msprof",
            "python3",
            "bench_runtime.py",
            "run-one",
            "--bench-file",
            bench_file.name,
            "--operator-file",
            operator_file.name,
            "--case-id",
            selected_case,
        ],
        stall_timeout_seconds=_profile_timeout(),
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
    )

def _resolve_local_profile_dir(search_root: Path) -> Path:
    candidates = [candidate for candidate in search_root.iterdir() if candidate.is_dir() and candidate.name.startswith("PROF_")]
    if not candidates:
        raise FileNotFoundError(f"No PROF_* directory found under {search_root}")
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    _validate_profile_dir(latest)
    return latest


def _resolve_remote_profile_name(
    spec: RemoteSpec,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        (
            "python3 -c "
            + repr(
                "import pathlib; "
                "candidates = [p for p in pathlib.Path('.').iterdir() if p.is_dir() and p.name.startswith('PROF_')]; "
                "candidates.sort(key=lambda p: p.stat().st_mtime); "
                "print(candidates[-1].name if candidates else '')"
            )
        ),
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(result):
        raise RuntimeError(str(result["stderr"]) or str(result["stdout"]) or "Failed to resolve remote profiler output.")
    profile_name = str(result["stdout"]).strip().splitlines()[-1].strip() if str(result["stdout"]).strip() else ""
    if not profile_name:
        raise FileNotFoundError(f"No PROF_* directory found in remote workspace {remote_workspace}")
    return profile_name


def _validate_profile_dir(profile_dir: Path) -> None:
    output_dir = profile_dir / "mindstudio_profiler_output"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Profiler output is incomplete: missing {output_dir}")
    if not list(output_dir.glob("op_statistic_*.csv")):
        raise FileNotFoundError(f"Profiler output is incomplete: no op_statistic_*.csv under {output_dir}")


def profile_local_torch_npu_profiler_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    runtime = _load_bench_runtime_module()
    return runtime.profile_local_bench_case(bench_file, operator_file, case_id)


def _bench_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("bench_runtime.py")


def _bench_runtime_support_paths() -> list[Path]:
    runtime = _load_bench_runtime_module()
    return list(runtime.runtime_support_paths())


def _load_bench_runtime_module():
    script_path = _bench_runtime_script_path()
    module_name = f"triton_agent_bench_runtime_{script_path.stem}"
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


def _resolve_bench_case(bench_file: Path, operator_file: Path, case_id: str | None) -> str:
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
    if case_id is None:
        if len(cases) == 1:
            return cases[0].case_id
        available = ", ".join(case.case_id for case in cases)
        raise ValueError(
            "Benchmark profiling requires --case-id when multiple cases exist. "
            f"Available case ids: {available}"
        )
    for case in cases:
        if case.case_id == case_id:
            return case.case_id
    available = ", ".join(case.case_id for case in cases)
    raise ValueError(f"Unknown benchmark case id '{case_id}'. Available case ids: {available}")


def _build_remote_torch_npu_profiler_profile_script() -> str:
    return (
        "import pathlib, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "result = runtime.profile_local_bench_case(bench_file, operator_file, case_id); "
        "raise SystemExit(int(result['return_code']))"
    )
