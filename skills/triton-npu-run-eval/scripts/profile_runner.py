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
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)


def _profile_timeout() -> int:
    return env_int("TRITON_AGENT_PROFILE_TIMEOUT_SECONDS", 900)


def run_local_profile_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    bench_case: int | None = None,
    case_id: str | None = None,
    kernel_name: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    if bench_mode == "msprof":
        if case_id is not None:
            raise ValueError("--case-id is only valid for standalone benchmark profiling")
        result = _run_local_profile_msprof(bench_file, operator_file, bench_case, kernel_name)
    else:
        if case_id is None:
            raise ValueError("Standalone benchmark profiling requires --case-id <id>.")
        result = _run_local_profile_standalone(bench_file, operator_file, case_id)
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
    bench_case: int | None = None,
    case_id: str | None = None,
    kernel_name: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
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
                raise ValueError("--case-id is only valid for standalone benchmark profiling")
            result = _run_remote_profile_msprof(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
                bench_case,
                kernel_name,
                verbose=verbose,
                stderr=stderr,
            )
        else:
            if case_id is None:
                raise ValueError("Standalone benchmark profiling requires --case-id <id>.")
            result = _run_remote_profile_standalone(
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


def _run_local_profile_standalone(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    return profile_local_standalone_case(bench_file, operator_file, case_id)


def _run_local_profile_msprof(
    bench_file: Path,
    operator_file: Path,
    bench_case: int | None,
    _requested_kernel_name: str | None,
) -> ResultPayload:
    selected_case = _resolve_bench_case_local(bench_file, bench_case)
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    return run_streaming_process(
        [
            "msprof",
            local_python_executable(),
            bench_file.name,
            "--operator-file",
            operator_arg,
            "--bench",
            str(selected_case),
        ],
        str(bench_file.parent),
        stall_timeout_seconds=_profile_timeout(),
    )


def _run_remote_profile_standalone(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> ResultPayload:
    for support_path in _standalone_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    return run_remote_command_streaming(
        spec,
        remote_workspace,
        [
            "python3",
            "-c",
            _build_remote_standalone_profile_script(),
            bench_file.name,
            operator_file.name,
            case_id,
        ],
        stall_timeout_seconds=_profile_timeout(),
        verbose=verbose,
        stderr=stderr,
    )


def _run_remote_profile_msprof(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    bench_case: int | None,
    _requested_kernel_name: str | None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> ResultPayload:
    selected_case = _resolve_bench_case_remote(
        spec,
        remote_workspace,
        bench_file,
        bench_case,
        verbose=verbose,
        stderr=stderr,
    )
    return run_remote_command_streaming(
        spec,
        remote_workspace,
        [
            "msprof",
            "op",
            "python3",
            bench_file.name,
            "--operator-file",
            operator_file.name,
            "--bench",
            str(selected_case),
        ],
        stall_timeout_seconds=_profile_timeout(),
        verbose=verbose,
        stderr=stderr,
    )

def _resolve_bench_case_local(bench_file: Path, bench_case: int | None) -> int:
    count_result = run_buffered_process(
        [local_python_executable(), bench_file.name, "--num-bench"],
        str(bench_file.parent),
        stall_timeout_seconds=_profile_timeout(),
    )
    if not result_succeeded(count_result):
        raise RuntimeError(str(count_result["stderr"]) or str(count_result["stdout"]) or "Unable to query benchmark cases.")
    return _normalize_bench_case(_parse_case_count(str(count_result["stdout"])), bench_case)


def _resolve_bench_case_remote(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    bench_case: int | None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    count_result = run_remote_command_buffered(
        spec,
        remote_workspace,
        ["python3", bench_file.name, "--num-bench"],
        stall_timeout_seconds=_profile_timeout(),
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(count_result):
        raise RuntimeError(str(count_result["stderr"]) or str(count_result["stdout"]) or "Unable to query benchmark cases.")
    return _normalize_bench_case(_parse_case_count(str(count_result["stdout"])), bench_case)


def _normalize_bench_case(case_count: int, bench_case: int | None) -> int:
    selected_case = 1 if bench_case is None else bench_case
    if selected_case < 1 or selected_case > case_count:
        raise ValueError(
            f"Requested benchmark case {selected_case} is out of range; available cases: 1..{case_count}"
        )
    return selected_case


def _parse_case_count(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ValueError("Unable to parse benchmark case count from --num-bench output.")


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


def profile_local_standalone_case(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
) -> ResultPayload:
    runtime = _load_standalone_runtime_module()
    return runtime.profile_local_standalone_case(bench_file, operator_file, case_id)


def _standalone_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("standalone_bench_runtime.py")


def _standalone_runtime_support_paths() -> list[Path]:
    runtime = _load_standalone_runtime_module()
    return list(runtime.runtime_support_paths())


def _load_standalone_runtime_module():
    script_path = _standalone_runtime_script_path()
    module_name = f"triton_agent_standalone_bench_runtime_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load standalone runtime helper: {script_path}")
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


def _build_remote_standalone_profile_script() -> str:
    return (
        "import pathlib, sys; "
        "import standalone_bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "result = runtime.profile_local_standalone_case(bench_file, operator_file, case_id); "
        "raise SystemExit(int(result['return_code']))"
    )
