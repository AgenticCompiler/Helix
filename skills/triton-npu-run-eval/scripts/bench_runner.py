from __future__ import annotations
# pyright: reportUnusedImport=false, reportUnusedFunction=false

import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import bench_runner_msprof as _msprof
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, TypeVar, cast

from bench_contract import (  # noqa: F401
    KernelResolution,
    parse_bench_metadata,
    resolve_bench_kernel_names,
    resolve_bench_kernel_resolution,
)
from npu_affinity import NpuDevicePool, affinity_env_for_device, parse_npu_devices
from debug_device import maybe_print_visible_devices
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    PerfOpRow,
    perf_output_path,
    render_perf_case_records,
    render_perf_case_records_jsonl,
    write_perf_lines,
)
from profile_csv_parser import (
    find_latest_op_statistic_csv,
    parse_op_statistic_csv,
    resolve_perf_metrics,
)
from result_payload import ResultPayload, make_result
from run_runtime import (
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    emit_verbose,
    env_int,
    local_python_executable,
    make_result,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)

_LOCAL_BENCH_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_OUTPUT_DIR"
_BENCH_COPY_FILES_ENV = "TRITON_AGENT_BENCH_COPY_FILES"
_bench_runtime_module_cache = None
_bench_runtime_module_lock = threading.Lock()
_T = TypeVar("_T")
_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched op_statistic csv"
_PRESERVED_RUN_DIR_NONE_SENTINEL = "__NONE__"


@dataclass(frozen=True)
class _MsprofCaseOutcome:
    case_id: str
    record: PerfCaseRecord
    stdout: str
    stderr: str
    stalled: bool
    session_id: str | None


def _bench_timeout() -> int:
    return env_int("TRITON_AGENT_BENCH_TIMEOUT_SECONDS", 900)


def _collect_env_copy_files(search_dir: Path) -> list[Path]:
    patterns_str = os.environ.get(_BENCH_COPY_FILES_ENV, "")
    if not patterns_str.strip():
        return []
    patterns = [p.strip() for p in patterns_str.split(",") if p.strip()]
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for matched in sorted(search_dir.glob(pattern)):
            if matched.is_file():
                resolved = matched.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(matched)
    return paths


def _normalize_bench_mode(bench_mode: str) -> str:
    return "torch-npu-profiler" if bench_mode == "standalone" else bench_mode


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
    extract_dest_dir: Path | None = None,
    simulator_case_idx: int = 1,
) -> tuple[ResultPayload, Path | None]:
    bench_mode = _normalize_bench_mode(bench_mode)
    invocation_root = Path.cwd().resolve()
    devices = parse_npu_devices(npu_devices)
    maybe_print_visible_devices()
    with _local_bench_workdir(bench_file.parent):
        if bench_mode == "msprof-simulator":
            resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
            with ThreadPoolExecutor(max_workers=2) as executor:
                msprof_future = executor.submit(
                    _run_local_bench_msprof,
                    bench_file,
                    operator_file,
                    verbose=verbose,
                )
                kernel_name = _run_local_msprof_single_case_for_kernel(
                    bench_file,
                    operator_file,
                    resolution.kernel_names,
                    bench_case=simulator_case_idx,
                    verbose=verbose,
                )
                simulator_future = executor.submit(
                    _run_local_bench_msprof_simulator,
                    bench_file,
                    operator_file,
                    extract_dest_dir=extract_dest_dir,
                    kernel_name=kernel_name,
                    simulator_case_idx=simulator_case_idx,
                    verbose=verbose,
                )
                simulator_future.result()
                return msprof_future.result()
        if bench_mode == "msprof":
            if devices is not None:
                source_root, json_search_root = _resolve_case_workspace_roots(
                    bench_file,
                    operator_file,
                    invocation_root=invocation_root,
                )
                return _run_local_bench_msprof_parallel(
                    bench_file,
                    operator_file,
                    devices,
                    source_root=source_root,
                    json_search_root=json_search_root,
                    verbose=verbose,
                    output=output,
                )
            return _run_local_bench_msprof(bench_file, operator_file, verbose=verbose,
                                           output=output)
        if devices is not None:
            source_root, json_search_root = _resolve_case_workspace_roots(
                bench_file,
                operator_file,
                invocation_root=invocation_root,
            )
            return _run_local_bench_torch_npu_profiler_parallel(
                bench_file,
                operator_file,
                devices,
                source_root=source_root,
                json_search_root=json_search_root,
                verbose=verbose,
                output=output,
            )
        resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
        with ThreadPoolExecutor(max_workers=2) as executor:
            standalone_future = executor.submit(
                _run_local_bench_torch_npu_profiler,
                bench_file,
                operator_file,
                verbose=verbose,
                output=output,
            )
            kernel_name = _run_local_msprof_single_case_for_kernel(
                bench_file,
                operator_file,
                resolution.kernel_names,
                verbose=verbose,
            )
            simulator_future = executor.submit(
                _run_local_bench_msprof_simulator_standalone,
                bench_file,
                operator_file,
                extract_dest_dir=extract_dest_dir,
                kernel_name=kernel_name,
                verbose=verbose,
            )
            simulator_future.result()
            return standalone_future.result()
        return _run_local_bench_torch_npu_profiler(bench_file, operator_file, verbose=verbose,
                                           output=output)


def run_remote_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    npu_devices: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    bench_mode = _normalize_bench_mode(bench_mode)
    invocation_root = Path.cwd().resolve()
    devices = parse_npu_devices(npu_devices)
    maybe_print_visible_devices()
    spec, remote_workspace = create_remote_workspace(
        remote, remote_workdir, verbose=verbose, stderr=stderr
    )
    local_bench_cases = bench_file.with_suffix(".json")
    try:
        copy_file_to_remote(
            spec, bench_file, f"{remote_workspace}/{bench_file.name}", verbose=verbose, stderr=stderr
        )
        if local_bench_cases.exists():
            copy_file_to_remote(
                spec,
                local_bench_cases,
                f"{remote_workspace}/{local_bench_cases.name}",
                verbose=verbose,
                stderr=stderr,
            )
        copy_file_to_remote(
            spec,
            operator_file,
            f"{remote_workspace}/{operator_file.name}",
            verbose=verbose,
            stderr=stderr,
        )
        if bench_mode == "msprof":
            if devices is not None:
                source_root, json_search_root = _resolve_case_workspace_roots(
                    bench_file,
                    operator_file,
                    invocation_root=invocation_root,
                )
                return _run_remote_bench_msprof_parallel(
                    spec,
                    remote_workspace,
                    bench_file,
                    operator_file,
                    devices,
                    source_root=source_root,
                    json_search_root=json_search_root,
                    verbose=verbose,
                    stderr=stderr,
                    output=output,
                )
            return _run_remote_bench_msprof(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
                verbose=verbose,
                stderr=stderr,
                output=output,
            )
        if devices is not None:
            source_root, json_search_root = _resolve_case_workspace_roots(
                bench_file,
                operator_file,
                invocation_root=invocation_root,
            )
            return _run_remote_bench_torch_npu_profiler_parallel(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
                devices,
                source_root=source_root,
                json_search_root=json_search_root,
                verbose=verbose,
                stderr=stderr,
                output=output,
            )
        return _run_remote_bench_torch_npu_profiler(
            spec,
            remote_workspace,
            bench_file,
            operator_file,
            verbose=verbose,
            stderr=stderr,
            output=output,
        )
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _run_local_bench_torch_npu_profiler(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    runtime = _load_bench_runtime_module()
    return runtime.profile_all_bench_cases(
        bench_file,
        operator_file,
        verbose=verbose,
        output=output,
    )


@contextlib.contextmanager
def _local_bench_workdir(workdir: Path):
    original_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def _cleanup_local_bench_extra_info(workdir: Path) -> None:
    extra_info_dir = workdir / "extra-info"
    if not extra_info_dir.is_dir():
        return
    shutil.rmtree(extra_info_dir)


@contextlib.contextmanager
def _stream_target_for_verbosity(verbose: bool) -> Iterator[TextIO]:
    if verbose:
        yield sys.stdout
        return
    with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
        yield quiet_stdout


def _stage_remote_bench_runtime_support_files(
    spec: RemoteSpec,
    remote_workspace: str,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    for support_path in _bench_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )


def _run_remote_bench_torch_npu_profiler(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    _stage_remote_bench_runtime_support_files(
        spec,
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    extra_env: dict[str, str] | None = {"TRITON_ALWAYS_COMPILE": "1"}
    with _stream_target_for_verbosity(verbose) as stream_target:
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                _build_remote_torch_npu_profiler_run_all_script(verbose=verbose),
                bench_file.name,
                operator_file.name,
                perf_path.name,
            ],
            stdout=stream_target,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=_bench_timeout(),
            extra_env=extra_env,
        )
    copied_perf_path: Path | None = None
    try:
        copy_file_from_remote(
            spec,
            f"{remote_workspace}/{perf_path.name}",
            perf_path,
            verbose=verbose,
            stderr=stderr,
        )
        copied_perf_path = perf_path
    except RuntimeError:
        if result_succeeded(result):
            raise
    return result, copied_perf_path, remote_workspace


def _run_local_bench_torch_npu_profiler_parallel(
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path]:
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)
    preserved_run_dir: Path | None = None
    create_preserved_run_dir = getattr(runtime, "create_local_preserved_profile_run_dir", None)
    if callable(create_preserved_run_dir):
        preserved_run_dir = cast(
            Path | None,
            create_preserved_run_dir(prefix="triton-agent-torch-npu-profiler-bench-"),
        )

    def _worker(case_id: str) -> PerfCaseRecord:
        case_workspace, cleanup = _create_local_torch_npu_profiler_case_workspace(
            bench_file,
            operator_file,
            case_id,
            source_root=source_root,
            json_search_root=json_search_root,
            verbose=verbose,
        )
        try:
            with pool.acquire() as device:
                return _run_local_torch_npu_profiler_case_in_subprocess(
                    case_workspace,
                    bench_file,
                    operator_file,
                    case_id,
                    device,
                    preserved_run_dir=preserved_run_dir,
                    source_root=source_root,
                    verbose=verbose,
                )
        finally:
            cleanup()

    case_records = _run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    _sort_case_records(case_records, case_ids)
    perf_path = _write_torch_npu_profiler_perf(operator_file, case_records, output=output)
    return _build_torch_npu_profiler_result(case_records), perf_path


def _run_remote_bench_torch_npu_profiler_parallel(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[ResultPayload, Path, str]:
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)

    def _worker(case_id: str) -> PerfCaseRecord:
        case_workspace = f"{remote_workspace}/case-{case_id}"
        run_remote_command_buffered(
            spec,
            remote_workspace,
            ["mkdir", "-p", case_workspace],
            verbose=verbose,
            stderr=stderr,
        )
        workspace_root = _stage_remote_torch_npu_profiler_case_workspace(
            spec,
            bench_file,
            operator_file,
            case_workspace,
            source_root=source_root,
            json_search_root=json_search_root,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            with pool.acquire() as device:
                return _run_remote_torch_npu_profiler_case(
                    spec,
                    workspace_root,
                    bench_file,
                    operator_file,
                    case_id,
                    device,
                    source_root=source_root,
                    verbose=verbose,
                    stderr=stderr,
                )
        finally:
            run_remote_command_buffered(
                spec,
                remote_workspace,
                ["rm", "-rf", case_workspace],
                verbose=verbose,
                stderr=stderr,
            )

    case_records = _run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    _sort_case_records(case_records, case_ids)
    perf_path = _write_torch_npu_profiler_perf(operator_file, case_records, output=output)
    return _build_torch_npu_profiler_result(case_records), perf_path, remote_workspace


def _bench_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("bench_runtime.py")


def _bench_runtime_support_paths() -> list[Path]:
    runtime = _load_bench_runtime_module()
    return cast(list[Path], runtime.runtime_support_paths())


def _bench_flat_input_paths(bench_file: Path) -> list[Path]:
    return _bench_runtime_support_paths() + _collect_env_copy_files(bench_file.parent)


def _load_bench_runtime_module():
    global _bench_runtime_module_cache
    cached_module = _bench_runtime_module_cache
    if cached_module is not None:
        return cached_module

    with _bench_runtime_module_lock:
        cached_module = _bench_runtime_module_cache
        if cached_module is not None:
            return cached_module

        script_path = _bench_runtime_script_path()
        module_name = f"triton_agent_bench_runtime_{script_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load bench runtime helper: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        _bench_runtime_module_cache = module
        return module


def _run_local_bench_msprof(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    runtime = _load_bench_runtime_module()
    cases, _ignored_resolution = runtime.load_bench_cases(bench_file, operator_file)
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    case_records: list[PerfCaseRecord] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    for case in cases:
        output_dir, temp_dir = _create_local_msprof_output_dir(case.case_id, preserved_run_dir)
        try:
            command = [
                "msprof",
                f"--output={output_dir}",
                local_python_executable(),
                str(_bench_runtime_script_path()),
                "run-one",
                "--bench-file",
                bench_file.name,
                "--operator-file",
                operator_arg,
                "--case-id",
                case.case_id,
            ]
            t0 = time.monotonic()
            with _stream_target_for_verbosity(verbose) as stream_target:
                result = run_streaming_process(
                    command,
                    str(bench_file.parent),
                    stall_timeout_seconds=_bench_timeout(),
                    stdout=stream_target,
                    extra_env={"TRITON_ALWAYS_COMPILE": "1"},
                )
            elapsed = time.monotonic() - t0
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=case.case_id,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=_format_msprof_command_failure(result),
                        case_wall_clock_seconds=elapsed,
                    ),
                )
                continue

            try:
                metrics = _read_local_msprof_metrics(output_dir, resolution.kernel_names)
            except (FileNotFoundError, ValueError) as exc:
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=case.case_id,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                        case_wall_clock_seconds=elapsed,
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=case.case_id,
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                    case_wall_clock_seconds=elapsed,
                )
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            _cleanup_local_bench_extra_info(bench_file.parent)

    perf_path = _write_msprof_perf(operator_file, case_records, output=output)
    return (
        make_result(
            return_code=1 if had_case_failures else 0,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            stalled=had_stalls,
            session_id=session_id,
        ),
        perf_path,
    )


def _run_local_bench_msprof_parallel(
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None]:
    runtime = _load_bench_runtime_module()
    cases, _ignored_resolution = runtime.load_bench_cases(bench_file, operator_file)
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    bench_arg = _case_workspace_command_path(bench_file, source_root=source_root)
    operator_arg = _case_workspace_command_path(operator_file, source_root=source_root)
    runtime_arg = _bench_runtime_script_path().name
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)

    def _worker(case_id: str) -> _MsprofCaseOutcome:
        return _run_local_msprof_case_parallel(
            bench_file,
            operator_file,
            operator_arg,
            bench_arg,
            runtime_arg,
            resolution,
            case_id,
            pool,
            preserved_run_dir,
            source_root,
            json_search_root,
            verbose,
        )

    outcomes = _run_parallel_case_workers(
        case_ids,
        min(len(case_ids), len(devices)),
        _worker,
    )
    case_order = {case_id: index for index, case_id in enumerate(case_ids)}
    outcomes.sort(key=lambda outcome: case_order[outcome.case_id])
    perf_path = _write_msprof_perf(operator_file, [outcome.record for outcome in outcomes], output=output)
    for outcome in outcomes:
        stdout_chunks.append(outcome.stdout)
        stderr_chunks.append(outcome.stderr)
    return _build_msprof_result(stdout_chunks, stderr_chunks, outcomes), perf_path



def _run_remote_bench_msprof(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    runtime = _load_bench_runtime_module()
    cases, _ignored_resolution = runtime.load_bench_cases(bench_file, operator_file)
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    case_records: list[PerfCaseRecord] = []
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    _stage_remote_bench_runtime_support_files(
        spec,
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )

    for case in cases:
        output_dir = _create_remote_msprof_output_dir(
            spec,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            t0 = time.monotonic()
            result = run_remote_command_streaming(
                spec,
                remote_workspace,
                [
                    "msprof",
                    f"--output={output_dir}",
                    "python3",
                    _bench_runtime_script_path().name,
                    "run-one",
                    "--bench-file",
                    bench_file.name,
                    "--operator-file",
                    operator_file.name,
                    "--case-id",
                    case.case_id,
                ],
                verbose=verbose,
                stderr=stderr,
                stall_timeout_seconds=_bench_timeout(),
            )
            elapsed = time.monotonic() - t0
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=case.case_id,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=_format_msprof_command_failure(result),
                        case_wall_clock_seconds=elapsed,
                    ),
                )
                continue

            try:
                metrics = _read_remote_msprof_metrics(
                    spec,
                    remote_workspace,
                    output_dir,
                    resolution.kernel_names,
                    verbose=verbose,
                    stderr=stderr,
                )
            except RuntimeError as exc:
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=case.case_id,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                        case_wall_clock_seconds=elapsed,
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=case.case_id,
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                    case_wall_clock_seconds=elapsed,
                )
            )
        finally:
            _cleanup_remote_msprof_output_dir(
                spec,
                remote_workspace,
                output_dir,
                verbose=verbose,
                stderr=stderr,
            )

    perf_path = _write_msprof_perf(operator_file, case_records, output=output)
    return (
        make_result(
            return_code=1 if had_case_failures else 0,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            stalled=had_stalls,
            session_id=session_id,
        ),
        perf_path,
        remote_workspace,
    )


def _run_remote_bench_msprof_parallel(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    runtime = _load_bench_runtime_module()
    cases, _ignored_resolution = runtime.load_bench_cases(bench_file, operator_file)
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)

    def _worker(case_id: str) -> _MsprofCaseOutcome:
        return _run_remote_msprof_case_parallel(
            spec,
            remote_workspace,
            bench_file,
            operator_file,
            resolution,
            case_id,
            pool,
            source_root,
            json_search_root,
            verbose,
            stderr,
        )

    outcomes = _run_parallel_case_workers(
        case_ids,
        min(len(case_ids), len(devices)),
        _worker,
    )
    case_order = {case_id: index for index, case_id in enumerate(case_ids)}
    outcomes.sort(key=lambda outcome: case_order[outcome.case_id])
    perf_path = _write_msprof_perf(operator_file, [outcome.record for outcome in outcomes], output=output)
    for outcome in outcomes:
        stdout_chunks.append(outcome.stdout)
        stderr_chunks.append(outcome.stderr)
    return _build_msprof_result(stdout_chunks, stderr_chunks, outcomes), perf_path, remote_workspace


def _run_parallel_case_workers(
    case_keys: Sequence[str],
    max_workers: int,
    worker: Callable[[str], _T],
) -> list[_T]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, case_key) for case_key in case_keys]
        return [future.result() for future in futures]


def _sort_case_records(case_records: list[PerfCaseRecord], ordered_case_labels: Sequence[str]) -> None:
    case_order = {label: index for index, label in enumerate(ordered_case_labels)}
    case_records.sort(key=lambda record: case_order[record.case_label])


def _resolve_local_bench_profile_output_root() -> tuple[str | None, str]:
    configured_root = os.environ.get(_LOCAL_BENCH_OUTPUT_DIR_ENV)
    if configured_root:
        return str(Path(configured_root).expanduser().resolve()), _LOCAL_BENCH_OUTPUT_DIR_ENV
    return None, _LOCAL_BENCH_OUTPUT_DIR_ENV


def _create_local_msprof_output_dir(
    case_label: str,
    preserved_run_dir: Path | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if preserved_run_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="triton-agent-msprof-")
        return Path(temp_dir.name), temp_dir
    output_dir = preserved_run_dir.resolve() / f"case-{case_label}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _set_directory_owner_only(output_dir)
    return output_dir, None


def _create_local_msprof_preserved_run_dir() -> Path | None:
    configured_root, configured_env = _resolve_local_bench_profile_output_root()
    if not configured_root:
        return None
    root = Path(configured_root).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"{configured_env} must point to a directory: {root}"
        )
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        _set_directory_owner_only(root)
    run_dir = Path(tempfile.mkdtemp(prefix="triton-agent-msprof-", dir=str(root)))
    _set_directory_owner_only(run_dir)
    return run_dir


def _bench_case_input_paths(
    bench_file: Path,
    operator_file: Path,
    *,
    json_search_root: Path | None = None,
) -> list[Path]:
    input_paths: list[Path] = [bench_file]
    json_roots = [bench_file.parent.resolve(), operator_file.parent.resolve()]
    if json_search_root is not None:
        resolved_json_root = json_search_root.resolve()
        if resolved_json_root not in json_roots:
            json_roots.insert(0, resolved_json_root)
    for json_root in json_roots:
        input_paths.extend(
            sorted(path for path in json_root.glob("*.json") if path.is_file())
        )
    input_paths.append(operator_file)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for input_path in input_paths:
        resolved_path = input_path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        unique_paths.append(input_path)
    return unique_paths


def _path_is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_case_workspace_roots(
    bench_file: Path,
    operator_file: Path,
    *,
    invocation_root: Path | None,
) -> tuple[Path, Path]:
    if invocation_root is not None:
        resolved_invocation_root = invocation_root.resolve()
        workspace_dirs = [bench_file.parent.resolve(), operator_file.parent.resolve()]
        if all(_path_is_within_root(path, resolved_invocation_root) for path in workspace_dirs):
            return resolved_invocation_root, resolved_invocation_root
    source_root = Path(
        os.path.commonpath(
            [
                str(bench_file.parent.resolve()),
                str(operator_file.parent.resolve()),
            ]
        )
    )
    return source_root, bench_file.parent.resolve()


def _case_workspace_root_name(source_root: Path) -> str:
    return source_root.name or "workspace"


def _case_workspace_root_relative_path(path: Path, *, source_root: Path) -> Path:
    try:
        return path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return Path(path.name)


def _case_workspace_command_path(path: Path, *, source_root: Path) -> str:
    return _case_workspace_root_relative_path(path, source_root=source_root).as_posix()


def _emit_case_workspace_verbose(message: str, *, stderr: TextIO | None = None) -> None:
    emit_verbose(stderr or sys.stderr, "files", message)


def _create_local_case_workspace(
    *,
    prefix: str,
    input_paths: Sequence[Path],
    flat_input_paths: Sequence[Path] = (),
    source_root: Path,
    verbose: bool = False,
) -> tuple[Path, Callable[[], None]]:
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    workspace = Path(temp_dir.name)
    workspace_root = workspace / _case_workspace_root_name(source_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    if verbose:
        _emit_case_workspace_verbose(f"created local case workspace: {workspace_root}")
    for input_path in input_paths:
        relative_path = _case_workspace_root_relative_path(input_path, source_root=source_root)
        target_path = workspace_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, target_path)
        if verbose:
            _emit_case_workspace_verbose(f"copied local case file: {input_path} -> {target_path}")
    for input_path in flat_input_paths:
        target_path = workspace_root / input_path.name
        shutil.copyfile(input_path, target_path)
        if verbose:
            _emit_case_workspace_verbose(f"copied local case support file: {input_path} -> {target_path}")
    return workspace_root, temp_dir.cleanup


def _create_local_msprof_case_workspace(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
) -> tuple[Path, Callable[[], None]]:
    return _create_local_case_workspace(
        prefix=f"triton-agent-msprof-case-{case_id}-",
        input_paths=_bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        flat_input_paths=_bench_flat_input_paths(bench_file),
        source_root=source_root,
        verbose=verbose,
    )


def _stage_remote_case_workspace(
    spec: RemoteSpec,
    case_workspace: str,
    input_paths: Sequence[Path],
    source_root: Path,
    *,
    flat_input_paths: Sequence[Path] = (),
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    workspace_root = f"{case_workspace}/{_case_workspace_root_name(source_root)}"
    run_remote_command_buffered(
        spec,
        case_workspace,
        ["mkdir", "-p", workspace_root],
        verbose=verbose,
        stderr=stderr,
    )
    if verbose:
        _emit_case_workspace_verbose(f"created remote case workspace: {workspace_root}", stderr=stderr)
    created_dirs = {workspace_root}
    for input_path in input_paths:
        relative_path = _case_workspace_root_relative_path(input_path, source_root=source_root)
        target_dir = (
            workspace_root
            if relative_path.parent == Path(".")
            else f"{workspace_root}/{relative_path.parent.as_posix()}"
        )
        if target_dir not in created_dirs:
            run_remote_command_buffered(
                spec,
                case_workspace,
                ["mkdir", "-p", target_dir],
                verbose=verbose,
                stderr=stderr,
            )
            created_dirs.add(target_dir)
        copy_file_to_remote(
            spec,
            input_path,
            f"{workspace_root}/{relative_path.as_posix()}",
            verbose=verbose,
            stderr=stderr,
        )
        if verbose:
            _emit_case_workspace_verbose(
                f"copied remote case file: {input_path} -> {workspace_root}/{relative_path.as_posix()}",
                stderr=stderr,
            )
    for input_path in flat_input_paths:
        target_path = f"{workspace_root}/{input_path.name}"
        copy_file_to_remote(
            spec,
            input_path,
            target_path,
            verbose=verbose,
            stderr=stderr,
        )
        if verbose:
            _emit_case_workspace_verbose(
                f"copied remote case support file: {input_path} -> {target_path}",
                stderr=stderr,
            )
    return workspace_root


def _stage_remote_msprof_case_workspace(
    spec: RemoteSpec,
    bench_file: Path,
    operator_file: Path,
    case_workspace: str,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    return _stage_remote_case_workspace(
        spec,
        case_workspace,
        _bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        source_root=source_root,
        flat_input_paths=_bench_flat_input_paths(bench_file),
        verbose=verbose,
        stderr=stderr,
    )


def _set_directory_owner_only(path: Path) -> None:
    path.chmod(0o700)


_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched op_statistic csv"
_TIMEOUT_MESSAGE = "[INFO]  The timeout has reached and the application will be forcibly killed."
_LAUNCH_COUNT = 5  # msprof op simulator: per-kernel max launches; last sample (idx=N-1) is post-warmup


def _iter_kernel_launch_bins(output_dir: Path, kernel_dir_name: str) -> list[Path]:
    kernel_root = output_dir / kernel_dir_name
    if not kernel_root.is_dir():
        return []
    bins: list[Path] = []
    # add commit
    # for entry in kernel_root.iterdir():
    for entry in sorted(kernel_root.iterdir(), key=lambda e: e.name):
        if entry.is_dir() and entry.name.isdigit():
            bin_path = entry / "simulator" / "visualize_data.bin"
            if bin_path.is_file():
                bins.append(bin_path)
    # add commit
    if bins:
        print(f"[msprof-simulator] kernel_dir={kernel_dir_name}: {len(bins)} bin(s)", flush=True)
        for b in bins:
            print(f"[msprof-simulator]   idx={b.parent.parent.name} {b}", flush=True)
    # add commit
    return bins


def _bin_sort_key(bin_path: Path) -> tuple[int, int]:
    # bin_path: .../<idx>/simulator/visualize_data.bin
    idx_dir = bin_path.parent.parent if bin_path.parent.name == "simulator" else None
    idx = int(idx_dir.name) if idx_dir is not None and idx_dir.name.isdigit() else -1
    return (idx, bin_path.stat().st_size)


def _resolve_target_visualize_data_bin(
    output_dir: Path,
    kernel_name: str | None,
    candidate_kernel_names: Sequence[str] | None,
) -> Path | None:
    if kernel_name:
        bins = _iter_kernel_launch_bins(output_dir, kernel_name)
        if bins:
            return max(bins, key=_bin_sort_key)
    if candidate_kernel_names:
        for kname in candidate_kernel_names:
            if kname == kernel_name:
                continue
            bins = _iter_kernel_launch_bins(output_dir, kname)
            if bins:
                return max(bins, key=_bin_sort_key)
    fallback_bins: list[Path] = []
    for entry in output_dir.iterdir():
        if entry.is_dir():
            fallback_bins.extend(_iter_kernel_launch_bins(output_dir, entry.name))
    if fallback_bins:
        return max(fallback_bins, key=_bin_sort_key)
    flat = output_dir / "simulator" / "visualize_data.bin"
    if flat.is_file():
        return flat
    matches = sorted(p for p in output_dir.rglob("visualize_data.bin") if p.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _run_extract_and_copy(
    output_dir: Path,
    bench_file: Path,
    *,
    isTimeOut: bool = False,
    dest_dir: Path | None = None,
    kernel_name: str | None = None,
    candidate_kernel_names: Sequence[str] | None = None,
) -> None:
    bin_file = _resolve_target_visualize_data_bin(output_dir, kernel_name, candidate_kernel_names)
    if bin_file is None:
        print("[msprof-simulator] no visualize_data.bin found, skipping extraction", flush=True)
        return
    print(f"[msprof-simulator] selected bin: {bin_file}", flush=True)

    extract_script = Path(__file__).resolve().parent / "extract_profile_bin_data.py"
    cmd = [sys.executable, str(extract_script), str(bin_file)]
    if isTimeOut:
        cmd.append("--isTimeOut")
    run_buffered_process(cmd, str(bin_file.parent), stall_timeout_seconds=_bench_timeout())

    extracted_dir = bin_file.parent / "extracted_bin_data"
    if extracted_dir.exists():
        tmp_dir = (dest_dir or bench_file.parent) / "extracted_bin_data"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        shutil.copytree(extracted_dir, tmp_dir)


def _run_local_msprof_single_case_for_kernel(
    bench_file: Path,
    operator_file: Path,
    kernel_names: list[str],
    *,
    bench_case: int = 1,
    verbose: bool = False,
) -> str | None:
    output_dir, temp_dir = _create_local_msprof_output_dir('0', None)
    try:
        operator_arg = os.path.relpath(operator_file, bench_file.parent)
        command = [
            "msprof",
            f"--output={output_dir}",
            local_python_executable(),
            bench_file.name,
            "--operator-file",
            operator_arg,
            "--bench", str(bench_case),
        ]
        with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
            result = run_streaming_process(
                command,
                str(bench_file.parent),
                stall_timeout_seconds=_bench_timeout(),
                stdout=quiet_stdout,
                extra_env={"TRITON_ALWAYS_COMPILE": "1"},
            )
        if not result_succeeded(result):
            return None
        try:
            metrics = _msprof._read_local_msprof_metrics(output_dir, kernel_names)
        except (FileNotFoundError, ValueError):
            return None
        if not metrics or not metrics.get("ops"):
            return None
        kernel_name_set = set(kernel_names)
        hottest_name: str | None = None
        hottest_time = -1.0
        for op in metrics["ops"]:
            op_type = op.get("op_type", "")
            avg_time = float(op.get("avg_time_us", 0))
            if op_type in kernel_name_set and avg_time > hottest_time:
                hottest_time = avg_time
                hottest_name = op_type
        if hottest_name:
            print(f"[msprof-simulator] resolved hottest kernel: {hottest_name} ({hottest_time}us)", flush=True)
        return hottest_name
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _cleanup_local_bench_extra_info(bench_file.parent)


def _run_local_bench_msprof_simulator(
    bench_file: Path,
    operator_file: Path,
    extract_dest_dir: Path | None = None,
    kernel_name: str | None = None,
    simulator_case_idx: int = 1,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()
    had_stalls = False
    session_id: str | None = None

    output_dir, temp_dir = _create_local_msprof_output_dir('0', preserved_run_dir)
    try:
        command = [
            "msprof",
            "op",
            "simulator",
            "--timeout=5",
            f"--launch-count={_LAUNCH_COUNT}",
            f"--output={output_dir}",
            local_python_executable(),
            str(bench_file),
            "--operator-file",
            str(operator_file),
            "--bench", str(simulator_case_idx),
        ]
        print(f"[msprof-simulator] kernel-name={kernel_name}, cmd: {' '.join(command)})", flush=True)
        t0 = time.monotonic()
        with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
            result = run_streaming_process(
                command,
                str(bench_file.parent),
                stall_timeout_seconds=_bench_timeout(),
                stdout=quiet_stdout,
            )
        elapsed = time.monotonic() - t0
        stdout_chunks.append(str(result["stdout"]))
        stderr_chunks.append(str(result["stderr"]))
        had_stalls = had_stalls or bool(result["stalled"])
        if result["session_id"] is not None:
            session_id = result["session_id"]

        isTimeOut = (
            _TIMEOUT_MESSAGE in str(result["stdout"])
            or _TIMEOUT_MESSAGE in str(result["stderr"])
        )

        if not result_succeeded(result):
            case_record = PerfCaseRecord(
                case_label="0",
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=_format_msprof_command_failure(result),
                case_wall_clock_seconds=elapsed,
            )
            perf_path = write_perf_lines(
                perf_output_path(operator_file),
                render_perf_case_records(
                    [case_record],
                    latency_prefix="latency-case",
                    raw_prefix="raw-op-statistic-case",
                    resolved_kernels_prefix="resolved-kernels-case",
                    kernel_source_prefix="kernel-source-case",
                    latency_error_prefix="latency-error-case",
                    missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
                    elapsed_id_prefix="case",
                ),
            )
            return (
                make_result(
                    return_code=1,
                    stdout="".join(stdout_chunks),
                    stderr="".join(stderr_chunks),
                    stalled=had_stalls,
                    session_id=session_id,
                ),
                perf_path,
            )

        _run_extract_and_copy(
            output_dir,
            bench_file,
            isTimeOut=isTimeOut,
            dest_dir=extract_dest_dir,
            kernel_name=kernel_name,
            candidate_kernel_names=resolution.kernel_names,
        )

        return (
            make_result(
                return_code=0,
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
                stalled=had_stalls,
                session_id=session_id,
            ),
            None,
        )

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _cleanup_local_bench_extra_info(bench_file.parent)


def _run_local_bench_msprof_simulator_standalone(
    bench_file: Path,
    operator_file: Path,
    extract_dest_dir: Path | None = None,
    kernel_name: str | None = None,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    runtime = _load_bench_runtime_module()
    cases, _ = runtime.load_standalone_bench_cases(bench_file, operator_file)
    if not cases:
        raise ValueError("No standalone bench cases found")
    selected_case = cases[len(cases) // 2]
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()
    had_stalls = False
    session_id: str | None = None
    wrapper_script = _msprof._build_standalone_msprof_wrapper_script()
    wrapper_script_path = bench_file.parent / f"_standalone_msprof_wrapper_{selected_case.case_id}.py"
    try:
        wrapper_script_path.write_text(wrapper_script, encoding="utf-8")
    except Exception:
        pass

    output_dir, temp_dir = _create_local_msprof_output_dir('0', preserved_run_dir)
    try:
        command = [
            "msprof",
            "op",
            "simulator",
            "--timeout=3",
            f"--launch-count={_LAUNCH_COUNT}",
            f"--output={output_dir}",
            local_python_executable(),
            str(wrapper_script_path),
            str(bench_file),
            str(operator_file),
            selected_case.case_id,
        ]
        if verbose:
            emit_verbose(sys.stderr, "standalone-msprof-simulator", f"kernel-name={kernel_name}, cmd: {' '.join(command)}")
        t0 = time.monotonic()
        with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
            result = run_streaming_process(
                command,
                str(bench_file.parent),
                stall_timeout_seconds=_bench_timeout(),
                stdout=quiet_stdout,
            )
        elapsed = time.monotonic() - t0
        stdout_text = str(result["stdout"])
        stderr_text = str(result["stderr"])
        stdout_chunks.append(stdout_text)
        stderr_chunks.append(stderr_text)
        had_stalls = had_stalls or bool(result["stalled"])
        if result["session_id"] is not None:
            session_id = result["session_id"]

        isTimeOut = (
            _TIMEOUT_MESSAGE in str(result["stdout"])
            or _TIMEOUT_MESSAGE in str(result["stderr"])
        )

        if not result_succeeded(result):
            case_record = PerfCaseRecord(
                case_label=selected_case.case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=_format_msprof_command_failure(result),
                case_wall_clock_seconds=elapsed,
            )
            perf_path = write_perf_lines(
                perf_output_path(operator_file),
                render_perf_case_records(
                    [case_record],
                    latency_prefix="latency-case",
                    raw_prefix="raw-op-statistic-case",
                    resolved_kernels_prefix="resolved-kernels-case",
                    kernel_source_prefix="kernel-source-case",
                    latency_error_prefix="latency-error-case",
                    missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
                    elapsed_id_prefix="case",
                ),
            )
            return (
                make_result(
                    return_code=1,
                    stdout="".join(stdout_chunks),
                    stderr="".join(stderr_chunks),
                    stalled=had_stalls,
                    session_id=session_id,
                ),
                perf_path,
            )

        print(f"[standalone-msprof-simulator] OK, output_dir={output_dir}", flush=True)
        _run_extract_and_copy(
            output_dir,
            bench_file,
            isTimeOut=isTimeOut,
            dest_dir=extract_dest_dir,
            kernel_name=kernel_name,
            candidate_kernel_names=resolution.kernel_names,
        )

        return (
            make_result(
                return_code=0,
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
                stalled=had_stalls,
                session_id=session_id,
            ),
            None,
        )

    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _cleanup_local_bench_extra_info(bench_file.parent)
        print("[standalone-msprof-simulator] done", flush=True)


def _format_msprof_command_failure(result: ResultPayload) -> str:
    return f"msprof command failed with return code {int(result['return_code'])}"


def _create_remote_msprof_output_dir(
    spec: RemoteSpec,
    remote_workspace: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        ["mktemp", "-d"],
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or "Failed to create remote msprof output directory.")
    output_dir = str(result["stdout"]).strip().splitlines()[-1].strip() if str(result["stdout"]).strip() else ""
    if not output_dir:
        raise RuntimeError("Remote msprof output directory command did not return a path.")
    return output_dir


def _read_remote_msprof_metrics(
    spec: RemoteSpec,
    remote_workspace: str,
    output_dir: str,
    kernel_names: list[str],
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> PerfMetrics:
    script = """
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
kernel_names = set(json.loads(sys.argv[2]))
matches = sorted(path for path in root.rglob("op_statistic_*.csv") if path.is_file())
if not matches:
    raise SystemExit(f"No op_statistic_*.csv found under {root}")
csv_path = max(matches, key=lambda path: path.stat().st_mtime_ns)

with csv_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    if "Avg Time(us)" not in fieldnames:
        raise SystemExit(f"Missing required column 'Avg Time(us)' in {csv_path}")
    if "OP Type" not in fieldnames:
        raise SystemExit(f"Missing required column 'OP Type' in {csv_path}")
    ops = []
    row_count = 0
    for row in reader:
        value = (row.get("Avg Time(us)") or "").strip()
        if not value:
            raise SystemExit(f"Empty 'Avg Time(us)' value in {csv_path}")
        op_type = (row.get("OP Type") or "").strip()
        if not op_type:
            raise SystemExit(f"Empty 'OP Type' value in {csv_path}")
        ops.append({"op_type": op_type, "avg_time_us": float(value)})
        row_count += 1

if row_count == 0:
    raise SystemExit(f"No rows found in {csv_path}")
matched = [row["avg_time_us"] for row in ops if row["op_type"] in kernel_names]
kernel_avg_time_us = sum(matched) if matched else None
print(json.dumps({"kernel_avg_time_us": kernel_avg_time_us, "ops": ops}, separators=(",", ":")))
""".strip()
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        ["python3", "-c", script, output_dir, json.dumps(kernel_names)],
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or "Failed to parse remote msprof statistic CSV.")
    value = str(result["stdout"]).strip().splitlines()[-1].strip() if str(result["stdout"]).strip() else ""
    if not value:
        raise RuntimeError(f"Remote msprof statistic parser did not return a value for {output_dir}.")
    parsed = json.loads(value)
    return {
        "kernel_avg_time_us": (
            None if parsed["kernel_avg_time_us"] is None else float(parsed["kernel_avg_time_us"])
        ),
        "ops": [
            {
                "op_type": str(row["op_type"]),
                "avg_time_us": float(row["avg_time_us"]),
            }
            for row in parsed["ops"]
        ],
    }


def _cleanup_remote_msprof_output_dir(
    spec: RemoteSpec,
    remote_workspace: str,
    output_dir: str,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    run_remote_command_buffered(
        spec,
        remote_workspace,
        ["rm", "-rf", output_dir],
        verbose=verbose,
        stderr=stderr,
    )


def _create_local_torch_npu_profiler_case_workspace(
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
) -> tuple[Path, Callable[[], None]]:
    return _create_local_case_workspace(
        prefix=f"triton-agent-torch-npu-profiler-case-{case_id}-",
        input_paths=_bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        flat_input_paths=_bench_flat_input_paths(bench_file),
        source_root=source_root,
        verbose=verbose,
    )


def _build_remote_torch_npu_profiler_run_all_script(*, verbose: bool = False) -> str:
    return (
        "import pathlib, shutil, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        f"result, perf_path = runtime.profile_all_bench_cases(bench_file, operator_file, verbose={verbose}); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_torch_npu_profiler_run_one_case_script(*, verbose: bool = False) -> str:
    return (
        "import json, pathlib, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "preserved_run_dir_arg = sys.argv[4]; "
        f"preserved_run_dir = None if preserved_run_dir_arg == {_PRESERVED_RUN_DIR_NONE_SENTINEL!r} else pathlib.Path(preserved_run_dir_arg); "
        "record = runtime.profile_bench_case("
        "bench_file, operator_file, case_id, preserved_run_dir=preserved_run_dir, "
        f"verbose={verbose}"
        "); "
        "payload = {"
        "'case_label': record.case_label, "
        "'kernel_names': record.kernel_names, "
        "'kernel_source': record.kernel_source, "
        "'metrics': record.metrics, "
        "'error_message': record.error_message, "
        "'case_wall_clock_seconds': record.case_wall_clock_seconds"
        "}; "
        "print(json.dumps(payload, separators=(',', ':')))"
    )


def _run_local_torch_npu_profiler_case_in_subprocess(
    workspace_root: Path,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    device: str,
    *,
    preserved_run_dir: Path | None,
    source_root: Path,
    verbose: bool = False,
) -> PerfCaseRecord:
    extra_env = affinity_env_for_device(device)
    configured_profile_root, _configured_env = _resolve_local_bench_profile_output_root()
    if configured_profile_root:
        extra_env[_LOCAL_BENCH_OUTPUT_DIR_ENV] = str(Path(configured_profile_root).expanduser().resolve())
    extra_env["TRITON_ALWAYS_COMPILE"] = "1"
    command = [
        local_python_executable(),
        "-c",
        _build_torch_npu_profiler_run_one_case_script(verbose=verbose),
        _case_workspace_command_path(bench_file, source_root=source_root),
        _case_workspace_command_path(operator_file, source_root=source_root),
        case_id,
        (
            _PRESERVED_RUN_DIR_NONE_SENTINEL
            if preserved_run_dir is None
            else preserved_run_dir.resolve().as_posix()
        ),
    ]
    if verbose:
        with _stream_target_for_verbosity(True) as stream_target:
            result = run_streaming_process(
                command,
                str(workspace_root),
                stall_timeout_seconds=_bench_timeout(),
                stdout=stream_target,
                extra_env=extra_env,
            )
    else:
        result = run_buffered_process(
            command,
            str(workspace_root),
            stall_timeout_seconds=_bench_timeout(),
            extra_env=extra_env,
        )
    return _parse_torch_npu_profiler_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _stage_remote_torch_npu_profiler_case_workspace(
    spec: RemoteSpec,
    bench_file: Path,
    operator_file: Path,
    case_workspace: str,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> str:
    return _stage_remote_case_workspace(
        spec,
        case_workspace,
        _bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        source_root=source_root,
        flat_input_paths=_bench_flat_input_paths(bench_file),
        verbose=verbose,
        stderr=stderr,
    )


def _run_remote_torch_npu_profiler_case(
    spec: RemoteSpec,
    case_workspace: str,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    device: str,
    *,
    source_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> PerfCaseRecord:
    extra_env = affinity_env_for_device(device)
    extra_env["TRITON_ALWAYS_COMPILE"] = "1"
    result = run_remote_command_streaming(
        spec,
        case_workspace,
        [
            "python3",
            "-c",
            _build_torch_npu_profiler_run_one_case_script(verbose=verbose),
            _case_workspace_command_path(bench_file, source_root=source_root),
            _case_workspace_command_path(operator_file, source_root=source_root),
            case_id,
            _PRESERVED_RUN_DIR_NONE_SENTINEL,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
        stall_timeout_seconds=_bench_timeout(),
    )
    return _parse_torch_npu_profiler_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _parse_torch_npu_profiler_case_result_payload(
    result: ResultPayload,
    *,
    case_id: str,
    fallback_kernel_source: str,
) -> PerfCaseRecord:
    if not result_succeeded(result):
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=_format_torch_npu_profiler_command_failure(result),
            case_wall_clock_seconds=None,
        )
    stdout_text = str(result["stdout"]).strip()
    if not stdout_text:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message="torch-npu-profiler worker produced no JSON payload",
            case_wall_clock_seconds=None,
        )
    try:
        payload = stdout_text.splitlines()[-1].strip()
        parsed = json.loads(payload)
    except (IndexError, json.JSONDecodeError) as exc:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=f"failed to parse torch-npu-profiler worker payload: {exc}",
            case_wall_clock_seconds=None,
        )
    metrics_payload = parsed["metrics"]
    return PerfCaseRecord(
        case_label=str(parsed["case_label"]),
        kernel_names=[str(name) for name in parsed["kernel_names"]],
        kernel_source=str(parsed["kernel_source"]),
        metrics=None if metrics_payload is None else cast(PerfMetrics, metrics_payload),
        error_message=None if parsed["error_message"] is None else str(parsed["error_message"]),
        case_wall_clock_seconds=None
        if parsed["case_wall_clock_seconds"] is None
        else float(parsed["case_wall_clock_seconds"]),
    )


def _format_torch_npu_profiler_command_failure(result: ResultPayload) -> str:
    details = str(result["stderr"]).strip() or str(result["stdout"]).strip()
    prefix = f"torch-npu-profiler command failed with return code {int(result['return_code'])}"
    return f"{prefix}: {details}" if details else prefix


def _write_torch_npu_profiler_perf(
    operator_file: Path,
    case_records: list[PerfCaseRecord],
    output: str | None = None,
) -> Path:
    return write_perf_lines(
        _resolve_perf_output_path(operator_file, output=output),
        render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error="no resolved kernels matched profiler operator details",
        ),
    )


def _resolve_perf_output_path(operator_file: Path, *, output: str | None = None) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return perf_output_path(operator_file)


def _build_torch_npu_profiler_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    errors = [
        f"{record.case_label}: {record.error_message}"
        for record in case_records
        if record.error_message is not None
    ]
    return make_result(
        return_code=1 if errors else 0,
        stdout="",
        stderr="\n".join(errors),
    )


def _run_local_msprof_case_parallel(
    bench_file: Path,
    operator_file: Path,
    operator_arg: str,
    bench_arg: str,
    runtime_arg: str,
    resolution: KernelResolution,
    case_id: str,
    pool: NpuDevicePool,
    preserved_run_dir: Path | None,
    source_root: Path,
    json_search_root: Path,
    verbose: bool,
) -> _MsprofCaseOutcome:
    case_workspace, cleanup = _create_local_msprof_case_workspace(
        bench_file,
        operator_file,
        case_id,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
    )
    output_dir, temp_dir = _create_local_msprof_output_dir(case_id, preserved_run_dir)
    try:
        with pool.acquire() as device:
            extra_env = affinity_env_for_device(device)
            extra_env["TRITON_ALWAYS_COMPILE"] = "1"
            command = [
                "msprof",
                f"--output={output_dir}",
                local_python_executable(),
                runtime_arg,
                "run-one",
                "--bench-file",
                bench_arg,
                "--operator-file",
                operator_arg,
                "--case-id",
                case_id,
            ]
            t0 = time.monotonic()
            with _stream_target_for_verbosity(verbose) as stream_target:
                result = run_streaming_process(
                    command,
                    str(case_workspace),
                    stall_timeout_seconds=_bench_timeout(),
                    stdout=stream_target,
                    extra_env=extra_env,
                )
            elapsed = time.monotonic() - t0
        return _build_local_msprof_case_outcome(result, resolution, case_id, output_dir, elapsed)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        _cleanup_local_bench_extra_info(case_workspace)
        cleanup()


def _run_remote_msprof_case_parallel(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    resolution: KernelResolution,
    case_id: str,
    pool: NpuDevicePool,
    source_root: Path,
    json_search_root: Path,
    verbose: bool,
    stderr: TextIO | None,
) -> _MsprofCaseOutcome:
    case_workspace = f"{remote_workspace}/case-{case_id}"
    run_remote_command_buffered(
        spec,
        remote_workspace,
        ["mkdir", "-p", case_workspace],
        verbose=verbose,
        stderr=stderr,
    )
    workspace_root = _stage_remote_msprof_case_workspace(
        spec,
        bench_file,
        operator_file,
        case_workspace,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
        stderr=stderr,
    )
    bench_arg = _case_workspace_command_path(bench_file, source_root=source_root)
    operator_arg = _case_workspace_command_path(operator_file, source_root=source_root)
    output_dir = f"{workspace_root}/msprof-output"
    try:
        with pool.acquire() as device:
            extra_env = affinity_env_for_device(device)
            extra_env["TRITON_ALWAYS_COMPILE"] = "1"
            t0 = time.monotonic()
            result = run_remote_command_streaming(
                spec,
                workspace_root,
                [
                    "msprof",
                    f"--output={output_dir}",
                    "python3",
                    _bench_runtime_script_path().name,
                    "run-one",
                    "--bench-file",
                    bench_arg,
                    "--operator-file",
                    operator_arg,
                    "--case-id",
                    case_id,
                ],
                verbose=verbose,
                stderr=stderr,
                extra_env=extra_env,
                stall_timeout_seconds=_bench_timeout(),
            )
            elapsed = time.monotonic() - t0
        return _build_remote_msprof_case_outcome(
            spec,
            workspace_root,
            result,
            resolution,
            case_id,
            output_dir,
            elapsed,
            verbose=verbose,
            stderr=stderr,
        )
    finally:
        run_remote_command_buffered(
            spec,
            remote_workspace,
            ["rm", "-rf", case_workspace],
            verbose=verbose,
            stderr=stderr,
        )


def _build_local_msprof_case_outcome(
    result: ResultPayload,
    resolution: KernelResolution,
    case_id: str,
    output_dir: Path,
    elapsed: float,
) -> _MsprofCaseOutcome:
    if not result_succeeded(result):
        record = PerfCaseRecord(
            case_label=case_id,
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=_format_msprof_command_failure(result),
            case_wall_clock_seconds=elapsed,
        )
    else:
        try:
            metrics = _read_local_msprof_metrics(output_dir, resolution.kernel_names)
            record = PerfCaseRecord(
                case_label=case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                metrics=metrics,
                case_wall_clock_seconds=elapsed,
            )
        except (FileNotFoundError, ValueError) as exc:
            record = PerfCaseRecord(
                case_label=case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=str(exc),
                case_wall_clock_seconds=elapsed,
            )
    return _MsprofCaseOutcome(
        case_id=case_id,
        record=record,
        stdout=str(result["stdout"]),
        stderr=str(result["stderr"]),
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )


def _build_remote_msprof_case_outcome(
    spec: RemoteSpec,
    remote_workspace: str,
    result: ResultPayload,
    resolution: KernelResolution,
    case_id: str,
    output_dir: str,
    elapsed: float,
    *,
    verbose: bool,
    stderr: TextIO | None,
) -> _MsprofCaseOutcome:
    if not result_succeeded(result):
        record = PerfCaseRecord(
            case_label=case_id,
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=_format_msprof_command_failure(result),
            case_wall_clock_seconds=elapsed,
        )
    else:
        try:
            metrics = _read_remote_msprof_metrics(
                spec,
                remote_workspace,
                output_dir,
                resolution.kernel_names,
                verbose=verbose,
                stderr=stderr,
            )
            record = PerfCaseRecord(
                case_label=case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                metrics=metrics,
                case_wall_clock_seconds=elapsed,
            )
        except RuntimeError as exc:
            record = PerfCaseRecord(
                case_label=case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=str(exc),
                case_wall_clock_seconds=elapsed,
            )
    return _MsprofCaseOutcome(
        case_id=case_id,
        record=record,
        stdout=str(result["stdout"]),
        stderr=str(result["stderr"]),
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )


def _build_msprof_result(
    stdout_chunks: list[str],
    stderr_chunks: list[str],
    outcomes: list[_MsprofCaseOutcome],
) -> ResultPayload:
    return make_result(
        return_code=1 if any(outcome.record.error_message is not None for outcome in outcomes) else 0,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        stalled=any(outcome.stalled for outcome in outcomes),
        session_id=next((outcome.session_id for outcome in outcomes if outcome.session_id is not None), None),
    )


def _write_msprof_perf(
    operator_file: Path,
    case_records: list[PerfCaseRecord],
    output: str | None = None,
) -> Path:
    return write_perf_lines(
        _resolve_msprof_output_path(operator_file, output=output),
        render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
        ),
    )


def _resolve_msprof_output_path(
    operator_file: Path,
    *,
    output: str | None = None,
) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return perf_output_path(operator_file)


def _load_msprof_avg_rows(output_dir: Path) -> list[PerfOpRow]:
    csv_path = find_latest_op_statistic_csv(output_dir)
    if csv_path is None:
        raise FileNotFoundError(f"No op_statistic_*.csv found under {output_dir}")
    return parse_op_statistic_csv(csv_path).ops


def _resolve_msprof_metrics(
    rows: list[PerfOpRow],
    kernel_names: list[str],
) -> PerfMetrics:
    return resolve_perf_metrics(rows, kernel_names)


def _read_local_msprof_metrics(output_dir: Path, kernel_names: list[str]) -> PerfMetrics:
    return _resolve_msprof_metrics(_load_msprof_avg_rows(output_dir), kernel_names)

