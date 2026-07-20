from __future__ import annotations

import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, TextIO, TypeVar, cast

from bench_contract import (
    KernelResolution,
    resolve_bench_kernel_resolution,
)
from env_registry import (
    ASCEND_RT_VISIBLE_DEVICES,
    HELIX_BENCH_COPY_FILES,
    HELIX_BENCH_OUTPUT_DIR,
    TRITON_ALWAYS_COMPILE,
)
from npu_affinity import NpuDevicePool, affinity_env_for_device, parse_npu_devices
from debug_device import maybe_print_visible_devices
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    PerfOpRow,
    perf_output_path,
    render_perf_case_records_jsonl,
    write_perf_lines,
)
from profile_csv_parser import (
    find_latest_op_statistic_csv,
    parse_op_statistic_csv,
    resolve_perf_metrics,
)
from result_payload import ResultPayload, make_result
from remote_python_bundle import resolve_remote_python_bundle
import run_bench_execution
from run_runtime import (
    RemoteSpec,
    copy_file_from_remote,
    copy_file_to_remote,
    eval_timeout_seconds,
    emit_verbose,
    local_python_executable,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)


NpuDevices = tuple[str, ...]
ExecutionLimits = tuple[int, int]
BenchRunResult = tuple[ResultPayload, Optional[Path]]
BenchRunResultWithPerfPath = tuple[ResultPayload, Path]
RemoteBenchRunResult = tuple[ResultPayload, Optional[Path], str]
RemoteBenchRunResultWithPerfPath = tuple[ResultPayload, Path, str]
ResolvedProfileOutputRoot = tuple[Optional[str], str]
PreservedRunDir = tuple[Path, Optional[tempfile.TemporaryDirectory[str]]]
CaseWorkspaceRoots = tuple[Path, Path]
CaseWorkspace = tuple[Path, Callable[[], None]]
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


@dataclass(frozen=True)
class _RemoteCasePlan:
    case_ids: list[str]
    iterations_by_case: dict[str, int]
    resolution: KernelResolution


def _collect_env_copy_files(search_dir: Path) -> list[Path]:
    patterns_str = os.environ.get(HELIX_BENCH_COPY_FILES, "")
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


def normalize_bench_mode(bench_mode: str) -> str:
    return "torch-npu-profiler" if bench_mode == "standalone" else bench_mode


def execute_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
    execution_limits: ExecutionLimits | None = None,
 ) -> BenchRunResult:
    bench_mode = normalize_bench_mode(bench_mode)
    invocation_root = Path.cwd().resolve()
    devices = parse_npu_devices(npu_devices)
    maybe_print_visible_devices()
    with _local_bench_workdir(bench_file.parent):
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
        if bench_mode == "perf-counter":
            if devices is not None:
                source_root, json_search_root = _resolve_case_workspace_roots(
                    bench_file,
                    operator_file,
                    invocation_root=invocation_root,
                )
                return _run_local_bench_perf_counter_parallel(
                    bench_file,
                    operator_file,
                    devices,
                    source_root=source_root,
                    json_search_root=json_search_root,
                    verbose=verbose,
                    output=output,
                )
            return _run_local_bench_perf_counter(bench_file, operator_file, verbose=verbose,
                                                 output=output)
        if devices is not None and execution_limits is None:
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
        return _run_local_bench_torch_npu_profiler(
            bench_file,
            operator_file,
            verbose=verbose,
            output=output,
            execution_limits=execution_limits,
        )


def execute_remote_bench_workspace(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    spec: RemoteSpec,
    remote_workspace: str,
    npu_devices: str | None = None,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
    execution_limits: ExecutionLimits | None = None,
) -> RemoteBenchRunResult:
    bench_mode = normalize_bench_mode(bench_mode)
    invocation_root = Path.cwd().resolve()
    devices = parse_npu_devices(npu_devices)
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
    if bench_mode == "perf-counter":
        if devices is not None:
            source_root, json_search_root = _resolve_case_workspace_roots(
                bench_file,
                operator_file,
                invocation_root=invocation_root,
            )
            return _run_remote_bench_perf_counter_parallel(
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
        return _run_remote_bench_perf_counter(
            spec,
            remote_workspace,
            bench_file,
            operator_file,
            verbose=verbose,
            stderr=stderr,
            output=output,
        )
    if devices is not None and execution_limits is None:
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
        execution_limits=execution_limits,
        devices=devices,
    )


def _run_local_bench_torch_npu_profiler(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
    execution_limits: ExecutionLimits | None = None,
) -> BenchRunResult:
    if execution_limits is not None:
        warmup_cap, repeats_cap = execution_limits
        cases, resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
        preloaded = (
            [
                replace(
                    case,
                    warmup=min(case.warmup, warmup_cap),
                    repeats=min(case.repeats, repeats_cap),
                )
                for case in cases
            ],
            resolution,
        )
        return run_bench_execution.profile_all_bench_cases(
            bench_file,
            operator_file,
            preloaded=preloaded,
            verbose=verbose,
            output=output,
        )
    return run_bench_execution.profile_all_bench_cases(
        bench_file,
        operator_file,
        verbose=verbose,
        output=output,
    )


def _run_local_bench_perf_counter(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    return run_bench_execution.time_all_bench_cases(
        bench_file,
        operator_file,
        bench_mode="perf-counter",
        output=output,
    )


def _run_local_bench_perf_counter_parallel(
    bench_file: Path,
    operator_file: Path,
    devices: NpuDevices,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    cases, _resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)

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
                return _run_local_perf_counter_case_in_subprocess(
                    case_workspace,
                    bench_file,
                    operator_file,
                    case_id,
                    device,
                    source_root=source_root,
                    verbose=verbose,
                )
        finally:
            cleanup()

    case_records = _run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    _sort_case_records(case_records, case_ids)
    perf_path = _write_perf_counter_perf(operator_file, case_records, output=output)
    return _build_perf_counter_result(case_records), perf_path


def _write_perf_counter_perf(
    operator_file: Path,
    case_records: list[PerfCaseRecord],
    *,
    output: str | None = None,
) -> Path:
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    write_perf_lines(perf_path, render_perf_case_records_jsonl(case_records))
    return perf_path


def _build_perf_counter_result(case_records: list[PerfCaseRecord]) -> ResultPayload:
    had_failures = any(record.error_message is not None for record in case_records)
    stderr = "\n".join(
        str(record.error_message) for record in case_records if record.error_message is not None
    )
    return make_result(return_code=1 if had_failures else 0, stdout="", stderr=stderr)


def _run_remote_bench_perf_counter(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> RemoteBenchRunResult:
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    extra_env: dict[str, str] | None = {TRITON_ALWAYS_COMPILE: "1"}
    with stream_target_for_verbosity(verbose) as stream_target:
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                _remote_worker_path().name,
                "perf-counter-all",
                "--bench-file",
                bench_file.name,
                "--operator-file",
                operator_file.name,
                "--output",
                perf_path.name,
            ],
            stdout=stream_target,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=eval_timeout_seconds(),
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


def _run_remote_bench_perf_counter_parallel(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    devices: NpuDevices,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> RemoteBenchRunResult:
    case_plan = _load_remote_case_plan(
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
    )
    case_ids = case_plan.case_ids
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
        with pool.acquire() as device:
            return _run_remote_perf_counter_case(
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

    case_records = _run_parallel_case_workers(case_ids, min(len(case_ids), len(devices)), _worker)
    _sort_case_records(case_records, case_ids)
    perf_path = _write_perf_counter_perf(operator_file, case_records, output=output)
    return _build_perf_counter_result(case_records), perf_path, remote_workspace


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
def stream_target_for_verbosity(verbose: bool) -> Iterator[TextIO]:
    if verbose:
        yield sys.stdout
        return
    with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
        yield quiet_stdout


def _load_remote_case_plan(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> _RemoteCasePlan:
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        [
            "python3",
            _remote_worker_path().name,
            "case-plan",
            "--bench-file",
            bench_file.name,
            "--operator-file",
            operator_file.name,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env={TRITON_ALWAYS_COMPILE: "1"},
        stall_timeout_seconds=eval_timeout_seconds(),
    )
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or "Failed to load remote benchmark cases.")
    payload_text = str(result["stdout"])
    payload_line = next(
        (line for line in reversed(payload_text.splitlines()) if line.lstrip().startswith("{")),
        "",
    )
    try:
        payload = cast(dict[str, object], json.loads(payload_line))
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Remote benchmark case-plan worker returned an invalid payload.") from exc
    raw_case_ids = payload.get("case_ids")
    raw_iterations = payload.get("iterations_by_case")
    raw_kernel_names = payload.get("kernel_names")
    kernel_source = payload.get("kernel_source")
    if not isinstance(raw_case_ids, list) or not raw_case_ids or not isinstance(raw_iterations, dict):
        raise RuntimeError("Remote benchmark case-plan worker returned an invalid payload.")
    case_ids = cast(list[str], raw_case_ids)
    if not all(case_id for case_id in case_ids):
        raise RuntimeError("Remote benchmark case-plan worker returned an invalid payload.")
    if not isinstance(raw_kernel_names, list) or not isinstance(kernel_source, str):
        raise RuntimeError("Remote benchmark case-plan worker returned an invalid payload.")
    kernel_names = cast(list[str], raw_kernel_names)
    if not all(name for name in kernel_names):
        raise RuntimeError("Remote benchmark case-plan worker returned an invalid payload.")
    raw_iterations_by_case = cast(dict[str, object], raw_iterations)
    iterations_by_case: dict[str, int] = {}
    for case_id in case_ids:
        iterations = raw_iterations_by_case.get(case_id)
        if not isinstance(iterations, int) or iterations <= 0:
            raise RuntimeError("Remote benchmark case-plan worker returned invalid case iterations.")
        iterations_by_case[case_id] = iterations
    return _RemoteCasePlan(
        case_ids=case_ids,
        iterations_by_case=iterations_by_case,
        resolution=KernelResolution(kernel_names=kernel_names, kernel_source=kernel_source),
    )


def stage_remote_bench_input_files(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> None:
    primary_paths = {bench_file.resolve(), operator_file.resolve()}
    for support_path in _collect_env_copy_files(bench_file.parent):
        if support_path.resolve() in primary_paths:
            continue
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
    execution_limits: ExecutionLimits | None = None,
    devices: NpuDevices | None = None,
) -> RemoteBenchRunResult:
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    command = [
        "python3",
        _remote_worker_path().name,
        "profile-all",
        "--bench-file",
        bench_file.name,
        "--operator-file",
        operator_file.name,
        "--output",
        perf_path.name,
    ]
    if execution_limits is not None:
        command.extend(["--warmup-cap", str(execution_limits[0]), "--repeats-cap", str(execution_limits[1])])
    if verbose:
        command.append("--verbose")
    extra_env: dict[str, str] = {TRITON_ALWAYS_COMPILE: "1"}
    if devices is not None:
        extra_env[ASCEND_RT_VISIBLE_DEVICES] = ",".join(devices)
    with stream_target_for_verbosity(verbose) as stream_target:
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            command,
            stdout=stream_target,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=eval_timeout_seconds(),
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
    devices: NpuDevices,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResultWithPerfPath:
    cases, _resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)
    preserved_run_dir: Path | None = None
    create_preserved_run_dir = getattr(
        run_bench_execution,
        "create_local_preserved_profile_run_dir",
        None,
    )
    if callable(create_preserved_run_dir):
        preserved_run_dir = cast(
            Path | None,
            create_preserved_run_dir(prefix="helix-torch-npu-profiler-bench-"),
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
    devices: NpuDevices,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> RemoteBenchRunResultWithPerfPath:
    case_plan = _load_remote_case_plan(
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
    )
    case_ids = case_plan.case_ids
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


def _run_bench_execution_script_path() -> Path:
    return Path(__file__).resolve().with_name("run_bench_execution.py")


def _remote_worker_path() -> Path:
    return Path(__file__).resolve().with_name("run_bench_remote_worker.py")


def _local_worker_path() -> Path:
    return Path(__file__).resolve().with_name("run_bench_local_worker.py")


def _bench_local_flat_input_paths(bench_file: Path) -> list[Path]:
    return [
        *resolve_remote_python_bundle([_local_worker_path()]),
        *_collect_env_copy_files(bench_file.parent),
    ]


def _bench_remote_flat_input_paths(bench_file: Path) -> list[Path]:
    return [
        *resolve_remote_python_bundle([_remote_worker_path()]),
        *_collect_env_copy_files(bench_file.parent),
    ]


def _run_local_bench_msprof(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    cases, _ignored_resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
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
                str(_run_bench_execution_script_path()),
                "run-one",
                "--bench-file",
                bench_file.name,
                "--operator-file",
                operator_arg,
                "--case-id",
                case.case_id,
                "--iterations",
                str(case.warmup + case.repeats),
            ]
            t0 = time.monotonic()
            with stream_target_for_verbosity(verbose) as stream_target:
                result = run_streaming_process(
                    command,
                    str(bench_file.parent),
                    stall_timeout_seconds=eval_timeout_seconds(),
                    stdout=stream_target,
                    extra_env={TRITON_ALWAYS_COMPILE: "1"},
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
                        bench_mode="msprof",
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
                        bench_mode="msprof",
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
                    bench_mode="msprof",
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
    devices: NpuDevices,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    cases, _ignored_resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    bench_arg = _case_workspace_command_path(bench_file, source_root=source_root)
    operator_arg = _case_workspace_command_path(operator_file, source_root=source_root)
    runtime_arg = _run_bench_execution_script_path().name
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()
    case_ids = [case.case_id for case in cases]
    iterations_by_case = {case.case_id: case.warmup + case.repeats for case in cases}
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
            iterations_by_case[case_id],
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
) -> RemoteBenchRunResult:
    case_plan = _load_remote_case_plan(
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
    )
    resolution = case_plan.resolution
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    case_records: list[PerfCaseRecord] = []
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    for case_id in case_plan.case_ids:
        output_dir = _create_remote_msprof_output_dir(
            spec,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            t0 = time.monotonic()
            with stream_target_for_verbosity(verbose) as stream_target:
                result = run_remote_command_streaming(
                    spec,
                    remote_workspace,
                    [
                        "msprof",
                        f"--output={output_dir}",
                        "python3",
                        _run_bench_execution_script_path().name,
                        "run-one",
                        "--bench-file",
                        bench_file.name,
                        "--operator-file",
                        operator_file.name,
                        "--case-id",
                        case_id,
                        "--iterations",
                        str(case_plan.iterations_by_case[case_id]),
                    ],
                    stdout=stream_target,
                    verbose=verbose,
                    stderr=stderr,
                    stall_timeout_seconds=eval_timeout_seconds(),
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
                        case_label=case_id,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=_format_msprof_command_failure(result),
                        case_wall_clock_seconds=elapsed,
                        bench_mode="msprof",
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
                        case_label=case_id,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                        case_wall_clock_seconds=elapsed,
                        bench_mode="msprof",
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=case_id,
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                    case_wall_clock_seconds=elapsed,
                    bench_mode="msprof",
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
    devices: NpuDevices,
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> RemoteBenchRunResult:
    case_plan = _load_remote_case_plan(
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
    )
    resolution = case_plan.resolution
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    case_ids = case_plan.case_ids
    iterations_by_case = case_plan.iterations_by_case
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
            iterations_by_case[case_id],
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


def _resolve_local_bench_profile_output_root() -> ResolvedProfileOutputRoot:
    configured_root = os.environ.get(HELIX_BENCH_OUTPUT_DIR)
    if configured_root:
        return str(Path(configured_root).expanduser().resolve()), HELIX_BENCH_OUTPUT_DIR
    return None, HELIX_BENCH_OUTPUT_DIR


def _create_local_msprof_output_dir(
    case_label: str,
    preserved_run_dir: Path | None,
) -> PreservedRunDir:
    if preserved_run_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="helix-msprof-")
        return Path(temp_dir.name), temp_dir
    output_dir = preserved_run_dir.resolve() / f"case-{case_label}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _set_directory_owner_only(output_dir)
    return output_dir, None


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
) -> CaseWorkspaceRoots:
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
) -> CaseWorkspace:
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
) -> CaseWorkspace:
    return _create_local_case_workspace(
        prefix=f"helix-msprof-case-{case_id}-",
        input_paths=_bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        flat_input_paths=_bench_remote_flat_input_paths(bench_file),
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
        flat_input_paths=_bench_local_flat_input_paths(bench_file),
        verbose=verbose,
        stderr=stderr,
    )


def _set_directory_owner_only(path: Path) -> None:
    path.chmod(0o700)


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
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        [
            "python3",
            _remote_worker_path().name,
            "msprof-metrics",
            "--metrics-root",
            output_dir,
            "--kernel-names",
            json.dumps(kernel_names),
        ],
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(result):
        raise RuntimeError(result["stderr"] or result["stdout"] or "Failed to parse remote msprof statistic CSV.")
    value = str(result["stdout"]).strip().splitlines()[-1].strip() if str(result["stdout"]).strip() else ""
    if not value:
        raise RuntimeError(f"Remote msprof statistic parser did not return a value for {output_dir}.")
    parsed = json.loads(value)
    metrics: PerfMetrics = {
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
    total_op_avg_time_us_raw = parsed.get("total_op_avg_time_us")
    if total_op_avg_time_us_raw is not None:
        metrics["total_op_avg_time_us"] = float(total_op_avg_time_us_raw)
    return metrics


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
) -> CaseWorkspace:
    return _create_local_case_workspace(
        prefix=f"helix-torch-npu-profiler-case-{case_id}-",
        input_paths=_bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        flat_input_paths=_bench_local_flat_input_paths(bench_file),
        source_root=source_root,
        verbose=verbose,
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
        extra_env[HELIX_BENCH_OUTPUT_DIR] = str(
            Path(configured_profile_root).expanduser().resolve()
        )
    extra_env[TRITON_ALWAYS_COMPILE] = "1"
    command = [
        local_python_executable(),
        str(_local_worker_path()),
        "profile-case",
        "--bench-file",
        _case_workspace_command_path(bench_file, source_root=source_root),
        "--operator-file",
        _case_workspace_command_path(operator_file, source_root=source_root),
        "--case-id",
        case_id,
        "--preserved-run-dir",
        (
            _PRESERVED_RUN_DIR_NONE_SENTINEL
            if preserved_run_dir is None
            else preserved_run_dir.resolve().as_posix()
        ),
    ]
    if verbose:
        command.append("--verbose")
    if verbose:
        with stream_target_for_verbosity(True) as stream_target:
            result = run_streaming_process(
                command,
                str(workspace_root),
                stall_timeout_seconds=eval_timeout_seconds(),
                stdout=stream_target,
                extra_env=extra_env,
            )
    else:
        result = run_buffered_process(
            command,
            str(workspace_root),
            stall_timeout_seconds=eval_timeout_seconds(),
            extra_env=extra_env,
        )
    return _parse_torch_npu_profiler_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _run_local_perf_counter_case_in_subprocess(
    workspace_root: Path,
    bench_file: Path,
    operator_file: Path,
    case_id: str,
    device: str,
    *,
    source_root: Path,
    verbose: bool = False,
) -> PerfCaseRecord:
    extra_env = affinity_env_for_device(device)
    extra_env[TRITON_ALWAYS_COMPILE] = "1"
    command = [
        local_python_executable(),
        str(_local_worker_path()),
        "perf-counter-case",
        "--bench-file",
        _case_workspace_command_path(bench_file, source_root=source_root),
        "--operator-file",
        _case_workspace_command_path(operator_file, source_root=source_root),
        "--case-id",
        case_id,
    ]
    if verbose:
        with stream_target_for_verbosity(True) as stream_target:
            result = run_streaming_process(
                command,
                str(workspace_root),
                stall_timeout_seconds=eval_timeout_seconds(),
                stdout=stream_target,
                extra_env=extra_env,
            )
    else:
        result = run_buffered_process(
            command,
            str(workspace_root),
            stall_timeout_seconds=eval_timeout_seconds(),
            extra_env=extra_env,
        )
    return _parse_worker_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
        bench_mode="perf-counter",
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
        flat_input_paths=_bench_remote_flat_input_paths(bench_file),
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
    extra_env[TRITON_ALWAYS_COMPILE] = "1"
    result = run_remote_command_streaming(
        spec,
        case_workspace,
        [
            "python3",
            _remote_worker_path().name,
            "profile-case",
            "--bench-file",
            _case_workspace_command_path(bench_file, source_root=source_root),
            "--operator-file",
            _case_workspace_command_path(operator_file, source_root=source_root),
            "--case-id",
            case_id,
            "--preserved-run-dir",
            _PRESERVED_RUN_DIR_NONE_SENTINEL,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
        stall_timeout_seconds=eval_timeout_seconds(),
    )
    return _parse_torch_npu_profiler_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
    )


def _run_remote_perf_counter_case(
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
    extra_env[TRITON_ALWAYS_COMPILE] = "1"
    result = run_remote_command_streaming(
        spec,
        case_workspace,
        [
            "python3",
            _remote_worker_path().name,
            "perf-counter-case",
            "--bench-file",
            _case_workspace_command_path(bench_file, source_root=source_root),
            "--operator-file",
            _case_workspace_command_path(operator_file, source_root=source_root),
            "--case-id",
            case_id,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
        stall_timeout_seconds=eval_timeout_seconds(),
    )
    return _parse_worker_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source="metadata",
        bench_mode="perf-counter",
    )


def _parse_torch_npu_profiler_case_result_payload(
    result: ResultPayload,
    *,
    case_id: str,
    fallback_kernel_source: str,
) -> PerfCaseRecord:
    return _parse_worker_case_result_payload(
        result,
        case_id=case_id,
        fallback_kernel_source=fallback_kernel_source,
        bench_mode="torch-npu-profiler",
    )


def _parse_worker_case_result_payload(
    result: ResultPayload,
    *,
    case_id: str,
    fallback_kernel_source: str,
    bench_mode: str,
) -> PerfCaseRecord:
    if not result_succeeded(result):
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=_format_worker_command_failure(result, bench_mode),
            case_wall_clock_seconds=None,
            bench_mode=bench_mode,
        )
    stdout_text = str(result["stdout"]).strip()
    if not stdout_text:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=f"{bench_mode} worker produced no JSON payload",
            case_wall_clock_seconds=None,
            bench_mode=bench_mode,
        )
    try:
        payload = stdout_text.splitlines()[-1].strip()
        parsed = json.loads(payload)
    except (IndexError, json.JSONDecodeError) as exc:
        return PerfCaseRecord(
            case_label=case_id,
            kernel_names=[],
            kernel_source=fallback_kernel_source,
            error_message=f"failed to parse {bench_mode} worker payload: {exc}",
            case_wall_clock_seconds=None,
            bench_mode=bench_mode,
        )
    metrics_payload = parsed["metrics"]
    parsed_bench_mode = parsed.get("bench_mode")
    return PerfCaseRecord(
        case_label=str(parsed["case_label"]),
        kernel_names=[str(name) for name in parsed["kernel_names"]],
        kernel_source=str(parsed["kernel_source"]),
        metrics=None if metrics_payload is None else cast(PerfMetrics, metrics_payload),
        error_message=None if parsed["error_message"] is None else str(parsed["error_message"]),
        case_wall_clock_seconds=None
        if parsed["case_wall_clock_seconds"] is None
        else float(parsed["case_wall_clock_seconds"]),
        bench_mode=None if parsed_bench_mode is None else str(parsed_bench_mode),
    )


def _format_worker_command_failure(result: ResultPayload, bench_mode: str) -> str:
    details = str(result["stderr"]).strip() or str(result["stdout"]).strip()
    prefix = f"{bench_mode} command failed with return code {int(result['return_code'])}"
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
            missing_kernel_match_error="no resolved kernels matched profiler kernel view",
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
    iterations: int,
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
            extra_env[TRITON_ALWAYS_COMPILE] = "1"
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
                "--iterations",
                str(iterations),
            ]
            t0 = time.monotonic()
            with stream_target_for_verbosity(verbose) as stream_target:
                result = run_streaming_process(
                    command,
                    str(case_workspace),
                    stall_timeout_seconds=eval_timeout_seconds(),
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
    iterations: int,
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
            extra_env[TRITON_ALWAYS_COMPILE] = "1"
            t0 = time.monotonic()
            result = run_remote_command_streaming(
                spec,
                workspace_root,
                [
                    "msprof",
                    f"--output={output_dir}",
                    "python3",
                    _run_bench_execution_script_path().name,
                    "run-one",
                    "--bench-file",
                    bench_arg,
                    "--operator-file",
                    operator_arg,
                    "--case-id",
                    case_id,
                    "--iterations",
                    str(iterations),
                ],
                verbose=verbose,
                stderr=stderr,
                extra_env=extra_env,
                stall_timeout_seconds=eval_timeout_seconds(),
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
            bench_mode="msprof",
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
                bench_mode="msprof",
            )
        except (FileNotFoundError, ValueError) as exc:
            record = PerfCaseRecord(
                case_label=case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=str(exc),
                case_wall_clock_seconds=elapsed,
                bench_mode="msprof",
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
            bench_mode="msprof",
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
                bench_mode="msprof",
            )
        except RuntimeError as exc:
            record = PerfCaseRecord(
                case_label=case_id,
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=str(exc),
                case_wall_clock_seconds=elapsed,
                bench_mode="msprof",
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
    run_dir = Path(tempfile.mkdtemp(prefix="helix-msprof-", dir=str(root)))
    _set_directory_owner_only(run_dir)
    return run_dir
