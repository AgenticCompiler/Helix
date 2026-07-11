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
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO, TypeVar, cast

from bench_contract import (  # noqa: F401
    KernelResolution,
    parse_bench_metadata,
    resolve_bench_kernel_names,
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
from run_runtime import (
    RemoteSpec,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    eval_stall_timeout_seconds,
    create_remote_workspace,
    emit_verbose,
    local_python_executable,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)


NpuDevices = tuple[str, ...]
ProbeCaps = tuple[int, int]
BenchRunResult = tuple[ResultPayload, Path | None]
BenchRunResultWithPerfPath = tuple[ResultPayload, Path]
RemoteBenchRunResult = tuple[ResultPayload, Path | None, str]
RemoteBenchRunResultWithPerfPath = tuple[ResultPayload, Path, str]
ResolvedProfileOutputRoot = tuple[str | None, str]
PreservedRunDir = tuple[Path, tempfile.TemporaryDirectory[str] | None]
CaseWorkspaceRoots = tuple[Path, Path]
CaseWorkspace = tuple[Path, Callable[[], None]]
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


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
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
    probe_caps: ProbeCaps | None = None,
) -> RemoteBenchRunResult:
    bench_mode = normalize_bench_mode(bench_mode)
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
        if devices is not None and probe_caps is None:
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
            probe_caps=probe_caps,
            devices=devices,
        )
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def run_local_probe(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    *,
    warmup_cap: int,
    repeats_cap: int,
    npu_devices: str | None = None,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    bench_mode = normalize_bench_mode(bench_mode)
    if bench_mode != "torch-npu-profiler":
        return run_local_bench(
            bench_file,
            operator_file,
            bench_mode,
            npu_devices=npu_devices,
            verbose=verbose,
            output=output,
        )
    runtime = _load_bench_runtime_module()
    cases, resolution = runtime.load_bench_cases(bench_file, operator_file)
    clamped = [
        replace(
            case,
            warmup=min(case.warmup, warmup_cap),
            repeats=min(case.repeats, repeats_cap),
        )
        for case in cases
    ]
    return runtime.profile_all_bench_cases(
        bench_file,
        operator_file,
        preloaded=(clamped, resolution),
        verbose=verbose,
        output=output,
    )


def run_remote_probe(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    warmup_cap: int,
    repeats_cap: int,
    npu_devices: str | None = None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
    output: str | None = None,
) -> RemoteBenchRunResult:
    return run_remote_bench(
        bench_file,
        operator_file,
        bench_mode,
        remote,
        remote_workdir,
        npu_devices=npu_devices,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
        output=output,
        probe_caps=(warmup_cap, repeats_cap),
    )


def _run_local_bench_torch_npu_profiler(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
    runtime = _load_bench_runtime_module()
    return runtime.profile_all_bench_cases(
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
    runtime = _load_bench_runtime_module()
    return runtime.time_all_bench_cases(
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
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
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
    _stage_remote_bench_runtime_support_files(
        spec,
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    extra_env: dict[str, str] | None = {TRITON_ALWAYS_COMPILE: "1"}
    with stream_target_for_verbosity(verbose) as stream_target:
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                _build_remote_perf_counter_run_all_script(verbose=verbose),
                bench_file.name,
                operator_file.name,
                perf_path.name,
            ],
            stdout=stream_target,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=eval_stall_timeout_seconds(),
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
    probe_caps: ProbeCaps | None = None,
    devices: NpuDevices | None = None,
) -> RemoteBenchRunResult:
    _stage_remote_bench_runtime_support_files(
        spec,
        remote_workspace,
        verbose=verbose,
        stderr=stderr,
    )
    perf_path = _resolve_perf_output_path(operator_file, output=output)
    if probe_caps is not None:
        script = _build_remote_torch_npu_profiler_probe_run_all_script(
            verbose=verbose, warmup_cap=probe_caps[0], repeats_cap=probe_caps[1]
        )
    else:
        script = _build_remote_torch_npu_profiler_run_all_script(verbose=verbose)
    extra_env: dict[str, str] = {TRITON_ALWAYS_COMPILE: "1"}
    if devices is not None:
        extra_env[ASCEND_RT_VISIBLE_DEVICES] = ",".join(devices)
    with stream_target_for_verbosity(verbose) as stream_target:
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                script,
                bench_file.name,
                operator_file.name,
                perf_path.name,
            ],
            stdout=stream_target,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=eval_stall_timeout_seconds(),
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
    runtime = _load_bench_runtime_module()
    cases, _resolution = runtime.load_bench_cases(bench_file, operator_file)
    case_ids = [case.case_id for case in cases]
    pool = NpuDevicePool(devices)
    preserved_run_dir: Path | None = None
    create_preserved_run_dir = getattr(runtime, "create_local_preserved_profile_run_dir", None)
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
        module_name = f"helix_bench_runtime_{script_path.stem}"
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
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        finally:
            if added:
                sys.path.remove(script_dir)
        _bench_runtime_module_cache = module
        return module


def _run_local_bench_msprof(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    output: str | None = None,
) -> BenchRunResult:
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
                "--iterations",
                str(case.warmup + case.repeats),
            ]
            t0 = time.monotonic()
            with stream_target_for_verbosity(verbose) as stream_target:
                result = run_streaming_process(
                    command,
                    str(bench_file.parent),
                    stall_timeout_seconds=eval_stall_timeout_seconds(),
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
            with stream_target_for_verbosity(verbose) as stream_target:
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
                        "--iterations",
                        str(case.warmup + case.repeats),
                    ],
                    stdout=stream_target,
                    verbose=verbose,
                    stderr=stderr,
                    stall_timeout_seconds=eval_stall_timeout_seconds(),
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
    runtime = _load_bench_runtime_module()
    cases, _ignored_resolution = runtime.load_bench_cases(bench_file, operator_file)
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    case_ids = [case.case_id for case in cases]
    iterations_by_case = {case.case_id: case.warmup + case.repeats for case in cases}
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
    script = _build_remote_msprof_metrics_script()
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


def _build_remote_msprof_metrics_script() -> str:
    return """
import json
import pathlib
import sys

from profile_csv_parser import (
    find_latest_op_statistic_csv,
    parse_op_statistic_csv,
    resolve_perf_metrics,
)

root = pathlib.Path(sys.argv[1])
kernel_names = json.loads(sys.argv[2])
csv_path = find_latest_op_statistic_csv(root)
if csv_path is None:
    raise SystemExit(f"No op_statistic_*.csv or op_statistic.csv found under {root}")
rows = parse_op_statistic_csv(csv_path)
metrics = resolve_perf_metrics(
    rows.ops,
    kernel_names,
    total_op_avg_time_us=rows.total_op_avg_time_us,
)
print(json.dumps(metrics, separators=(",", ":")))
""".strip()


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


def _build_remote_torch_npu_profiler_probe_run_all_script(
    *, verbose: bool = False, warmup_cap: int, repeats_cap: int
) -> str:
    return (
        "import dataclasses, pathlib, shutil, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        "cases, resolution = runtime.load_bench_cases(bench_file, operator_file); "
        f"clamped = [dataclasses.replace(c, warmup=min(c.warmup, {warmup_cap}), repeats=min(c.repeats, {repeats_cap})) for c in cases]; "
        f"result, perf_path = runtime.profile_all_bench_cases(bench_file, operator_file, preloaded=(clamped, resolution), verbose={verbose}); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_remote_perf_counter_run_all_script(*, verbose: bool = False) -> str:
    del verbose
    return (
        "import pathlib, shutil, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        "result, perf_path = runtime.time_all_bench_cases(bench_file, operator_file, bench_mode='perf-counter'); "  # noqa: E501
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _build_perf_counter_run_one_case_script(*, verbose: bool = False) -> str:
    del verbose
    return (
        "import json, pathlib, sys; "
        "import bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "case_id = sys.argv[3]; "
        "cases, resolution = runtime.load_bench_cases(bench_file, operator_file); "
        "case = runtime.select_bench_case(cases, case_id); "
        "record = runtime._time_bench_case(case, resolution, bench_mode='perf-counter'); "  # noqa: E501
        "payload = {"
        "'case_label': record.case_label, "
        "'kernel_names': record.kernel_names, "
        "'kernel_source': record.kernel_source, "
        "'metrics': record.metrics, "
        "'error_message': record.error_message, "
        "'case_wall_clock_seconds': record.case_wall_clock_seconds, "
        "'bench_mode': record.bench_mode"
        "}; "
        "print(json.dumps(payload, separators=(',', ':')))"
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
        "'case_wall_clock_seconds': record.case_wall_clock_seconds, "
        "'bench_mode': getattr(record, 'bench_mode', None)"
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
        extra_env[HELIX_BENCH_OUTPUT_DIR] = str(
            Path(configured_profile_root).expanduser().resolve()
        )
    extra_env[TRITON_ALWAYS_COMPILE] = "1"
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
        with stream_target_for_verbosity(True) as stream_target:
            result = run_streaming_process(
                command,
                str(workspace_root),
                stall_timeout_seconds=eval_stall_timeout_seconds(),
                stdout=stream_target,
                extra_env=extra_env,
            )
    else:
        result = run_buffered_process(
            command,
            str(workspace_root),
            stall_timeout_seconds=eval_stall_timeout_seconds(),
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
        "-c",
        _build_perf_counter_run_one_case_script(),
        _case_workspace_command_path(bench_file, source_root=source_root),
        _case_workspace_command_path(operator_file, source_root=source_root),
        case_id,
    ]
    if verbose:
        with stream_target_for_verbosity(True) as stream_target:
            result = run_streaming_process(
                command,
                str(workspace_root),
                stall_timeout_seconds=eval_stall_timeout_seconds(),
                stdout=stream_target,
                extra_env=extra_env,
            )
    else:
        result = run_buffered_process(
            command,
            str(workspace_root),
            stall_timeout_seconds=eval_stall_timeout_seconds(),
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
    extra_env[TRITON_ALWAYS_COMPILE] = "1"
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
        stall_timeout_seconds=eval_stall_timeout_seconds(),
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
            "-c",
            _build_perf_counter_run_one_case_script(),
            _case_workspace_command_path(bench_file, source_root=source_root),
            _case_workspace_command_path(operator_file, source_root=source_root),
            case_id,
        ],
        verbose=verbose,
        stderr=stderr,
        extra_env=extra_env,
        stall_timeout_seconds=eval_stall_timeout_seconds(),
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


def _format_torch_npu_profiler_command_failure(result: ResultPayload) -> str:
    return _format_worker_command_failure(result, "torch-npu-profiler")


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
                    stall_timeout_seconds=eval_stall_timeout_seconds(),
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
                    _bench_runtime_script_path().name,
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
                stall_timeout_seconds=eval_stall_timeout_seconds(),
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
