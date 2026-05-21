from __future__ import annotations
# pyright: reportUnusedImport=false, reportUnusedFunction=false

import contextlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO, TypeVar, cast

import bench_runner_msprof as _msprof
import bench_runner_standalone as _standalone
from bench_runner_deps import BenchRunnerDeps
from bench_contract import (
    KernelResolution,
    parse_bench_metadata as _parse_bench_metadata,
    resolve_bench_kernel_names as _resolve_bench_kernel_names,
    resolve_bench_kernel_resolution as _resolve_bench_kernel_resolution,
)
from npu_affinity import parse_npu_devices
from perf_artifacts import (
    MetricSource,
    PerfCaseRecord,
    PerfMetrics,
    RequiredLatencyIds,
    compare_perf_files as _compare_perf_files,
    parse_perf_file as _parse_perf_file,
    parse_perf_file_for_metric_source as _parse_perf_file_for_metric_source,
    parse_required_perf_file as _parse_required_perf_file,
    parse_required_perf_file_for_metric_source as _parse_required_perf_file_for_metric_source,
    perf_output_path,
    render_perf_case_records,
    write_perf_lines,
)
from run_runtime import (
    RemoteSpec,
    ResultPayload,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    env_int,
    local_python_executable,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)

_LOCAL_BENCH_PROFILE_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR"
_T = TypeVar("_T")


class _BenchRunnerDeps:
    def resolve_bench_kernel_resolution(
        self,
        bench_file: Path,
        operator_file: Path | None = None,
    ) -> KernelResolution:
        return resolve_bench_kernel_resolution(bench_file, operator_file)

    def run_buffered_process(
        self,
        command: list[str],
        workdir: str,
        stall_timeout_seconds: int,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload:
        return run_buffered_process(command, workdir, stall_timeout_seconds, extra_env=extra_env)

    def local_python_executable(self) -> str:
        return local_python_executable()

    def _now(self) -> float:
        return time.monotonic()

    def _bench_timeout(self) -> int:
        return _bench_timeout()

    def _parse_case_count(self, stdout: str) -> int:
        return _parse_case_count(stdout)

    def _create_local_msprof_output_dir(
        self,
        case_idx: int,
        preserved_run_dir: Path | None,
    ) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
        return _create_local_msprof_output_dir(case_idx, preserved_run_dir)

    def _stream_target_for_verbosity(self, verbose: bool) -> contextlib.AbstractContextManager[TextIO]:
        return _stream_target_for_verbosity(verbose)

    def run_streaming_process(
        self,
        command: list[str],
        workdir: str,
        stall_timeout_seconds: int,
        stdout: TextIO | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload:
        return run_streaming_process(
            command,
            workdir,
            stall_timeout_seconds,
            stdout=stdout,
            extra_env=extra_env,
        )

    def _format_msprof_command_failure(self, result: ResultPayload) -> str:
        return _format_msprof_command_failure(result)

    def _cleanup_local_bench_extra_info(self, workdir: Path) -> None:
        _cleanup_local_bench_extra_info(workdir)

    def _case_workspace_command_path(self, path: Path, *, source_root: Path) -> str:
        return _case_workspace_command_path(path, source_root=source_root)

    def _run_parallel_case_workers(
        self,
        case_keys: Sequence[str],
        max_workers: int,
        worker: Callable[[str], _T],
    ) -> list[_T]:
        return _run_parallel_case_workers(case_keys, max_workers, worker)

    def run_remote_command_buffered(
        self,
        spec: RemoteSpec,
        remote_workdir: str,
        remote_command: str | Sequence[str],
        verbose: bool = False,
        stderr: TextIO | None = None,
        stall_timeout_seconds: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload:
        return run_remote_command_buffered(
            spec,
            remote_workdir,
            remote_command,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=stall_timeout_seconds,
            extra_env=extra_env,
        )

    def _create_remote_msprof_output_dir(
        self,
        spec: RemoteSpec,
        remote_workspace: str,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> str:
        return _create_remote_msprof_output_dir(
            spec,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )

    def run_remote_command_streaming(
        self,
        spec: RemoteSpec,
        remote_workdir: str,
        remote_command: str | Sequence[str],
        stdout: TextIO | None = None,
        verbose: bool = False,
        stderr: TextIO | None = None,
        stall_timeout_seconds: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload:
        return run_remote_command_streaming(
            spec,
            remote_workdir,
            remote_command,
            stdout=stdout,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=stall_timeout_seconds,
            extra_env=extra_env,
        )

    def _read_remote_msprof_metrics(
        self,
        spec: RemoteSpec,
        remote_workspace: str,
        output_dir: str,
        kernel_names: list[str],
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> PerfMetrics:
        return _read_remote_msprof_metrics(
            spec,
            remote_workspace,
            output_dir,
            kernel_names,
            verbose=verbose,
            stderr=stderr,
        )

    def _cleanup_remote_msprof_output_dir(
        self,
        spec: RemoteSpec,
        remote_workspace: str,
        output_dir: str,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> None:
        _cleanup_remote_msprof_output_dir(
            spec,
            remote_workspace,
            output_dir,
            verbose=verbose,
            stderr=stderr,
        )

    def _create_local_msprof_case_workspace(
        self,
        bench_file: Path,
        operator_file: Path,
        case_idx: int,
        *,
        source_root: Path,
        json_search_root: Path,
    ) -> tuple[Path, Callable[[], None]]:
        return _create_local_msprof_case_workspace(
            bench_file,
            operator_file,
            case_idx,
            source_root=source_root,
            json_search_root=json_search_root,
        )

    def _stage_remote_msprof_case_workspace(
        self,
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
        return _stage_remote_msprof_case_workspace(
            spec,
            bench_file,
            operator_file,
            case_workspace,
            source_root=source_root,
            json_search_root=json_search_root,
            verbose=verbose,
            stderr=stderr,
        )

    def write_perf_lines(self, path: Path, lines: Sequence[str]) -> Path:
        return write_perf_lines(path, list(lines))

    def perf_output_path(self, operator_file: Path) -> Path:
        return perf_output_path(operator_file)

    def render_perf_case_records(
        self,
        case_records: list[PerfCaseRecord],
        *,
        latency_prefix: str,
        raw_prefix: str,
        resolved_kernels_prefix: str,
        kernel_source_prefix: str,
        latency_error_prefix: str,
        missing_kernel_match_error: str,
        elapsed_id_prefix: str | None = None,
    ) -> list[str]:
        kwargs: dict[str, str] = {
            "latency_prefix": latency_prefix,
            "raw_prefix": raw_prefix,
            "resolved_kernels_prefix": resolved_kernels_prefix,
            "kernel_source_prefix": kernel_source_prefix,
            "latency_error_prefix": latency_error_prefix,
            "missing_kernel_match_error": missing_kernel_match_error,
        }
        if elapsed_id_prefix is not None:
            kwargs["elapsed_id_prefix"] = elapsed_id_prefix
        return render_perf_case_records(case_records, **kwargs)

    def _resolve_local_bench_profile_output_root(self) -> tuple[str | None, str]:
        return _resolve_local_bench_profile_output_root()

    def _set_directory_owner_only(self, path: Path) -> None:
        _set_directory_owner_only(path)

    def _standalone_runtime_support_paths(self) -> list[Path]:
        return _standalone_runtime_support_paths()

    def copy_file_to_remote(
        self,
        spec: RemoteSpec,
        local_path: Path,
        remote_path: str,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> ResultPayload | None:
        return copy_file_to_remote(
            spec,
            local_path,
            remote_path,
            verbose=verbose,
            stderr=stderr,
        )

    def copy_file_from_remote(
        self,
        spec: RemoteSpec,
        remote_path: str,
        local_path: Path,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> ResultPayload | None:
        return copy_file_from_remote(
            spec,
            remote_path,
            local_path,
            verbose=verbose,
            stderr=stderr,
        )

    def _load_standalone_runtime_module(self):
        return _load_standalone_runtime_module()

    def _sort_case_records(
        self,
        case_records: list[PerfCaseRecord],
        ordered_case_labels: Sequence[str],
    ) -> None:
        _sort_case_records(case_records, ordered_case_labels)

    def _create_local_case_workspace(
        self,
        *,
        prefix: str,
        input_paths: Sequence[Path],
        source_root: Path,
    ) -> tuple[Path, Callable[[], None]]:
        return _create_local_case_workspace(
            prefix=prefix,
            input_paths=input_paths,
            source_root=source_root,
        )

    def _bench_case_input_paths(
        self,
        bench_file: Path,
        operator_file: Path,
        *,
        json_search_root: Path | None = None,
        support_paths: Sequence[Path] = (),
    ) -> list[Path]:
        return _bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
            support_paths=support_paths,
        )

    def _stage_remote_case_workspace(
        self,
        spec: RemoteSpec,
        case_workspace: str,
        input_paths: Sequence[Path],
        source_root: Path,
        *,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> str:
        return _stage_remote_case_workspace(
            spec,
            case_workspace,
            input_paths,
            source_root,
            verbose=verbose,
            stderr=stderr,
        )

    def parse_required_perf_file(
        self,
        path: Path,
        required_latency_ids: RequiredLatencyIds,
    ) -> dict[str, float]:
        return parse_required_perf_file(path, required_latency_ids)


_DEPS = cast(BenchRunnerDeps, _BenchRunnerDeps())


def _bench_timeout() -> int:
    return env_int("TRITON_AGENT_BENCH_TIMEOUT_SECONDS", 900)


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    return _parse_bench_metadata(bench_file)


def resolve_bench_kernel_names(
    bench_file: Path,
    operator_file: Path | None = None,
) -> list[str]:
    return _resolve_bench_kernel_names(bench_file, operator_file)


def resolve_bench_kernel_resolution(
    bench_file: Path,
    operator_file: Path | None = None,
) -> KernelResolution:
    return _resolve_bench_kernel_resolution(bench_file, operator_file)


def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: MetricSource = "auto",
) -> int:
    return _compare_perf_files(
        baseline_perf,
        compare_perf,
        skip_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )


def parse_perf_file(path: Path) -> dict[str, float]:
    return _parse_perf_file(path)


def parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    return _parse_required_perf_file(path, required_latency_ids)


def parse_perf_file_for_metric_source(
    path: Path,
    *,
    metric_source: MetricSource = "auto",
) -> dict[str, float]:
    return _parse_perf_file_for_metric_source(path, metric_source=metric_source)


def parse_required_perf_file_for_metric_source(
    path: Path,
    required_latency_ids: RequiredLatencyIds,
    *,
    metric_source: MetricSource = "auto",
) -> dict[str, float]:
    return _parse_required_perf_file_for_metric_source(
        path,
        required_latency_ids,
        metric_source=metric_source,
    )


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    npu_devices: str | None = None,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    invocation_root = Path.cwd().resolve()
    devices = parse_npu_devices(npu_devices)
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
                )
            return _run_local_bench_msprof(bench_file, operator_file, verbose=verbose)
        if devices is not None:
            source_root, json_search_root = _resolve_case_workspace_roots(
                bench_file,
                operator_file,
                invocation_root=invocation_root,
            )
            return _run_local_bench_standalone_parallel(
                bench_file,
                operator_file,
                devices,
                source_root=source_root,
                json_search_root=json_search_root,
                verbose=verbose,
            )
        return _run_local_bench_standalone(bench_file, operator_file, verbose=verbose)


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
) -> tuple[ResultPayload, Path | None, str]:
    invocation_root = Path.cwd().resolve()
    devices = parse_npu_devices(npu_devices)
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
                )
            return _run_remote_bench_msprof(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
                verbose=verbose,
                stderr=stderr,
            )
        if devices is not None:
            source_root, json_search_root = _resolve_case_workspace_roots(
                bench_file,
                operator_file,
                invocation_root=invocation_root,
            )
            return _run_remote_bench_standalone_parallel(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
                devices,
                source_root=source_root,
                json_search_root=json_search_root,
                verbose=verbose,
                stderr=stderr,
            )
        return _run_remote_bench_standalone(
            spec,
            remote_workspace,
            bench_file,
            operator_file,
            verbose=verbose,
            stderr=stderr,
        )
    finally:
        if not keep_remote_workdir:
            cleanup_remote_workspace(spec, remote_workspace, verbose=verbose, stderr=stderr)


def _run_local_bench_standalone(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    return run_local_standalone_bench(bench_file, operator_file, verbose=verbose)


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


def _run_remote_bench_standalone(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    return _standalone.run_remote_bench_standalone(
        _DEPS,
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
    )


def run_local_standalone_bench(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path]:
    return _standalone.run_local_standalone_bench(
        _DEPS,
        bench_file,
        operator_file,
        verbose=verbose,
    )


def _run_local_bench_standalone_parallel(
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
) -> tuple[ResultPayload, Path]:
    return _standalone.run_local_bench_standalone_parallel(
        _DEPS,
        bench_file,
        operator_file,
        devices,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
    )


def _run_remote_bench_standalone_parallel(
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
) -> tuple[ResultPayload, Path, str]:
    return _standalone.run_remote_bench_standalone_parallel(
        _DEPS,
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        devices,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
        stderr=stderr,
    )


def _standalone_runtime_script_path() -> Path:
    return Path(__file__).resolve().with_name("standalone_bench_runtime.py")


def _standalone_runtime_support_paths() -> list[Path]:
    runtime = _load_standalone_runtime_module()
    return cast(list[Path], runtime.runtime_support_paths())


def _load_standalone_runtime_module():
    script_path = _standalone_runtime_script_path()
    module_name = f"triton_agent_standalone_bench_runtime_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load standalone runtime helper: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _run_local_bench_msprof(
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    return _msprof.run_local_bench_msprof(
        _DEPS,
        bench_file,
        operator_file,
        verbose=verbose,
    )


def _run_local_bench_msprof_parallel(
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    return _msprof.run_local_bench_msprof_parallel(
        _DEPS,
        bench_file,
        operator_file,
        devices,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
    )


def _run_remote_bench_msprof(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    return _msprof.run_remote_bench_msprof(
        _DEPS,
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        verbose=verbose,
        stderr=stderr,
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
) -> tuple[ResultPayload, Path | None, str]:
    return _msprof.run_remote_bench_msprof_parallel(
        _DEPS,
        spec,
        remote_workspace,
        bench_file,
        operator_file,
        devices,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
        stderr=stderr,
    )


def _parse_case_count(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ValueError("Unable to parse benchmark case count from --num-bench output.")


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
    configured_root = os.environ.get(_LOCAL_BENCH_PROFILE_OUTPUT_DIR_ENV)
    if configured_root:
        return configured_root, _LOCAL_BENCH_PROFILE_OUTPUT_DIR_ENV
    return None, _LOCAL_BENCH_PROFILE_OUTPUT_DIR_ENV


def _create_local_msprof_output_dir(
    case_idx: int,
    preserved_run_dir: Path | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if preserved_run_dir is None:
        temp_dir = tempfile.TemporaryDirectory(prefix="triton-agent-msprof-")
        return Path(temp_dir.name), temp_dir
    output_dir = preserved_run_dir / f"case-{case_idx}"
    output_dir.mkdir(parents=True, exist_ok=False)
    _set_directory_owner_only(output_dir)
    return output_dir, None


def _bench_case_input_paths(
    bench_file: Path,
    operator_file: Path,
    *,
    json_search_root: Path | None = None,
    support_paths: Sequence[Path] = (),
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
    input_paths.extend(support_paths)

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


def _local_case_workspace_path(
    workspace_root: Path,
    source_path: Path,
    *,
    source_root: Path,
) -> Path:
    return workspace_root / _case_workspace_root_relative_path(source_path, source_root=source_root)


def _remote_case_workspace_path(
    workspace_root: str,
    source_path: Path,
    *,
    source_root: Path,
) -> str:
    relative_path = _case_workspace_root_relative_path(source_path, source_root=source_root)
    return f"{workspace_root}/{relative_path.as_posix()}"


def _create_local_case_workspace(
    *,
    prefix: str,
    input_paths: Sequence[Path],
    source_root: Path,
) -> tuple[Path, Callable[[], None]]:
    temp_dir = tempfile.TemporaryDirectory(prefix=prefix)
    workspace = Path(temp_dir.name)
    workspace_root = workspace / _case_workspace_root_name(source_root)
    workspace_root.mkdir(parents=True, exist_ok=True)
    for input_path in input_paths:
        relative_path = _case_workspace_root_relative_path(input_path, source_root=source_root)
        target_path = workspace_root / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(input_path, target_path)
    return workspace_root, temp_dir.cleanup


def _create_local_msprof_case_workspace(
    bench_file: Path,
    operator_file: Path,
    case_idx: int,
    *,
    source_root: Path,
    json_search_root: Path,
) -> tuple[Path, Callable[[], None]]:
    return _create_local_case_workspace(
        prefix=f"triton-agent-msprof-case-{case_idx}-",
        input_paths=_bench_case_input_paths(
            bench_file,
            operator_file,
            json_search_root=json_search_root,
        ),
        source_root=source_root,
    )


def _stage_remote_case_workspace(
    spec: RemoteSpec,
    case_workspace: str,
    input_paths: Sequence[Path],
    source_root: Path,
    *,
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
