from __future__ import annotations

import tempfile
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Protocol, TextIO, TypeVar

from bench_contract import KernelResolution
from perf_artifacts import PerfCaseRecord, PerfMetrics, RequiredLatencyIds
from result_payload import ResultPayload
from run_runtime import RemoteSpec

_T = TypeVar("_T")


class StandaloneBenchCaseLike(Protocol):
    case_id: str


class StandaloneRuntimeModule(Protocol):
    def run_local_standalone_bench(
        self,
        bench_file: Path,
        operator_file: Path,
        *,
        verbose: bool = False,
    ) -> tuple[ResultPayload, Path]: ...

    def load_standalone_bench_cases(
        self,
        bench_file: Path,
        operator_file: Path,
    ) -> tuple[list[StandaloneBenchCaseLike], KernelResolution]: ...

    def runtime_support_paths(self) -> list[Path]: ...


class BenchRunnerDeps(Protocol):
    def resolve_bench_kernel_resolution(
        self,
        bench_file: Path,
        operator_file: Path | None = None,
    ) -> KernelResolution: ...

    def run_buffered_process(
        self,
        command: list[str],
        workdir: str,
        stall_timeout_seconds: int,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload: ...

    def local_python_executable(self) -> str: ...

    def _now(self) -> float: ...

    def _bench_timeout(self) -> int: ...

    def _parse_case_count(self, stdout: str) -> int: ...

    def _create_local_msprof_output_dir(
        self,
        case_idx: int,
        preserved_run_dir: Path | None,
    ) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]: ...

    def _stream_target_for_verbosity(self, verbose: bool) -> AbstractContextManager[TextIO]: ...

    def run_streaming_process(
        self,
        command: list[str],
        workdir: str,
        stall_timeout_seconds: int,
        stdout: TextIO | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload: ...

    def _format_msprof_command_failure(self, result: ResultPayload) -> str: ...

    def _cleanup_local_bench_extra_info(self, workdir: Path) -> None: ...

    def _case_workspace_command_path(self, path: Path, *, source_root: Path) -> str: ...

    def _run_parallel_case_workers(
        self,
        case_keys: Sequence[str],
        max_workers: int,
        worker: Callable[[str], _T],
    ) -> list[_T]: ...

    def run_remote_command_buffered(
        self,
        spec: RemoteSpec,
        remote_workdir: str,
        remote_command: str | Sequence[str],
        verbose: bool = False,
        stderr: TextIO | None = None,
        stall_timeout_seconds: int | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> ResultPayload: ...

    def _create_remote_msprof_output_dir(
        self,
        spec: RemoteSpec,
        remote_workspace: str,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> str: ...

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
    ) -> ResultPayload: ...

    def _read_remote_msprof_metrics(
        self,
        spec: RemoteSpec,
        remote_workspace: str,
        output_dir: str,
        kernel_names: list[str],
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> PerfMetrics: ...

    def _cleanup_remote_msprof_output_dir(
        self,
        spec: RemoteSpec,
        remote_workspace: str,
        output_dir: str,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> None: ...

    def _create_local_msprof_case_workspace(
        self,
        bench_file: Path,
        operator_file: Path,
        case_idx: int,
        *,
        source_root: Path,
        json_search_root: Path,
    ) -> tuple[Path, Callable[[], None]]: ...

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
    ) -> str: ...

    def write_perf_lines(self, path: Path, lines: Sequence[str]) -> Path: ...

    def perf_output_path(self, operator_file: Path) -> Path: ...

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
    ) -> list[str]: ...

    def render_perf_case_records_jsonl(
        self,
        case_records: list[PerfCaseRecord],
        *,
        missing_kernel_match_error: str | None = None,
    ) -> list[str]: ...

    def _resolve_local_bench_profile_output_root(self) -> tuple[str | None, str]: ...

    def _set_directory_owner_only(self, path: Path) -> None: ...

    def _standalone_runtime_support_paths(self) -> list[Path]: ...

    def copy_file_to_remote(
        self,
        spec: RemoteSpec,
        local_path: Path,
        remote_path: str,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> ResultPayload | None: ...

    def copy_file_from_remote(
        self,
        spec: RemoteSpec,
        remote_path: str,
        local_path: Path,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> ResultPayload | None: ...

    def _load_standalone_runtime_module(self) -> StandaloneRuntimeModule: ...

    def _sort_case_records(
        self,
        case_records: list[PerfCaseRecord],
        ordered_case_labels: Sequence[str],
    ) -> None: ...

    def _create_local_case_workspace(
        self,
        *,
        prefix: str,
        input_paths: Sequence[Path],
        source_root: Path,
    ) -> tuple[Path, Callable[[], None]]: ...

    def _bench_case_input_paths(
        self,
        bench_file: Path,
        operator_file: Path,
        *,
        json_search_root: Path | None = None,
        support_paths: Sequence[Path] = (),
    ) -> list[Path]: ...

    def _stage_remote_case_workspace(
        self,
        spec: RemoteSpec,
        case_workspace: str,
        input_paths: Sequence[Path],
        source_root: Path,
        *,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> str: ...

    def parse_required_perf_file(
        self,
        path: Path,
        required_latency_ids: RequiredLatencyIds,
    ) -> dict[str, float]: ...
