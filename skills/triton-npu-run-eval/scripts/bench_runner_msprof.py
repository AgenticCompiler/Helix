from __future__ import annotations
# pyright: reportPrivateUsage=false

import csv
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bench_runner_deps import BenchRunnerDeps
from bench_contract import KernelResolution
from npu_affinity import NpuDevicePool, affinity_env_for_device
from perf_artifacts import PerfCaseRecord, PerfMetrics, PerfOpRow
from run_runtime import RemoteSpec, ResultPayload, make_result, result_succeeded

_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched op_statistic csv"


@dataclass(frozen=True)
class _MsprofCaseOutcome:
    case_idx: int
    record: PerfCaseRecord
    stdout: str
    stderr: str
    stalled: bool
    session_id: str | None


def _msprof_case_outcome_sort_key(outcome: _MsprofCaseOutcome) -> int:
    return outcome.case_idx


def _msprof_case_label(case_idx: int) -> str:
    return f"case-{case_idx}"


def run_local_bench_msprof(
    deps: BenchRunnerDeps,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    resolution = deps.resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = deps.run_buffered_process(
        [deps.local_python_executable(), bench_file.name, "--num-bench"],
        str(bench_file.parent),
        stall_timeout_seconds=deps._bench_timeout(),
    )
    if not result_succeeded(count_result):
        return count_result, None

    case_count = deps._parse_case_count(str(count_result["stdout"]))
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_records: list[PerfCaseRecord] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir(deps)
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    for case_idx in range(1, case_count + 1):
        output_dir, temp_dir = deps._create_local_msprof_output_dir(case_idx, preserved_run_dir)
        try:
            command = [
                "msprof",
                f"--output={output_dir}",
                deps.local_python_executable(),
                bench_file.name,
                "--operator-file",
                operator_arg,
                "--bench",
                str(case_idx),
            ]
            t0 = deps._now()
            with deps._stream_target_for_verbosity(verbose) as stream_target:
                result = deps.run_streaming_process(
                    command,
                    str(bench_file.parent),
                    stall_timeout_seconds=deps._bench_timeout(),
                    stdout=stream_target,
                )
            elapsed = deps._now() - t0
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=_msprof_case_label(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=deps._format_msprof_command_failure(result),
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
                        case_label=_msprof_case_label(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                        case_wall_clock_seconds=elapsed,
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=_msprof_case_label(case_idx),
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                    case_wall_clock_seconds=elapsed,
                )
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            deps._cleanup_local_bench_extra_info(bench_file.parent)

    perf_path = _write_msprof_perf(deps, operator_file, case_records)
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


def run_local_bench_msprof_parallel(
    deps: BenchRunnerDeps,
    bench_file: Path,
    operator_file: Path,
    devices: tuple[str, ...],
    *,
    source_root: Path,
    json_search_root: Path,
    verbose: bool = False,
) -> tuple[ResultPayload, Path | None]:
    resolution = deps.resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = deps.run_buffered_process(
        [deps.local_python_executable(), bench_file.name, "--num-bench"],
        str(bench_file.parent),
        stall_timeout_seconds=deps._bench_timeout(),
    )
    if not result_succeeded(count_result):
        return count_result, None

    case_count = deps._parse_case_count(str(count_result["stdout"]))
    bench_arg = deps._case_workspace_command_path(bench_file, source_root=source_root)
    operator_arg = deps._case_workspace_command_path(operator_file, source_root=source_root)
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    preserved_run_dir = _create_local_msprof_preserved_run_dir(deps)
    case_labels = [str(case_idx) for case_idx in range(1, case_count + 1)]
    pool = NpuDevicePool(devices)

    def _worker(case_label: str) -> _MsprofCaseOutcome:
        return _run_local_msprof_case_parallel(
            deps,
            bench_file,
            operator_file,
            operator_arg,
            bench_arg,
            resolution,
            int(case_label),
            pool,
            preserved_run_dir,
            source_root,
            json_search_root,
            verbose,
        )

    outcomes = deps._run_parallel_case_workers(
        case_labels,
        min(case_count, len(devices)),
        _worker,
    )
    outcomes.sort(key=_msprof_case_outcome_sort_key)
    perf_path = _write_msprof_perf(deps, operator_file, [outcome.record for outcome in outcomes])
    for outcome in outcomes:
        stdout_chunks.append(outcome.stdout)
        stderr_chunks.append(outcome.stderr)
    return _build_msprof_result(stdout_chunks, stderr_chunks, outcomes), perf_path


def run_remote_bench_msprof(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    resolution = deps.resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = deps.run_remote_command_buffered(
        spec,
        remote_workspace,
        ["python3", bench_file.name, "--num-bench"],
        verbose=verbose,
        stderr=stderr,
        stall_timeout_seconds=deps._bench_timeout(),
    )
    if not result_succeeded(count_result):
        return count_result, None, remote_workspace

    case_count = deps._parse_case_count(str(count_result["stdout"]))
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_records: list[PerfCaseRecord] = []
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    for case_idx in range(1, case_count + 1):
        output_dir = deps._create_remote_msprof_output_dir(
            spec,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            t0 = deps._now()
            result = deps.run_remote_command_streaming(
                spec,
                remote_workspace,
                [
                    "msprof",
                    f"--output={output_dir}",
                    "python3",
                    bench_file.name,
                    "--operator-file",
                    operator_file.name,
                    "--bench",
                    str(case_idx),
                ],
                verbose=verbose,
                stderr=stderr,
                stall_timeout_seconds=deps._bench_timeout(),
            )
            elapsed = deps._now() - t0
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=_msprof_case_label(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=deps._format_msprof_command_failure(result),
                        case_wall_clock_seconds=elapsed,
                    ),
                )
                continue

            try:
                metrics = deps._read_remote_msprof_metrics(
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
                        case_label=_msprof_case_label(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                        case_wall_clock_seconds=elapsed,
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=_msprof_case_label(case_idx),
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                    case_wall_clock_seconds=elapsed,
                )
            )
        finally:
            deps._cleanup_remote_msprof_output_dir(
                spec,
                remote_workspace,
                output_dir,
                verbose=verbose,
                stderr=stderr,
            )

    perf_path = _write_msprof_perf(deps, operator_file, case_records)
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


def run_remote_bench_msprof_parallel(
    deps: BenchRunnerDeps,
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
    resolution = deps.resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = deps.run_remote_command_buffered(
        spec,
        remote_workspace,
        ["python3", bench_file.name, "--num-bench"],
        verbose=verbose,
        stderr=stderr,
        stall_timeout_seconds=deps._bench_timeout(),
    )
    if not result_succeeded(count_result):
        return count_result, None, remote_workspace

    case_count = deps._parse_case_count(str(count_result["stdout"]))
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_labels = [str(case_idx) for case_idx in range(1, case_count + 1)]
    pool = NpuDevicePool(devices)

    def _worker(case_label: str) -> _MsprofCaseOutcome:
        return _run_remote_msprof_case_parallel(
            deps,
            spec,
            remote_workspace,
            bench_file,
            operator_file,
            resolution,
            int(case_label),
            pool,
            source_root,
            json_search_root,
            verbose,
            stderr,
        )

    outcomes = deps._run_parallel_case_workers(
        case_labels,
        min(case_count, len(devices)),
        _worker,
    )
    outcomes.sort(key=_msprof_case_outcome_sort_key)
    perf_path = _write_msprof_perf(deps, operator_file, [outcome.record for outcome in outcomes])
    for outcome in outcomes:
        stdout_chunks.append(outcome.stdout)
        stderr_chunks.append(outcome.stderr)
    return _build_msprof_result(stdout_chunks, stderr_chunks, outcomes), perf_path, remote_workspace


def _run_local_msprof_case_parallel(
    deps: BenchRunnerDeps,
    bench_file: Path,
    operator_file: Path,
    operator_arg: str,
    bench_arg: str,
    resolution: KernelResolution,
    case_idx: int,
    pool: NpuDevicePool,
    preserved_run_dir: Path | None,
    source_root: Path,
    json_search_root: Path,
    verbose: bool,
) -> _MsprofCaseOutcome:
    case_workspace, cleanup = deps._create_local_msprof_case_workspace(
        bench_file,
        operator_file,
        case_idx,
        source_root=source_root,
        json_search_root=json_search_root,
    )
    output_dir, temp_dir = deps._create_local_msprof_output_dir(case_idx, preserved_run_dir)
    try:
        with pool.acquire() as device:
            command = [
                "msprof",
                f"--output={output_dir}",
                deps.local_python_executable(),
                bench_arg,
                "--operator-file",
                operator_arg,
                "--bench",
                str(case_idx),
            ]
            t0 = deps._now()
            with deps._stream_target_for_verbosity(verbose) as stream_target:
                result = deps.run_streaming_process(
                    command,
                    str(case_workspace),
                    stall_timeout_seconds=deps._bench_timeout(),
                    stdout=stream_target,
                    extra_env=affinity_env_for_device(device),
                )
            elapsed = deps._now() - t0
        return _build_local_msprof_case_outcome(deps, result, resolution, case_idx, output_dir, elapsed)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
        deps._cleanup_local_bench_extra_info(case_workspace)
        cleanup()


def _run_remote_msprof_case_parallel(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    resolution: KernelResolution,
    case_idx: int,
    pool: NpuDevicePool,
    source_root: Path,
    json_search_root: Path,
    verbose: bool,
    stderr: TextIO | None,
) -> _MsprofCaseOutcome:
    case_workspace = f"{remote_workspace}/case-{case_idx}"
    deps.run_remote_command_buffered(
        spec,
        remote_workspace,
        ["mkdir", "-p", case_workspace],
        verbose=verbose,
        stderr=stderr,
    )
    workspace_root = deps._stage_remote_msprof_case_workspace(
        spec,
        bench_file,
        operator_file,
        case_workspace,
        source_root=source_root,
        json_search_root=json_search_root,
        verbose=verbose,
        stderr=stderr,
    )
    bench_arg = deps._case_workspace_command_path(bench_file, source_root=source_root)
    operator_arg = deps._case_workspace_command_path(operator_file, source_root=source_root)
    output_dir = f"{workspace_root}/msprof-output"
    try:
        with pool.acquire() as device:
            t0 = deps._now()
            result = deps.run_remote_command_streaming(
                spec,
                workspace_root,
                [
                    "msprof",
                    f"--output={output_dir}",
                    "python3",
                    bench_arg,
                    "--operator-file",
                    operator_arg,
                    "--bench",
                    str(case_idx),
                ],
                verbose=verbose,
                stderr=stderr,
                extra_env=affinity_env_for_device(device),
                stall_timeout_seconds=deps._bench_timeout(),
            )
            elapsed = deps._now() - t0
        return _build_remote_msprof_case_outcome(
            deps,
            spec,
            workspace_root,
            result,
            resolution,
            case_idx,
            output_dir,
            elapsed,
            verbose=verbose,
            stderr=stderr,
        )
    finally:
        deps.run_remote_command_buffered(
            spec,
            remote_workspace,
            ["rm", "-rf", case_workspace],
            verbose=verbose,
            stderr=stderr,
        )


def _build_local_msprof_case_outcome(
    deps: BenchRunnerDeps,
    result: ResultPayload,
    resolution: KernelResolution,
    case_idx: int,
    output_dir: Path,
    elapsed: float,
) -> _MsprofCaseOutcome:
    if not result_succeeded(result):
        record = PerfCaseRecord(
            case_label=_msprof_case_label(case_idx),
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=deps._format_msprof_command_failure(result),
            case_wall_clock_seconds=elapsed,
        )
    else:
        try:
            metrics = _read_local_msprof_metrics(output_dir, resolution.kernel_names)
            record = PerfCaseRecord(
                case_label=_msprof_case_label(case_idx),
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                metrics=metrics,
                case_wall_clock_seconds=elapsed,
            )
        except (FileNotFoundError, ValueError) as exc:
            record = PerfCaseRecord(
                case_label=_msprof_case_label(case_idx),
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=str(exc),
                case_wall_clock_seconds=elapsed,
            )
    return _MsprofCaseOutcome(
        case_idx=case_idx,
        record=record,
        stdout=str(result["stdout"]),
        stderr=str(result["stderr"]),
        stalled=bool(result["stalled"]),
        session_id=result["session_id"],
    )


def _build_remote_msprof_case_outcome(
    deps: BenchRunnerDeps,
    spec: RemoteSpec,
    remote_workspace: str,
    result: ResultPayload,
    resolution: KernelResolution,
    case_idx: int,
    output_dir: str,
    elapsed: float,
    *,
    verbose: bool,
    stderr: TextIO | None,
) -> _MsprofCaseOutcome:
    if not result_succeeded(result):
        record = PerfCaseRecord(
            case_label=_msprof_case_label(case_idx),
            kernel_names=resolution.kernel_names,
            kernel_source=resolution.kernel_source,
            error_message=deps._format_msprof_command_failure(result),
            case_wall_clock_seconds=elapsed,
        )
    else:
        try:
            metrics = deps._read_remote_msprof_metrics(
                spec,
                remote_workspace,
                output_dir,
                resolution.kernel_names,
                verbose=verbose,
                stderr=stderr,
            )
            record = PerfCaseRecord(
                case_label=_msprof_case_label(case_idx),
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                metrics=metrics,
                case_wall_clock_seconds=elapsed,
            )
        except RuntimeError as exc:
            record = PerfCaseRecord(
                case_label=_msprof_case_label(case_idx),
                kernel_names=resolution.kernel_names,
                kernel_source=resolution.kernel_source,
                error_message=str(exc),
                case_wall_clock_seconds=elapsed,
            )
    return _MsprofCaseOutcome(
        case_idx=case_idx,
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
    deps: BenchRunnerDeps,
    operator_file: Path,
    case_records: list[PerfCaseRecord],
) -> Path:
    return deps.write_perf_lines(
        deps.perf_output_path(operator_file),
        deps.render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
        ),
    )


def _load_msprof_avg_rows(output_dir: Path) -> list[PerfOpRow]:
    csv_path = _find_latest_msprof_statistic_csv(output_dir)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Avg Time(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Avg Time(us)' in {csv_path}")
        if "OP Type" not in fieldnames:
            raise ValueError(f"Missing required column 'OP Type' in {csv_path}")
        rows: list[PerfOpRow] = []
        row_count = 0
        for row in reader:
            avg_time = (row.get("Avg Time(us)") or "").strip()
            if not avg_time:
                raise ValueError(f"Empty 'Avg Time(us)' value in {csv_path}")
            op_type = (row.get("OP Type") or "").strip()
            if not op_type:
                raise ValueError(f"Empty 'OP Type' value in {csv_path}")
            rows.append(
                {
                    "op_type": op_type,
                    "avg_time_us": float(avg_time),
                }
            )
            row_count += 1
    if row_count == 0:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _find_latest_msprof_statistic_csv(output_dir: Path) -> Path:
    matches = sorted(
        path for path in output_dir.rglob("op_statistic_*.csv") if path.is_file()
    )
    if not matches:
        raise FileNotFoundError(f"No op_statistic_*.csv found under {output_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _resolve_msprof_metrics(
    rows: list[PerfOpRow],
    kernel_names: list[str],
) -> PerfMetrics:
    kernel_name_set = set(kernel_names)
    matched_avg_times = [
        float(row["avg_time_us"]) for row in rows if str(row["op_type"]) in kernel_name_set
    ]
    kernel_avg_time_us = sum(matched_avg_times) if matched_avg_times else None
    return {
        "kernel_avg_time_us": kernel_avg_time_us,
        "ops": [
            {
                "op_type": row["op_type"],
                "avg_time_us": row["avg_time_us"],
            }
            for row in rows
        ],
    }


def _read_local_msprof_metrics(output_dir: Path, kernel_names: list[str]) -> PerfMetrics:
    return _resolve_msprof_metrics(_load_msprof_avg_rows(output_dir), kernel_names)


def _create_local_msprof_preserved_run_dir(deps: BenchRunnerDeps) -> Path | None:
    configured_root, configured_env = deps._resolve_local_bench_profile_output_root()
    if not configured_root:
        return None
    root = Path(configured_root).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"{configured_env} must point to a directory: {root}"
        )
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        deps._set_directory_owner_only(root)
    run_dir = Path(tempfile.mkdtemp(prefix="triton-agent-msprof-", dir=str(root)))
    deps._set_directory_owner_only(run_dir)
    return run_dir
