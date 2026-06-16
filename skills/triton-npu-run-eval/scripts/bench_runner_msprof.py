from __future__ import annotations
# pyright: reportPrivateUsage=false

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from bench_runner_deps import BenchRunnerDeps
from bench_contract import KernelResolution
from npu_affinity import NpuDevicePool, affinity_env_for_device
from perf_artifacts import PerfCaseRecord, PerfMetrics, PerfOpRow
from profile_csv_parser import (
    find_latest_op_statistic_csv,
    parse_op_statistic_csv,
    resolve_perf_metrics,
)
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
    output: str | None = None,
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
                    extra_env={"TRITON_ALWAYS_COMPILE": "1"},
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

    perf_path = _write_msprof_perf(deps, operator_file, case_records, output=output)
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
    output: str | None = None,
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
    perf_path = _write_msprof_perf(deps, operator_file, [outcome.record for outcome in outcomes], output=output)
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
    output: str | None = None,
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

    perf_path = _write_msprof_perf(deps, operator_file, case_records, output=output)
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
    output: str | None = None,
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
    perf_path = _write_msprof_perf(deps, operator_file, [outcome.record for outcome in outcomes], output=output)
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
        verbose=verbose,
    )
    output_dir, temp_dir = deps._create_local_msprof_output_dir(case_idx, preserved_run_dir)
    try:
        with pool.acquire() as device:
            extra_env = affinity_env_for_device(device)
            extra_env["TRITON_ALWAYS_COMPILE"] = "1"
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
                    extra_env=extra_env,
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
            extra_env = affinity_env_for_device(device)
            extra_env["TRITON_ALWAYS_COMPILE"] = "1"
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
                extra_env=extra_env,
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
    output: str | None = None,
) -> Path:
    return deps.write_perf_lines(
        _resolve_msprof_output_path(deps, operator_file, output=output),
        deps.render_perf_case_records_jsonl(
            case_records,
            missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
        ),
    )


def _resolve_msprof_output_path(
    deps: BenchRunnerDeps,
    operator_file: Path,
    *,
    output: str | None = None,
) -> Path:
    if output is not None:
        return Path(output).expanduser().resolve()
    return deps.perf_output_path(operator_file)


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


_STANDALONE_RUNTIME_DIR = str(Path(__file__).resolve().parent)


def _build_standalone_msprof_wrapper_script() -> str:
    return (
        "import importlib.util\n"
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        f"sys.path.insert(0, {_STANDALONE_RUNTIME_DIR!r})\n"
        "sys.path.insert(0, '.')\n"
        "\n"
        "\n"
        "def _load_module(module_path: pathlib.Path, module_name: str):\n"
        "    spec = importlib.util.spec_from_file_location(f'{module_name}_{time.time_ns()}', module_path)\n"
        "    if spec is None or spec.loader is None:\n"
        "        raise ImportError(f'Unable to load module from {module_path}')\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name] = module\n"
        "    try:\n"
        "        spec.loader.exec_module(module)\n"
        "    finally:\n"
        "        sys.modules.pop(spec.name, None)\n"
        "    return module\n"
        "\n"
        "\n"
        "bench_file = pathlib.Path(sys.argv[1]).resolve()\n"
        "operator_file = pathlib.Path(sys.argv[2]).resolve()\n"
        "case_id = sys.argv[3]\n"
        "bench_module = _load_module(bench_file, f'standalone_msprof_bench_{bench_file.stem}')\n"
        "operator_module = _load_module(operator_file, f'standalone_msprof_operator_{operator_file.stem}')\n"
        "build_operator_api = getattr(bench_module, 'build_operator_api')\n"
        "build_cases = getattr(bench_module, 'build_bench_cases')\n"
        "build_case_fn = getattr(bench_module, 'build_bench_case_fn')\n"
        "operator_api = build_operator_api(operator_module)\n"
        "raw_cases = list(build_cases())\n"
        "matching = [case for case in raw_cases if isinstance(case, dict) and case.get('id') == case_id]\n"
        "if not matching:\n"
        "    raise SystemExit(f'case_id {case_id!r} not found, available: "
        "        {[c.case_id for c in cases]!r}')\n"
        "case = matching[0]\n"
        "print(f'[standalone-msprof] case={case.case_id} repeats={case.repeats}', "
        "      flush=True)\n"
        "for i in range(case.repeats):\n"
        "    case.fn()\n"
        "    print(f'[standalone-msprof] repeat {i+1}/{case.repeats} done', flush=True)\n"
        "print('[standalone-msprof] done', flush=True)\n"
    )
