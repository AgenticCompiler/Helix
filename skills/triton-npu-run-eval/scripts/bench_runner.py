from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import TextIO, cast

from bench_contract import (
    KernelResolution,
    parse_bench_metadata as _parse_bench_metadata,
    resolve_bench_kernel_names as _resolve_bench_kernel_names,
    resolve_bench_kernel_resolution as _resolve_bench_kernel_resolution,
)
from perf_artifacts import (
    PerfCaseRecord,
    PerfMetrics,
    PerfOpRow,
    RequiredLatencyIds,
    compare_perf_files as _compare_perf_files,
    parse_perf_file as _parse_perf_file,
    parse_required_perf_file as _parse_required_perf_file,
    perf_output_path,
    render_perf_case_records,
    write_perf_lines,
)

from run_runtime import (
    RemoteSpec,
    ResultPayload,
    env_int,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    local_python_executable,
    make_result,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)

_LOCAL_BENCH_PROFILE_OUTPUT_DIR_ENV = "TRITON_AGENT_BENCH_PROFILE_OUTPUT_DIR"
_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched op_statistic csv"


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


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    return _compare_perf_files(baseline_perf, compare_perf)


def parse_perf_file(path: Path) -> dict[str, float]:
    return _parse_perf_file(path)


def parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    return _parse_required_perf_file(path, required_latency_ids)


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
) -> tuple[ResultPayload, Path | None]:
    with _local_bench_workdir(bench_file.parent):
        if bench_mode == "msprof":
            return _run_local_bench_msprof(bench_file, operator_file)
        return _run_local_bench_standalone(bench_file, operator_file)


def run_remote_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
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
            return _run_remote_bench_msprof(
                spec,
                remote_workspace,
                bench_file,
                operator_file,
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
) -> tuple[ResultPayload, Path | None]:
    return run_local_standalone_bench(bench_file, operator_file)


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


def _run_remote_bench_standalone(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    for support_path in _standalone_runtime_support_paths():
        copy_file_to_remote(
            spec,
            support_path,
            f"{remote_workspace}/{support_path.name}",
            verbose=verbose,
            stderr=stderr,
        )
    perf_path = perf_output_path(operator_file)
    with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "python3",
                "-c",
                _build_remote_standalone_run_all_script(),
                bench_file.name,
                operator_file.name,
                perf_path.name,
            ],
            stdout=quiet_stdout,
            verbose=verbose,
            stderr=stderr,
            stall_timeout_seconds=_bench_timeout(),
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


def run_local_standalone_bench(
    bench_file: Path,
    operator_file: Path,
) -> tuple[ResultPayload, Path]:
    runtime = _load_standalone_runtime_module()
    return runtime.run_local_standalone_bench(bench_file, operator_file)


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


def _build_remote_standalone_run_all_script() -> str:
    return (
        "import pathlib, shutil, sys; "
        "import standalone_bench_runtime as runtime; "
        "bench_file = pathlib.Path(sys.argv[1]); "
        "operator_file = pathlib.Path(sys.argv[2]); "
        "target_path = pathlib.Path(sys.argv[3]); "
        "result, perf_path = runtime.run_local_standalone_bench(bench_file, operator_file); "
        "target_path.parent.mkdir(parents=True, exist_ok=True); "
        "shutil.copyfile(perf_path, target_path) if perf_path != target_path else None; "
        "raise SystemExit(int(result['return_code']))"
    )


def _run_local_bench_msprof(
    bench_file: Path,
    operator_file: Path,
) -> tuple[ResultPayload, Path | None]:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = run_buffered_process(
        [local_python_executable(), bench_file.name, "--num-bench"],
        str(bench_file.parent),
        stall_timeout_seconds=_bench_timeout(),
    )
    if not result_succeeded(count_result):
        return count_result, None

    case_count = _parse_case_count(str(count_result["stdout"]))
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_records: list[PerfCaseRecord] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    for case_idx in range(1, case_count + 1):
        output_dir, temp_dir = _create_local_msprof_output_dir(case_idx, preserved_run_dir)
        try:
            command = [
                "msprof",
                f"--output={output_dir}",
                local_python_executable(),
                bench_file.name,
                "--operator-file",
                operator_arg,
                "--bench",
                str(case_idx),
            ]
            with open(os.devnull, "w", encoding="utf-8") as quiet_stdout:
                result = run_streaming_process(
                    command,
                    str(bench_file.parent),
                    stall_timeout_seconds=_bench_timeout(),
                    stdout=quiet_stdout,
                )
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=str(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=_format_msprof_command_failure(result),
                    ),
                )
                continue

            try:
                metrics = _read_local_msprof_metrics(output_dir, resolution.kernel_names)
            except (FileNotFoundError, ValueError) as exc:
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=str(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=str(case_idx),
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                )
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
            _cleanup_local_bench_extra_info(bench_file.parent)

    perf_path = write_perf_lines(
        perf_output_path(operator_file),
        render_perf_case_records(
            case_records,
            latency_prefix="latency-case",
            raw_prefix="raw-op-statistic-case",
            resolved_kernels_prefix="resolved-kernels-case",
            kernel_source_prefix="kernel-source-case",
            latency_error_prefix="latency-error-case",
            missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
        ),
    )
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


def _run_remote_bench_msprof(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = run_remote_command_buffered(
        spec,
        remote_workspace,
        ["python3", bench_file.name, "--num-bench"],
        verbose=verbose,
        stderr=stderr,
        stall_timeout_seconds=_bench_timeout(),
    )
    if not result_succeeded(count_result):
        return count_result, None, remote_workspace

    case_count = _parse_case_count(str(count_result["stdout"]))
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_records: list[PerfCaseRecord] = []
    had_case_failures = False
    had_stalls = False
    session_id: str | None = None

    for case_idx in range(1, case_count + 1):
        output_dir = _create_remote_msprof_output_dir(
            spec,
            remote_workspace,
            verbose=verbose,
            stderr=stderr,
        )
        try:
            result = run_remote_command_streaming(
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
                stall_timeout_seconds=_bench_timeout(),
            )
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    PerfCaseRecord(
                        case_label=str(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=_format_msprof_command_failure(result),
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
                        case_label=str(case_idx),
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                    )
                )
                continue

            case_records.append(
                PerfCaseRecord(
                    case_label=str(case_idx),
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
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

    perf_path = write_perf_lines(
        perf_output_path(operator_file),
        render_perf_case_records(
            case_records,
            latency_prefix="latency-case",
            raw_prefix="raw-op-statistic-case",
            resolved_kernels_prefix="resolved-kernels-case",
            kernel_source_prefix="kernel-source-case",
            latency_error_prefix="latency-error-case",
            missing_kernel_match_error=_MISSING_KERNEL_MATCH_ERROR,
        ),
    )
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


def _parse_case_count(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ValueError("Unable to parse benchmark case count from --num-bench output.")


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
