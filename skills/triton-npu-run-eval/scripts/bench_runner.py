from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Literal, TextIO, TypedDict, Union, cast

from run_runtime import (
    RemoteSpec,
    ResultPayload,
    cleanup_remote_workspace,
    copy_file_from_remote,
    copy_file_to_remote,
    create_remote_workspace,
    make_result,
    result_succeeded,
    run_buffered_process,
    run_remote_command_buffered,
    run_remote_command_streaming,
    run_streaming_process,
)

_LOCAL_MSPROF_OUTPUT_DIR_ENV = "TRITON_AGENT_MSPROF_OUTPUT_DIR"
_MISSING_KERNEL_MATCH_ERROR = "no resolved kernels matched op_statistic csv"


class MsprofAvgRow(TypedDict):
    op_type: str
    avg_time_us: float


class MsprofMetrics(TypedDict):
    kernel_avg_time_us: float | None
    ops: list[MsprofAvgRow]


ComparisonMode = Literal["latency", "total-op"]


@dataclass(frozen=True)
class PerfEntry:
    display_value: str
    numeric_value: float
    comparison_mode: ComparisonMode


class PerfValueMap(dict[str, float]):
    def __init__(
        self,
        values: dict[str, float],
        *,
        comparison_modes: dict[str, ComparisonMode],
    ) -> None:
        super().__init__(values)
        self.comparison_modes = comparison_modes


RequiredLatencyIds = Union[Collection[str], dict[str, PerfEntry], PerfValueMap]


@dataclass(frozen=True)
class KernelResolution:
    kernel_names: list[str]
    kernel_source: str


@dataclass(frozen=True)
class MsprofCaseRecord:
    case_idx: int
    kernel_names: list[str]
    kernel_source: str
    metrics: MsprofMetrics | None = None
    error_message: str | None = None


def parse_bench_metadata(bench_file: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in bench_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if metadata:
                break
            continue
        if not stripped.startswith("#"):
            break
        body = stripped[1:].strip()
        if ":" not in body:
            continue
        key, value = body.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    try:
        baseline_entries = _parse_perf_entries(baseline_perf)
        compare_entries = _parse_required_perf_entries(compare_perf, baseline_entries)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    baseline = {
        latency_id: entry.numeric_value for latency_id, entry in baseline_entries.items()
    }
    compare = {
        latency_id: entry.numeric_value for latency_id, entry in compare_entries.items()
    }
    baseline_ids = set(baseline)
    compare_ids = set(compare)
    if baseline_ids != compare_ids:
        missing = sorted(baseline_ids - compare_ids)
        extra = sorted(compare_ids - baseline_ids)
        details: list[str] = []
        if missing:
            details.append(f"missing in compare: {missing}")
        if extra:
            details.append(f"extra in compare: {extra}")
        print(f"FAIL: latency ids do not match ({'; '.join(details)})")
        return 1

    print("Perf comparison:")
    for latency_id in sorted(baseline):
        baseline_value = baseline[latency_id]
        compare_value = compare[latency_id]
        baseline_display = baseline_entries[latency_id].display_value
        compare_display = compare_entries[latency_id].display_value
        print(
            f"{latency_id}: baseline={baseline_display}, "
            f"compare={compare_display}, "
            f"delta={_format_delta_percent(baseline_value, compare_value)}"
        )
    avg_improvement, geomean_speedup, total_speedup = _summarize_perf_metrics(baseline, compare)
    print(f"Avg improvement: {_format_improvement_percent(avg_improvement)}")
    print(f"Geomean speedup: {_format_speedup(geomean_speedup)}")
    print(f"Total speedup: {_format_speedup(total_speedup)}")
    print(f"Metric source: {_summarize_metric_source(baseline_entries)}")
    print(f"PASS: compared {len(baseline)} latency entries")
    return 0


def parse_perf_file(path: Path) -> dict[str, float]:
    return _parse_perf_file(path)


def parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    return _parse_required_perf_file(path, required_latency_ids)


def run_local_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
) -> tuple[ResultPayload, Path | None]:
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


def _run_remote_bench_standalone(
    spec: RemoteSpec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[ResultPayload, Path | None, str]:
    helper_script = _standalone_runtime_script_path()
    copy_file_to_remote(
        spec,
        helper_script,
        f"{remote_workspace}/{helper_script.name}",
        verbose=verbose,
        stderr=stderr,
    )
    perf_path = _perf_output_path(bench_file, operator_file)
    result = run_remote_command_streaming(
        spec,
        remote_workspace,
        [
            "python3",
            helper_script.name,
            "run-all",
            "--bench-file",
            bench_file.name,
            "--operator-file",
            operator_file.name,
            "--perf-file",
            perf_path.name,
        ],
        verbose=verbose,
        stderr=stderr,
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
) -> tuple[ResultPayload, Path | None]:
    resolution = resolve_bench_kernel_resolution(bench_file, operator_file)
    count_result = run_buffered_process(
        [sys.executable, bench_file.name, "--num-bench"],
        str(bench_file.parent),
        stall_timeout_seconds=900,
    )
    if not result_succeeded(count_result):
        return count_result, None

    case_count = _parse_case_count(str(count_result["stdout"]))
    operator_arg = os.path.relpath(operator_file, bench_file.parent)
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_records: list[MsprofCaseRecord] = []
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
                sys.executable,
                bench_file.name,
                "--operator-file",
                operator_arg,
                "--bench",
                str(case_idx),
            ]
            result = run_streaming_process(command, str(bench_file.parent), stall_timeout_seconds=900)
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    MsprofCaseRecord(
                        case_idx=case_idx,
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
                    MsprofCaseRecord(
                        case_idx=case_idx,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                    )
                )
                continue

            case_records.append(
                MsprofCaseRecord(
                    case_idx=case_idx,
                    kernel_names=resolution.kernel_names,
                    kernel_source=resolution.kernel_source,
                    metrics=metrics,
                )
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    perf_path = _write_perf_lines(
        _perf_output_path(bench_file, operator_file),
        _render_msprof_case_records(case_records),
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
    )
    if not result_succeeded(count_result):
        return count_result, None, remote_workspace

    case_count = _parse_case_count(str(count_result["stdout"]))
    stdout_chunks = [str(count_result["stdout"])]
    stderr_chunks = [str(count_result["stderr"])]
    case_records: list[MsprofCaseRecord] = []
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
            )
            stdout_chunks.append(str(result["stdout"]))
            stderr_chunks.append(str(result["stderr"]))
            had_stalls = had_stalls or bool(result["stalled"])
            if result["session_id"] is not None:
                session_id = result["session_id"]
            if not result_succeeded(result):
                had_case_failures = True
                case_records.append(
                    MsprofCaseRecord(
                        case_idx=case_idx,
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
                    MsprofCaseRecord(
                        case_idx=case_idx,
                        kernel_names=resolution.kernel_names,
                        kernel_source=resolution.kernel_source,
                        error_message=str(exc),
                    )
                )
                continue

            case_records.append(
                MsprofCaseRecord(
                    case_idx=case_idx,
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

    perf_path = _write_perf_lines(
        _perf_output_path(bench_file, operator_file),
        _render_msprof_case_records(case_records),
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


def _perf_output_path(bench_file: Path, operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"


def _write_perf_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _parse_case_count(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ValueError("Unable to parse benchmark case count from --num-bench output.")


def resolve_bench_kernel_names(
    bench_file: Path,
    operator_file: Path | None = None,
) -> list[str]:
    return resolve_bench_kernel_resolution(bench_file, operator_file).kernel_names


def resolve_bench_kernel_resolution(
    bench_file: Path,
    operator_file: Path | None = None,
) -> KernelResolution:
    metadata = parse_bench_metadata(bench_file)
    metadata_kernel_names = _parse_kernel_names(metadata, bench_file, allow_empty=True)
    operator_kernel_names = (
        _discover_operator_triton_kernels(operator_file) if operator_file is not None else []
    )
    kernel_names = _stable_kernel_union(metadata_kernel_names, operator_kernel_names)
    if not kernel_names:
        raise ValueError(
            f"Benchmark metadata and operator file did not resolve any Triton kernels: {bench_file}"
        )
    return KernelResolution(
        kernel_names=kernel_names,
        kernel_source=_describe_kernel_source(metadata_kernel_names, operator_kernel_names),
    )


def _parse_kernel_names(
    metadata: dict[str, str],
    bench_file: Path,
    *,
    allow_empty: bool = False,
) -> list[str]:
    kernels_value = metadata.get("kernels")
    if kernels_value is not None:
        kernel_names = [part.strip() for part in kernels_value.split(",") if part.strip()]
    else:
        kernel_name = (metadata.get("kernel") or "").strip()
        kernel_names = [kernel_name] if kernel_name else []
    if not kernel_names and not allow_empty:
        raise ValueError(
            f"Benchmark metadata is missing required 'kernels' entry: {bench_file}"
        )
    return kernel_names


def _discover_operator_triton_kernels(operator_file: Path) -> list[str]:
    try:
        tree = ast.parse(operator_file.read_text(encoding="utf-8"), filename=str(operator_file))
    except SyntaxError as exc:
        raise ValueError(f"Failed to parse operator file for Triton kernels: {operator_file}") from exc
    kernels: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            _is_triton_jit_decorator(decorator) for decorator in node.decorator_list
        ):
            kernels.append(node.name)
    return kernels


def _is_triton_jit_decorator(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        return _is_triton_jit_decorator(node.func)
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr == "jit"
    return isinstance(node, ast.Name) and node.id == "jit"


def _stable_kernel_union(primary: list[str], secondary: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for kernel_name in [*primary, *secondary]:
        if kernel_name in seen:
            continue
        seen.add(kernel_name)
        merged.append(kernel_name)
    return merged


def _describe_kernel_source(metadata_kernels: list[str], operator_kernels: list[str]) -> str:
    if metadata_kernels and operator_kernels:
        return "metadata+operator"
    if metadata_kernels:
        return "metadata"
    return "operator"


def _load_msprof_avg_rows(output_dir: Path) -> list[MsprofAvgRow]:
    csv_path = _find_latest_msprof_statistic_csv(output_dir)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Avg Time(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Avg Time(us)' in {csv_path}")
        if "OP Type" not in fieldnames:
            raise ValueError(f"Missing required column 'OP Type' in {csv_path}")
        rows: list[MsprofAvgRow] = []
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


def _format_latency_value(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        rendered += ".0"
    return rendered


def _resolve_msprof_metrics(
    rows: list[MsprofAvgRow],
    kernel_names: list[str],
) -> MsprofMetrics:
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


def _read_local_msprof_metrics(output_dir: Path, kernel_names: list[str]) -> MsprofMetrics:
    return _resolve_msprof_metrics(_load_msprof_avg_rows(output_dir), kernel_names)


def _create_local_msprof_preserved_run_dir() -> Path | None:
    configured_root = os.environ.get(_LOCAL_MSPROF_OUTPUT_DIR_ENV)
    if not configured_root:
        return None
    root = Path(configured_root).expanduser()
    if root.exists() and not root.is_dir():
        raise ValueError(
            f"{_LOCAL_MSPROF_OUTPUT_DIR_ENV} must point to a directory: {root}"
        )
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        _set_directory_owner_only(root)
    run_dir = Path(tempfile.mkdtemp(prefix="triton-agent-msprof-", dir=str(root)))
    _set_directory_owner_only(run_dir)
    return run_dir


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


def _render_msprof_case_records(records: list[MsprofCaseRecord]) -> list[str]:
    rendered: list[str] = []
    for record in records:
        rendered.extend(_render_msprof_case_record(record))
    return rendered


def _render_msprof_case_record(record: MsprofCaseRecord) -> list[str]:
    lines = [f"latency-case-{record.case_idx}: {_format_case_latency_value(record)}"]
    if record.metrics is not None:
        raw_payload = json.dumps({"ops": record.metrics["ops"]}, separators=(",", ":"))
        lines.append(f"# raw-op-statistic-case-{record.case_idx}: {raw_payload}")
        if record.metrics["kernel_avg_time_us"] is None:
            lines.append(f"# latency-error-case-{record.case_idx}: {_MISSING_KERNEL_MATCH_ERROR}")
    if record.error_message is not None:
        lines.append(f"# latency-error-case-{record.case_idx}: {record.error_message}")
    lines.append(f"# resolved-kernels-case-{record.case_idx}: {','.join(record.kernel_names)}")
    lines.append(f"# kernel-source-case-{record.case_idx}: {record.kernel_source}")
    return lines


def _format_case_latency_value(record: MsprofCaseRecord) -> str:
    if record.metrics is None or record.metrics["kernel_avg_time_us"] is None:
        return "NA"
    return _format_latency_value(record.metrics["kernel_avg_time_us"])


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
) -> MsprofMetrics:
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


def _parse_perf_file(path: Path) -> dict[str, float]:
    entries = _parse_perf_entries(path)
    return PerfValueMap(
        {latency_id: entry.numeric_value for latency_id, entry in entries.items()},
        comparison_modes={latency_id: entry.comparison_mode for latency_id, entry in entries.items()},
    )


def _parse_perf_entries(path: Path) -> dict[str, PerfEntry]:
    lines = path.read_text(encoding="utf-8").splitlines()
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
    latency_errors = _parse_latency_errors(path, lines)
    entries: dict[str, PerfEntry] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no} is not a 'latency-<id>: <value>' line")
        key, value = line.split(":", 1)
        latency_id = key.strip()
        if not latency_id.startswith("latency-"):
            raise ValueError(f"{path}:{line_no} does not start with 'latency-'")
        value_text = value.strip()
        if latency_id in entries:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{latency_id}'")
        _raise_for_uncomparable_latency_error(path, line_no, latency_id, latency_errors)
        if value_text == "NA":
            total_op_value = _require_raw_total(path, line_no, latency_id, raw_totals)
            entries[latency_id] = PerfEntry(
                display_value=f"NA ({_format_total_op_display(total_op_value)})",
                numeric_value=total_op_value,
                comparison_mode="total-op",
            )
            continue
        try:
            parsed_value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc
        entries[latency_id] = PerfEntry(
            display_value=value_text,
            numeric_value=parsed_value,
            comparison_mode="latency",
        )
    if not entries:
        raise ValueError(f"{path} did not contain any latency-<id>: <value> entries")
    return entries


def _parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    return {
        latency_id: entry.numeric_value
        for latency_id, entry in _parse_required_perf_entries(
            path, required_latency_ids
        ).items()
    }


def _parse_required_perf_entries(
    path: Path, required_latency_ids: RequiredLatencyIds
) -> dict[str, PerfEntry]:
    required_ids, comparison_modes = _resolve_required_latency_requirements(required_latency_ids)
    if not required_ids:
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
    latency_errors = _parse_latency_errors(path, lines)
    entries: dict[str, PerfEntry] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        latency_id = key.strip()
        if latency_id not in required_ids:
            continue
        value_text = value.strip()
        if latency_id in entries:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{latency_id}'")
        _raise_for_uncomparable_latency_error(path, line_no, latency_id, latency_errors)
        if comparison_modes[latency_id] == "total-op":
            total_op_value = _require_raw_total(path, line_no, latency_id, raw_totals)
            display_value = (
                f"NA ({_format_total_op_display(total_op_value)})"
                if value_text == "NA"
                else _format_total_op_display(total_op_value)
            )
            entries[latency_id] = PerfEntry(
                display_value=display_value,
                numeric_value=total_op_value,
                comparison_mode="total-op",
            )
            continue
        if value_text == "NA":
            raise ValueError(
                f"{path}:{line_no} has latency value 'NA' but baseline requires kernel latency"
            )
        try:
            parsed_value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc
        entries[latency_id] = PerfEntry(
            display_value=value_text,
            numeric_value=parsed_value,
            comparison_mode="latency",
        )

    missing_ids = sorted(required_ids - set(entries))
    if missing_ids:
        raise ValueError(f"{path} is missing required latency ids: {missing_ids}")
    return entries


def _parse_raw_op_statistic_totals(path: Path, lines: list[str]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line.startswith("# raw-op-statistic-"):
            continue
        body = line[1:].strip()
        if ":" not in body:
            raise ValueError(f"{path}:{line_no} is not a '# raw-op-statistic-<id>: <json>' line")
        key, value = body.split(":", 1)
        raw_stat_id = key.strip()
        latency_id = f"latency-{raw_stat_id.removeprefix('raw-op-statistic-')}"
        if latency_id in totals:
            raise ValueError(f"{path}:{line_no} duplicates raw-op statistic for '{latency_id}'")
        try:
            payload = json.loads(value.strip())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} has invalid raw-op-statistic JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} raw-op-statistic JSON must be an object")
        payload_dict = cast(dict[str, object], payload)
        ops = payload_dict.get("ops")
        if not isinstance(ops, list):
            raise ValueError(f"{path}:{line_no} raw-op-statistic JSON is missing an 'ops' list")
        total = 0.0
        typed_ops = cast(list[object], ops)
        for op in typed_ops:
            if not isinstance(op, dict):
                raise ValueError(f"{path}:{line_no} raw-op-statistic ops entries must be objects")
            op_dict = cast(dict[str, object], op)
            avg_time_us = op_dict.get("avg_time_us")
            if not isinstance(avg_time_us, (int, float)):
                raise ValueError(
                    f"{path}:{line_no} raw-op-statistic ops entries must include numeric 'avg_time_us'"
                )
            total += float(avg_time_us)
        totals[latency_id] = total
    return totals


def _parse_latency_errors(path: Path, lines: list[str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line.startswith("# latency-error-"):
            continue
        body = line[1:].strip()
        if ":" not in body:
            raise ValueError(f"{path}:{line_no} is not a '# latency-error-<id>: <message>' line")
        key, value = body.split(":", 1)
        error_id = key.strip()
        latency_id = f"latency-{error_id.removeprefix('latency-error-')}"
        if latency_id in errors:
            raise ValueError(f"{path}:{line_no} duplicates latency error for '{latency_id}'")
        errors[latency_id] = value.strip()
    return errors


def _resolve_required_latency_requirements(
    required_latency_ids: RequiredLatencyIds,
) -> tuple[set[str], dict[str, ComparisonMode]]:
    required_ids = set(required_latency_ids)
    comparison_modes: dict[str, ComparisonMode] = {
        latency_id: "latency" for latency_id in required_ids
    }
    if isinstance(required_latency_ids, dict):
        typed_required_latency_ids = cast(dict[str, PerfEntry] | PerfValueMap, required_latency_ids)
        for latency_id in required_ids:
            value = typed_required_latency_ids[latency_id]
            if isinstance(value, PerfEntry):
                comparison_modes[latency_id] = value.comparison_mode
        return required_ids, comparison_modes
    raw_modes = (
        required_latency_ids.comparison_modes
        if isinstance(required_latency_ids, PerfValueMap)
        else None
    )
    if raw_modes is not None:
        for latency_id in required_ids:
            mode = raw_modes.get(latency_id)
            if mode in ("latency", "total-op"):
                comparison_modes[latency_id] = mode
    return required_ids, comparison_modes


def _require_raw_total(
    path: Path,
    line_no: int,
    latency_id: str,
    raw_totals: dict[str, float],
) -> float:
    total = raw_totals.get(latency_id)
    if total is None:
        raise ValueError(
            f"{path}:{line_no} requires '# raw-op-statistic-{latency_id.removeprefix('latency-')}: ...' to provide total-op fallback"
        )
    return total


def _raise_for_uncomparable_latency_error(
    path: Path,
    line_no: int,
    latency_id: str,
    latency_errors: dict[str, str],
) -> None:
    error_message = latency_errors.get(latency_id)
    if error_message is None or error_message == _MISSING_KERNEL_MATCH_ERROR:
        return
    raise ValueError(
        f"{path}:{line_no} cannot compare '{latency_id}' because '# latency-error-{latency_id.removeprefix('latency-')}: {error_message}' is present"
    )


def _format_total_op_display(value: float) -> str:
    return f"total-op={_format_latency_value(value)}"


def _format_delta_percent(baseline: float, compare: float) -> str:
    if baseline == 0:
        if compare == 0:
            return "0.00%"
        return "inf"
    delta = ((compare - baseline) / baseline) * 100.0
    return f"{delta:.2f}%"


def _summarize_perf_metrics(
    baseline: dict[str, float],
    compare: dict[str, float],
) -> tuple[float | None, float | None, float | None]:
    pairs = [(baseline[latency_id], compare[latency_id]) for latency_id in sorted(baseline)]
    if not pairs:
        return None, None, None
    if any(baseline_value <= 0 or compare_value <= 0 for baseline_value, compare_value in pairs):
        return None, None, None

    improvements = [
        (baseline_value - compare_value) / baseline_value
        for baseline_value, compare_value in pairs
    ]
    ratios = [baseline_value / compare_value for baseline_value, compare_value in pairs]
    avg_improvement = sum(improvements) / len(improvements)
    geomean_speedup = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    total_speedup = sum(baseline_value for baseline_value, _ in pairs) / sum(
        compare_value for _, compare_value in pairs
    )
    return avg_improvement, geomean_speedup, total_speedup


def _format_improvement_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100:+.1f}%"


def _format_speedup(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}x"


def _summarize_metric_source(entries: dict[str, PerfEntry]) -> str:
    modes = {entry.comparison_mode for entry in entries.values()}
    if modes == {"latency"}:
        return "kernel"
    if modes == {"total-op"}:
        return "total-op"
    return "mixed (kernel + total-op fallback)"
