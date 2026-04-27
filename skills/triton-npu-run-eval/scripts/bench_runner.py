from __future__ import annotations

import csv
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from run_runtime import (
    ResultPayload,
    cleanup_remote_workspace,
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


def parse_required_perf_file(path: Path, required_latency_ids) -> dict[str, float]:
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
    stderr=None,
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
    command = [sys.executable, str(bench_file), "--operator-file", str(operator_file)]
    result = run_streaming_process(command, str(bench_file.parent), stall_timeout_seconds=900)
    if not result_succeeded(result):
        return result, None
    perf_path = _write_perf_lines(
        _perf_output_path(bench_file, operator_file),
        _extract_latency_lines(f"{result['stdout']}\n{result['stderr']}"),
    )
    return result, perf_path


def _run_remote_bench_standalone(
    spec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr=None,
) -> tuple[ResultPayload, Path | None, str]:
    result = run_remote_command_streaming(
        spec,
        remote_workspace,
        ["python3", bench_file.name, "--operator-file", operator_file.name],
        verbose=verbose,
        stderr=stderr,
    )
    if not result_succeeded(result):
        return result, None, remote_workspace
    perf_path = _write_perf_lines(
        _perf_output_path(bench_file, operator_file),
        _extract_latency_lines(f"{result['stdout']}\n{result['stderr']}"),
    )
    return result, perf_path, remote_workspace


def _run_local_bench_msprof(
    bench_file: Path,
    operator_file: Path,
) -> tuple[ResultPayload, Path | None]:
    kernel_name = _resolve_kernel_name(bench_file)
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
    normalized_lines: list[str] = []
    preserved_run_dir = _create_local_msprof_preserved_run_dir()

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
            if not result_succeeded(result):
                return (
                    make_result(
                        return_code=int(result["return_code"]),
                        stdout="".join(stdout_chunks),
                        stderr="".join(stderr_chunks),
                        stalled=bool(result["stalled"]),
                        session_id=result["session_id"],
                    ),
                    None,
                )

            metrics = _read_local_msprof_metrics(output_dir, kernel_name)
            normalized_lines.extend(_format_msprof_perf_lines(case_idx, metrics))
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    perf_path = _write_perf_lines(_perf_output_path(bench_file, operator_file), normalized_lines)
    return (make_result(return_code=0, stdout="".join(stdout_chunks), stderr="".join(stderr_chunks)), perf_path)


def _run_remote_bench_msprof(
    spec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr=None,
) -> tuple[ResultPayload, Path | None, str]:
    kernel_name = _resolve_kernel_name(bench_file)
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
    normalized_lines: list[str] = []

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
            if not result_succeeded(result):
                return (
                    make_result(
                        return_code=int(result["return_code"]),
                        stdout="".join(stdout_chunks),
                        stderr="".join(stderr_chunks),
                        stalled=bool(result["stalled"]),
                        session_id=result["session_id"],
                    ),
                    None,
                    remote_workspace,
                )

            metrics = _read_remote_msprof_metrics(
                spec,
                remote_workspace,
                output_dir,
                kernel_name,
                verbose=verbose,
                stderr=stderr,
            )
            normalized_lines.extend(_format_msprof_perf_lines(case_idx, metrics))
        finally:
            _cleanup_remote_msprof_output_dir(
                spec,
                remote_workspace,
                output_dir,
                verbose=verbose,
                stderr=stderr,
            )

    perf_path = _write_perf_lines(_perf_output_path(bench_file, operator_file), normalized_lines)
    return (
        make_result(return_code=0, stdout="".join(stdout_chunks), stderr="".join(stderr_chunks)),
        perf_path,
        remote_workspace,
    )


def _perf_output_path(bench_file: Path, operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"


def _extract_latency_lines(output: str) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip().startswith("latency-")]
    if not lines:
        raise FileNotFoundError("Benchmark output did not contain any latency-<id> lines.")
    return lines


def _write_perf_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _parse_case_count(stdout: str) -> int:
    for line in reversed(stdout.splitlines()):
        stripped = line.strip()
        if stripped.isdigit():
            return int(stripped)
    raise ValueError("Unable to parse benchmark case count from --num-bench output.")


def _resolve_kernel_name(bench_file: Path) -> str:
    metadata = parse_bench_metadata(bench_file)
    kernel_name = metadata.get("kernel")
    if not kernel_name:
        raise ValueError(f"Benchmark metadata is missing required 'kernel' entry: {bench_file}")
    return kernel_name


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
    kernel_name: str,
) -> MsprofMetrics:
    kernel_avg_time_us: float | None = None
    for row in rows:
        if str(row["op_type"]) == kernel_name:
            kernel_avg_time_us = float(row["avg_time_us"])
            break
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


def _read_local_msprof_metrics(output_dir: Path, kernel_name: str) -> MsprofMetrics:
    return _resolve_msprof_metrics(_load_msprof_avg_rows(output_dir), kernel_name)


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


def _format_msprof_perf_lines(case_idx: int, metrics: MsprofMetrics) -> list[str]:
    raw_payload = json.dumps({"ops": metrics["ops"]}, separators=(",", ":"))
    latency_value = (
        "NA"
        if metrics["kernel_avg_time_us"] is None
        else _format_latency_value(metrics["kernel_avg_time_us"])
    )
    return [
        f"latency-case-{case_idx}: {latency_value}",
        f"# raw-op-statistic-case-{case_idx}: {raw_payload}",
    ]


def _create_remote_msprof_output_dir(
    spec,
    remote_workspace: str,
    verbose: bool = False,
    stderr=None,
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
    spec,
    remote_workspace: str,
    output_dir: str,
    kernel_name: str,
    verbose: bool = False,
    stderr=None,
) -> MsprofMetrics:
    script = """
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
kernel_name = sys.argv[2]
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
kernel_avg_time_us = None
for row in ops:
    if row["op_type"] == kernel_name:
        kernel_avg_time_us = row["avg_time_us"]
        break
print(json.dumps({"kernel_avg_time_us": kernel_avg_time_us, "ops": ops}, separators=(",", ":")))
""".strip()
    result = run_remote_command_buffered(
        spec,
        remote_workspace,
        ["python3", "-c", script, output_dir, kernel_name],
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
    spec,
    remote_workspace: str,
    output_dir: str,
    verbose: bool = False,
    stderr=None,
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


def _parse_required_perf_file(path: Path, required_latency_ids) -> dict[str, float]:
    return {
        latency_id: entry.numeric_value
        for latency_id, entry in _parse_required_perf_entries(
            path, required_latency_ids
        ).items()
    }


def _parse_required_perf_entries(
    path: Path, required_latency_ids
) -> dict[str, PerfEntry]:
    required_ids, comparison_modes = _resolve_required_latency_requirements(required_latency_ids)
    if not required_ids:
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
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
        ops = payload.get("ops")
        if not isinstance(ops, list):
            raise ValueError(f"{path}:{line_no} raw-op-statistic JSON is missing an 'ops' list")
        total = 0.0
        for op in ops:
            if not isinstance(op, dict):
                raise ValueError(f"{path}:{line_no} raw-op-statistic ops entries must be objects")
            avg_time_us = op.get("avg_time_us")
            if not isinstance(avg_time_us, (int, float)):
                raise ValueError(
                    f"{path}:{line_no} raw-op-statistic ops entries must include numeric 'avg_time_us'"
                )
            total += float(avg_time_us)
        totals[latency_id] = total
    return totals


def _resolve_required_latency_requirements(
    required_latency_ids,
) -> tuple[set[str], dict[str, ComparisonMode]]:
    required_ids = set(required_latency_ids)
    comparison_modes: dict[str, ComparisonMode] = {
        latency_id: "latency" for latency_id in required_ids
    }
    if isinstance(required_latency_ids, dict):
        for latency_id in required_ids:
            value = required_latency_ids[latency_id]
            if isinstance(value, PerfEntry):
                comparison_modes[latency_id] = value.comparison_mode
        return required_ids, comparison_modes
    raw_modes = getattr(required_latency_ids, "comparison_modes", None)
    if isinstance(raw_modes, dict):
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
