from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

from run_runtime import (
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
        compare_entries = _parse_required_perf_entries(compare_perf, baseline_entries.keys())
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    baseline = {
        latency_id: value for latency_id, (_display_value, value) in baseline_entries.items()
    }
    compare = {
        latency_id: value for latency_id, (_display_value, value) in compare_entries.items()
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
        baseline_display = baseline_entries[latency_id][0]
        compare_display = compare_entries[latency_id][0]
        print(
            f"{latency_id}: baseline={baseline_display}, "
            f"compare={compare_display}, "
            f"delta={_format_delta_percent(baseline_value, compare_value)}"
        )
    avg_improvement, geomean_speedup, total_speedup = _summarize_perf_metrics(baseline, compare)
    print(f"Avg improvement: {_format_improvement_percent(avg_improvement)}")
    print(f"Geomean speedup: {_format_speedup(geomean_speedup)}")
    print(f"Total speedup: {_format_speedup(total_speedup)}")
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
) -> tuple[dict[str, object], Path | None]:
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
) -> tuple[dict[str, object], Path | None, str]:
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
) -> tuple[dict[str, object], Path | None]:
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
) -> tuple[dict[str, object], Path | None, str]:
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
) -> tuple[dict[str, object], Path | None]:
    metadata = parse_bench_metadata(bench_file)
    kernel_name = metadata.get("kernel")
    if not kernel_name:
        raise ValueError(f"Benchmark metadata is missing required 'kernel' entry: {bench_file}")

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

    for case_idx in range(1, case_count + 1):
        command = [
            "msprof",
            "op",
            f"--kernel-name={kernel_name}",
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

        duration = _extract_msprof_duration(f"{result['stdout']}\n{result['stderr']}")
        normalized_lines.append(f"latency-case-{case_idx}: {duration}")

    perf_path = _write_perf_lines(_perf_output_path(bench_file, operator_file), normalized_lines)
    return (make_result(return_code=0, stdout="".join(stdout_chunks), stderr="".join(stderr_chunks)), perf_path)


def _run_remote_bench_msprof(
    spec,
    remote_workspace: str,
    bench_file: Path,
    operator_file: Path,
    verbose: bool = False,
    stderr=None,
) -> tuple[dict[str, object], Path | None, str]:
    metadata = parse_bench_metadata(bench_file)
    kernel_name = metadata.get("kernel")
    if not kernel_name:
        raise ValueError(f"Benchmark metadata is missing required 'kernel' entry: {bench_file}")

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
        result = run_remote_command_streaming(
            spec,
            remote_workspace,
            [
                "msprof",
                "op",
                f"--kernel-name={kernel_name}",
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

        duration = _extract_msprof_duration(f"{result['stdout']}\n{result['stderr']}")
        normalized_lines.append(f"latency-case-{case_idx}: {duration}")

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


def _extract_msprof_duration(output: str) -> str:
    match = re.search(r"Task Duration\(us\):\s*([0-9]+(?:\.[0-9]+)?)", output)
    if not match:
        raise FileNotFoundError("msprof output did not contain Task Duration(us).")
    return match.group(1)


def _parse_perf_file(path: Path) -> dict[str, float]:
    return {latency_id: value for latency_id, (_display_value, value) in _parse_perf_entries(path).items()}


def _parse_perf_entries(path: Path) -> dict[str, tuple[str, float]]:
    values: dict[str, float] = {}
    display_values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_no} is not a 'latency-<id>: <value>' line")
        key, value = line.split(":", 1)
        latency_id = key.strip()
        if not latency_id.startswith("latency-"):
            raise ValueError(f"{path}:{line_no} does not start with 'latency-'")
        value_text = value.strip()
        try:
            parsed_value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc
        if latency_id in values:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{latency_id}'")
        values[latency_id] = parsed_value
        display_values[latency_id] = value_text
    if not values:
        raise ValueError(f"{path} did not contain any latency-<id>: <value> entries")
    return {
        latency_id: (display_values[latency_id], values[latency_id]) for latency_id in values
    }


def _parse_required_perf_file(path: Path, required_latency_ids) -> dict[str, float]:
    return {
        latency_id: value
        for latency_id, (_display_value, value) in _parse_required_perf_entries(
            path, required_latency_ids
        ).items()
    }


def _parse_required_perf_entries(
    path: Path, required_latency_ids
) -> dict[str, tuple[str, float]]:
    required_ids = set(required_latency_ids)
    if not required_ids:
        return {}

    values: dict[str, float] = {}
    display_values: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        latency_id = key.strip()
        if latency_id not in required_ids:
            continue
        value_text = value.strip()
        try:
            parsed_value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc
        if latency_id in values:
            raise ValueError(f"{path}:{line_no} duplicates latency id '{latency_id}'")
        values[latency_id] = parsed_value
        display_values[latency_id] = value_text

    missing_ids = sorted(required_ids - set(values))
    if missing_ids:
        raise ValueError(f"{path} is missing required latency ids: {missing_ids}")
    return {
        latency_id: (display_values[latency_id], values[latency_id]) for latency_id in values
    }


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
