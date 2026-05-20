from __future__ import annotations

import json
import math
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, Union, cast


class PerfOpRow(TypedDict):
    op_type: str
    avg_time_us: float


class PerfMetrics(TypedDict):
    kernel_avg_time_us: float | None
    ops: list[PerfOpRow]


@dataclass(frozen=True)
class PerfCaseRecord:
    case_label: str
    kernel_names: list[str]
    kernel_source: str
    metrics: PerfMetrics | None = None
    error_message: str | None = None
    elapsed_seconds: float | None = None


ComparisonMode = Literal["latency", "total-op"]
MetricSource = Literal["auto", "kernel", "total-op"]


@dataclass(frozen=True)
class PerfEntry:
    display_value: str
    numeric_value: float
    comparison_mode: ComparisonMode


@dataclass(frozen=True)
class PerfParseOutcome:
    entries: dict[str, PerfEntry]
    skipped_latency_errors: dict[str, str]


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


def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: MetricSource = "auto",
) -> int:
    try:
        baseline_outcome = _parse_perf_entries_for_comparison(
            baseline_perf,
            skip_latency_errors=skip_latency_errors,
            metric_source=metric_source,
        )
        compare_outcome = _parse_required_perf_entries_for_comparison(
            compare_perf,
            baseline_outcome.entries,
            skip_latency_errors=skip_latency_errors,
            metric_source=metric_source,
        )
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    comparable_ids = sorted(set(baseline_outcome.entries) & set(compare_outcome.entries))
    baseline = {
        latency_id: baseline_outcome.entries[latency_id].numeric_value
        for latency_id in comparable_ids
    }
    compare = {
        latency_id: compare_outcome.entries[latency_id].numeric_value
        for latency_id in comparable_ids
    }
    print("Perf comparison:")
    for latency_id in comparable_ids:
        baseline_value = baseline[latency_id]
        compare_value = compare[latency_id]
        baseline_display = baseline_outcome.entries[latency_id].display_value
        compare_display = compare_outcome.entries[latency_id].display_value
        print(
            f"{latency_id}: baseline={baseline_display}, "
            f"compare={compare_display}, "
            f"delta={_format_delta_percent(baseline_value, compare_value)}"
        )
    avg_improvement, geomean_speedup, total_speedup = _summarize_perf_metrics(baseline, compare)
    print(f"Avg improvement: {_format_improvement_percent(avg_improvement)}")
    print(f"Geomean speedup: {_format_speedup(geomean_speedup)}")
    print(f"Total speedup: {_format_speedup(total_speedup)}")
    compared_entries = {
        latency_id: baseline_outcome.entries[latency_id] for latency_id in comparable_ids
    }
    print(f"Metric source: {_summarize_metric_source(compared_entries, metric_source=metric_source)}")
    skipped_latency_errors = {
        **baseline_outcome.skipped_latency_errors,
        **compare_outcome.skipped_latency_errors,
    }
    if skipped_latency_errors:
        print(f"FAIL: skipped {len(skipped_latency_errors)} latency entries due to latency errors")
        for latency_id in sorted(skipped_latency_errors):
            print(skipped_latency_errors[latency_id])
        return 1
    print(f"PASS: compared {len(baseline)} latency entries")
    return 0


def parse_perf_file(path: Path) -> dict[str, float]:
    return _parse_perf_file(path)


def parse_required_perf_file(path: Path, required_latency_ids: RequiredLatencyIds) -> dict[str, float]:
    return _parse_required_perf_file(path, required_latency_ids)


def perf_output_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"


def write_perf_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def format_latency_value(value: float) -> str:
    rendered = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in rendered:
        rendered += ".0"
    return rendered


def render_perf_case_records(
    records: list[PerfCaseRecord],
    *,
    latency_prefix: str,
    raw_prefix: str,
    resolved_kernels_prefix: str,
    kernel_source_prefix: str,
    latency_error_prefix: str,
    missing_kernel_match_error: str,
    elapsed_id_prefix: str = "",
) -> list[str]:
    rendered: list[str] = []
    for record in records:
        rendered.extend(
            render_perf_case_record(
                record,
                latency_prefix=latency_prefix,
                raw_prefix=raw_prefix,
                resolved_kernels_prefix=resolved_kernels_prefix,
                kernel_source_prefix=kernel_source_prefix,
                latency_error_prefix=latency_error_prefix,
                missing_kernel_match_error=missing_kernel_match_error,
                elapsed_id_prefix=elapsed_id_prefix,
            )
        )
    return rendered


def render_perf_case_record(
    record: PerfCaseRecord,
    *,
    latency_prefix: str,
    raw_prefix: str,
    resolved_kernels_prefix: str,
    kernel_source_prefix: str,
    latency_error_prefix: str,
    missing_kernel_match_error: str,
    elapsed_id_prefix: str = "",
) -> list[str]:
    case_label = record.case_label
    lines = [f"{latency_prefix}-{case_label}: {_format_case_latency_value(record)}"]
    if record.elapsed_seconds is not None:
        elapsed_id = f"{elapsed_id_prefix}-{case_label}" if elapsed_id_prefix else case_label
        lines.append(f"# elapsed-seconds-{elapsed_id}: {record.elapsed_seconds:.6f}")
    if record.metrics is not None:
        raw_payload = json.dumps({"ops": record.metrics["ops"]}, separators=(",", ":"))
        lines.append(f"# {raw_prefix}-{case_label}: {raw_payload}")
        if record.metrics["kernel_avg_time_us"] is None:
            lines.append(f"# {latency_error_prefix}-{case_label}: {missing_kernel_match_error}")
    if record.error_message is not None:
        lines.append(f"# {latency_error_prefix}-{case_label}: {record.error_message}")
    lines.append(f"# {resolved_kernels_prefix}-{case_label}: {','.join(record.kernel_names)}")
    lines.append(f"# {kernel_source_prefix}-{case_label}: {record.kernel_source}")
    return lines


def _format_case_latency_value(record: PerfCaseRecord) -> str:
    if record.metrics is None or record.metrics["kernel_avg_time_us"] is None:
        return "NA"
    return format_latency_value(record.metrics["kernel_avg_time_us"])


def _parse_perf_file(path: Path) -> dict[str, float]:
    entries = _parse_perf_entries(path)
    return PerfValueMap(
        {latency_id: entry.numeric_value for latency_id, entry in entries.items()},
        comparison_modes={latency_id: entry.comparison_mode for latency_id, entry in entries.items()},
    )


def _parse_perf_entries(path: Path) -> dict[str, PerfEntry]:
    return _parse_perf_entries_strict(path).entries


def _parse_perf_entries_strict(path: Path) -> PerfParseOutcome:
    return _parse_perf_entries_impl(path, tolerate_latency_errors=False)


def _parse_perf_entries_for_comparison(
    path: Path,
    *,
    skip_latency_errors: bool,
    metric_source: MetricSource,
) -> PerfParseOutcome:
    return _parse_perf_entries_impl(
        path,
        tolerate_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )


def _parse_perf_entries_impl(
    path: Path,
    *,
    tolerate_latency_errors: bool,
    metric_source: MetricSource = "auto",
) -> PerfParseOutcome:
    lines = path.read_text(encoding="utf-8").splitlines()
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
    latency_errors = _parse_latency_errors(path, lines)
    entries: dict[str, PerfEntry] = {}
    skipped_latency_errors: dict[str, str] = {}
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
        uncomparable_error = _get_uncomparable_latency_error(
            path, line_no, latency_id, latency_errors
        )
        if uncomparable_error is not None:
            if tolerate_latency_errors:
                skipped_latency_errors[latency_id] = uncomparable_error
                continue
            raise ValueError(uncomparable_error)
        entries[latency_id] = _build_perf_entry_for_source(
            path=path,
            line_no=line_no,
            latency_id=latency_id,
            value_text=value_text,
            raw_totals=raw_totals,
            metric_source=metric_source,
        )
    if not entries and not skipped_latency_errors:
        raise ValueError(f"{path} did not contain any latency-<id>: <value> entries")
    return PerfParseOutcome(entries=entries, skipped_latency_errors=skipped_latency_errors)


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
    return _parse_required_perf_entries_strict(path, required_latency_ids).entries


def _parse_required_perf_entries_strict(
    path: Path, required_latency_ids: RequiredLatencyIds
) -> PerfParseOutcome:
    return _parse_required_perf_entries_impl(
        path, required_latency_ids, tolerate_latency_errors=False
    )


def _parse_required_perf_entries_for_comparison(
    path: Path,
    required_latency_ids: RequiredLatencyIds,
    *,
    skip_latency_errors: bool,
    metric_source: MetricSource,
) -> PerfParseOutcome:
    return _parse_required_perf_entries_impl(
        path,
        required_latency_ids,
        tolerate_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )


def _parse_required_perf_entries_impl(
    path: Path,
    required_latency_ids: RequiredLatencyIds,
    *,
    tolerate_latency_errors: bool,
    metric_source: MetricSource = "auto",
) -> PerfParseOutcome:
    required_ids, comparison_modes = _resolve_required_latency_requirements(required_latency_ids)
    if not required_ids:
        return PerfParseOutcome(entries={}, skipped_latency_errors={})

    lines = path.read_text(encoding="utf-8").splitlines()
    raw_totals = _parse_raw_op_statistic_totals(path, lines)
    latency_errors = _parse_latency_errors(path, lines)
    entries: dict[str, PerfEntry] = {}
    skipped_latency_errors: dict[str, str] = {}
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
        uncomparable_error = _get_uncomparable_latency_error(
            path, line_no, latency_id, latency_errors
        )
        if uncomparable_error is not None:
            if tolerate_latency_errors:
                skipped_latency_errors[latency_id] = uncomparable_error
                continue
            raise ValueError(uncomparable_error)
        effective_metric_source = (
            "total-op"
            if metric_source == "auto" and comparison_modes[latency_id] == "total-op"
            else metric_source
        )
        entries[latency_id] = _build_perf_entry_for_source(
            path=path,
            line_no=line_no,
            latency_id=latency_id,
            value_text=value_text,
            raw_totals=raw_totals,
            metric_source=effective_metric_source,
        )

    missing_ids = sorted(required_ids - set(entries) - set(skipped_latency_errors))
    if missing_ids:
        raise ValueError(f"{path} is missing required latency ids: {missing_ids}")
    return PerfParseOutcome(entries=entries, skipped_latency_errors=skipped_latency_errors)


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
        typed_required_latency_ids = cast(object, required_latency_ids)
        for latency_id in required_ids:
            value = cast(dict[str, object], typed_required_latency_ids)[latency_id]
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
    *,
    reason: str = "to provide total-op fallback",
) -> float:
    total = raw_totals.get(latency_id)
    if total is None:
        raise ValueError(
            f"{path}:{line_no} requires '# raw-op-statistic-{latency_id.removeprefix('latency-')}: ...' {reason}"
        )
    return total


def _build_perf_entry_for_source(
    *,
    path: Path,
    line_no: int,
    latency_id: str,
    value_text: str,
    raw_totals: dict[str, float],
    metric_source: MetricSource,
) -> PerfEntry:
    if metric_source == "kernel":
        if value_text == "NA":
            raise ValueError(
                f"{path}:{line_no} requires kernel latency for '{latency_id}' under --metric-source kernel"
            )
        try:
            parsed_value = float(value_text)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc
        return PerfEntry(
            display_value=value_text,
            numeric_value=parsed_value,
            comparison_mode="latency",
        )

    if metric_source == "total-op":
        total_op_value = _require_raw_total(
            path,
            line_no,
            latency_id,
            raw_totals,
            reason=f"for '{latency_id}' under --metric-source total-op",
        )
        display_value = (
            f"NA ({_format_total_op_display(total_op_value)})"
            if value_text == "NA"
            else _format_total_op_display(total_op_value)
        )
        return PerfEntry(
            display_value=display_value,
            numeric_value=total_op_value,
            comparison_mode="total-op",
        )

    if value_text == "NA":
        total_op_value = _require_raw_total(path, line_no, latency_id, raw_totals)
        return PerfEntry(
            display_value=f"NA ({_format_total_op_display(total_op_value)})",
            numeric_value=total_op_value,
            comparison_mode="total-op",
        )

    try:
        parsed_value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no} has invalid latency value '{value_text}'") from exc
    return PerfEntry(
        display_value=value_text,
        numeric_value=parsed_value,
        comparison_mode="latency",
    )


def _get_uncomparable_latency_error(
    path: Path,
    line_no: int,
    latency_id: str,
    latency_errors: dict[str, str],
) -> str | None:
    error_message = latency_errors.get(latency_id)
    if error_message is None or error_message.startswith("no resolved kernels matched"):
        return None
    return (
        f"{path}:{line_no} cannot compare '{latency_id}' because "
        f"'# latency-error-{latency_id.removeprefix('latency-')}: {error_message}' is present"
    )


def _format_total_op_display(value: float) -> str:
    return f"total-op={format_latency_value(value)}"


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


def _summarize_metric_source(
    entries: dict[str, PerfEntry],
    *,
    metric_source: MetricSource = "auto",
) -> str:
    if not entries:
        return "unknown"
    if metric_source == "kernel":
        return "kernel"
    if metric_source == "total-op":
        return "total-op"
    modes = {entry.comparison_mode for entry in entries.values()}
    if modes == {"latency"}:
        return "kernel"
    if modes == {"total-op"}:
        return "total-op"
    return "mixed (kernel + total-op fallback)"
