from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from perf_artifacts import PerfMetrics, PerfOpRow, format_latency_value


@dataclass(frozen=True)
class ParsedProfileCsvRows:
    source_path: Path
    ops: list[PerfOpRow]
    total_time_us: float


def find_optional_profile_csv(profile_root: Path, filename: str) -> Path | None:
    matches = sorted(path for path in profile_root.rglob(filename) if path.is_file())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def find_latest_op_statistic_csv(profile_root: Path) -> Path | None:
    matches = sorted(path for path in profile_root.rglob("op_statistic_*.csv") if path.is_file())
    plain_path = find_optional_profile_csv(profile_root, "op_statistic.csv")
    if plain_path is not None:
        matches.append(plain_path)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def parse_op_statistic_csv(csv_path: Path) -> ParsedProfileCsvRows:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "OP Type" not in fieldnames:
            raise ValueError(f"Missing required column 'OP Type' in {csv_path}")
        if "Avg Time(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Avg Time(us)' in {csv_path}")
        if "Total Time(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Total Time(us)' in {csv_path}")

        ops: list[PerfOpRow] = []
        total_time_us = 0.0
        row_count = 0
        for row in reader:
            op_type = (row.get("OP Type") or "").strip()
            if not op_type:
                raise ValueError(f"Empty 'OP Type' value in {csv_path}")
            avg_time_us = _parse_float_field(row.get("Avg Time(us)"), "Avg Time(us)", csv_path)
            total_time_us += _parse_float_field(row.get("Total Time(us)"), "Total Time(us)", csv_path)
            ops.append({"op_type": op_type, "avg_time_us": avg_time_us})
            row_count += 1

    if row_count == 0:
        raise ValueError(f"No rows found in {csv_path}")
    return ParsedProfileCsvRows(
        source_path=csv_path,
        ops=ops,
        total_time_us=total_time_us,
    )


def parse_operator_details_csv(
    csv_path: Path,
    *,
    active_count: int,
    kernel_names: list[str],
) -> ParsedProfileCsvRows:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Name" not in fieldnames:
            raise ValueError(f"Missing required column 'Name' in {csv_path}")
        if "Device Self Duration(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Device Self Duration(us)' in {csv_path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Profiler operator details are empty: {csv_path}")

    totals_by_name: dict[str, float] = {}
    ordered_names: list[str] = []
    total_time_us = 0.0
    kernel_name_set = set(kernel_names)
    for row in rows:
        op_name = (row.get("Name") or "").strip()
        if not op_name:
            raise ValueError(f"Encountered empty operator name in {csv_path}")
        duration = _parse_float_field(
            row.get("Device Self Duration(us)"),
            "Device Self Duration(us)",
            csv_path,
        )
        total_time_us += duration
        if duration == 0.0 and op_name not in kernel_name_set:
            continue
        if op_name not in totals_by_name:
            totals_by_name[op_name] = 0.0
            ordered_names.append(op_name)
        totals_by_name[op_name] += duration

    ops: list[PerfOpRow] = []
    for op_name in ordered_names:
        avg_time_us = totals_by_name[op_name] / active_count
        ops.append({"op_type": op_name, "avg_time_us": float(format_latency_value(avg_time_us))})

    return ParsedProfileCsvRows(
        source_path=csv_path,
        ops=ops,
        total_time_us=total_time_us,
    )


def parse_kernel_details_csv(
    csv_path: Path,
    *,
    active_count: int,
) -> ParsedProfileCsvRows:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "Name" not in fieldnames:
            raise ValueError(f"Missing required column 'Name' in {csv_path}")
        if "Duration(us)" not in fieldnames:
            raise ValueError(f"Missing required column 'Duration(us)' in {csv_path}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Profiler kernel details are empty: {csv_path}")

    totals_by_name: dict[str, float] = {}
    ordered_names: list[str] = []
    total_time_us = 0.0
    for row in rows:
        op_name = (row.get("Name") or "").strip()
        if not op_name:
            raise ValueError(f"Encountered empty operator name in {csv_path}")
        duration = _parse_float_field(row.get("Duration(us)"), "Duration(us)", csv_path)
        total_time_us += duration
        if op_name not in totals_by_name:
            totals_by_name[op_name] = 0.0
            ordered_names.append(op_name)
        totals_by_name[op_name] += duration

    ops: list[PerfOpRow] = []
    for op_name in ordered_names:
        avg_time_us = totals_by_name[op_name] / active_count
        ops.append({"op_type": op_name, "avg_time_us": float(format_latency_value(avg_time_us))})

    return ParsedProfileCsvRows(
        source_path=csv_path,
        ops=ops,
        total_time_us=total_time_us,
    )


def resolve_perf_metrics(ops: list[PerfOpRow], kernel_names: list[str]) -> PerfMetrics:
    kernel_name_set = set(kernel_names)
    matched_avg_times = [
        float(row["avg_time_us"]) for row in ops if str(row["op_type"]) in kernel_name_set
    ]
    return {
        "kernel_avg_time_us": sum(matched_avg_times) if matched_avg_times else None,
        "ops": [
            {
                "op_type": row["op_type"],
                "avg_time_us": row["avg_time_us"],
            }
            for row in ops
        ],
    }


def _parse_float_field(raw_value: object, field_name: str, csv_path: Path) -> float:
    stripped = "" if raw_value is None else str(raw_value).strip()
    if not stripped:
        raise ValueError(f"Empty '{field_name}' value in {csv_path}")
    return float(stripped)
