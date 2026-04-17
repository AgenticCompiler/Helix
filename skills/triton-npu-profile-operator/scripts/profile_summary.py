#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
from typing import Any


OP_NAME_COLUMNS = ("OP Type", "Op Type", "Op Name", "Kernel Name")
DURATION_COLUMNS = ("Task Duration(us)", "Task Duration", "Duration(us)")
STATISTIC_REQUIRED_COLUMNS = (
    "OP Type",
    "Core Type",
    "Count",
    "Total Time(us)",
    "Min Time(us)",
    "Avg Time(us)",
    "Max Time(us)",
    "Ratio(%)",
)
TRANSFER_HINTS = (
    "copy",
    "memcpy",
    "transdata",
    "dma",
    "load",
    "store",
    "move",
)
SUMMARY_RATIO_COLUMNS = {
    "aic_mac_ratio": "aic_mac_ratio",
    "aic_scalar_ratio": "aic_scalar_ratio",
    "aic_mte1_ratio": "aic_mte1_ratio",
    "aic_mte2_ratio": "aic_mte2_ratio",
    "aic_mte3_ratio": "aic_mte3_ratio",
    "aiv_vec_ratio": "aiv_vec_ratio",
    "aiv_scalar_ratio": "aiv_scalar_ratio",
    "aiv_mte2_ratio": "aiv_mte2_ratio",
    "aiv_mte3_ratio": "aiv_mte3_ratio",
}
SUMMARY_EXTRA_COLUMNS = {
    "Task Wait Time(us)": "task_wait_time_us",
    "Block Dim": "block_dim",
    "cube_utilization(%)": "cube_utilization_percent",
}
TASK_TIME_NAME_COLUMNS = ("kernel_name", "Kernel Name", "Op Name", "OP Type")
TASK_TIME_DURATION_COLUMNS = ("task_time(us)", "Task Duration(us)")
TASK_TIME_START_COLUMNS = ("task_start(us)", "Task Start Time(us)")
TASK_TIME_STOP_COLUMNS = ("task_stop(us)", "Task Stop Time(us)")
API_NAME_COLUMNS = ("API Name", "Name")
API_TIME_COLUMNS = ("Time(us)", "Duration(us)")


def _format_number(value: float) -> str:
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _parse_float(value: str) -> float:
    return float(value.strip())


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _aggregate_numeric(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


def _normalize_core_type(core_type: str) -> str:
    lowered = core_type.strip().lower()
    if "scalar" in lowered:
        return "scalar"
    if "vector" in lowered:
        return "vector"
    if "cube" in lowered:
        return "cube"
    return "other"


def resolve_profile_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        raise FileNotFoundError(f"Expected a directory, got file: {candidate}")
    if candidate.name.startswith("PROF_") and (candidate / "mindstudio_profiler_output").is_dir():
        return candidate
    if candidate.name == "mindstudio_profiler_output" and candidate.is_dir():
        return candidate.parent
    if (candidate / "mindstudio_profiler_output").is_dir():
        return candidate

    matches = [
        match
        for match in candidate.rglob("PROF_*")
        if match.is_dir() and (match / "mindstudio_profiler_output").is_dir()
    ]
    if not matches:
        raise FileNotFoundError(f"No PROF_* directory found under {candidate}")
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _find_newest_csv(output_dir: Path, prefix: str) -> Path | None:
    matches = sorted(output_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _find_newest_json(output_dir: Path, prefix: str) -> Path | None:
    matches = sorted(output_dir.glob(f"{prefix}_*.json"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _find_newest_bin(profile_dir: Path) -> Path | None:
    matches = [
        path
        for path in profile_dir.rglob("*.bin")
        if path.is_file()
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _load_statistic_rows(csv_path: Path) -> list[dict[str, float | str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in STATISTIC_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path}: {', '.join(missing)}"
            )

        rows: list[dict[str, float | str]] = []
        for row in reader:
            rows.append(
                {
                    "op_type": row["OP Type"].strip(),
                    "core_type": row["Core Type"].strip(),
                    "count": _parse_float(row["Count"]),
                    "total_time_us": _parse_float(row["Total Time(us)"]),
                    "min_time_us": _parse_float(row["Min Time(us)"]),
                    "avg_time_us": _parse_float(row["Avg Time(us)"]),
                    "max_time_us": _parse_float(row["Max Time(us)"]),
                    "ratio_percent": _parse_float(row["Ratio(%)"]),
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _select_target_row(
    rows: list[dict[str, float | str]], target_op: str | None
) -> tuple[dict[str, float | str], bool]:
    if target_op is not None:
        for row in rows:
            if str(row["op_type"]) == target_op:
                return row, False
        raise ValueError(f"Target operator not found in op_statistic: {target_op}")
    return max(rows, key=lambda item: float(item["total_time_us"])), True


def _serialize_row(row: dict[str, float | str]) -> dict[str, float | str]:
    return {
        "op_type": str(row["op_type"]),
        "core_type": str(row["core_type"]),
        "count": float(row["count"]),
        "total_time_us": float(row["total_time_us"]),
        "min_time_us": float(row["min_time_us"]),
        "avg_time_us": float(row["avg_time_us"]),
        "max_time_us": float(row["max_time_us"]),
        "ratio_percent": float(row["ratio_percent"]),
    }


def _aggregate_core_types(
    rows: list[dict[str, float | str]],
) -> dict[str, dict[str, float | list[str]]]:
    totals: dict[str, dict[str, float | list[str]]] = {}
    for row in rows:
        bucket = _normalize_core_type(str(row["core_type"]))
        entry = totals.setdefault(
            bucket,
            {
                "total_time_us": 0.0,
                "ratio_percent": 0.0,
                "count": 0.0,
                "raw_core_types": [],
            },
        )
        entry["total_time_us"] = float(entry["total_time_us"]) + float(row["total_time_us"])
        entry["ratio_percent"] = float(entry["ratio_percent"]) + float(row["ratio_percent"])
        entry["count"] = float(entry["count"]) + float(row["count"])
        raw_core_types = entry["raw_core_types"]
        assert isinstance(raw_core_types, list)
        core_type = str(row["core_type"])
        if core_type not in raw_core_types:
            raw_core_types.append(core_type)
    return totals


def _is_data_movement_op(op_type: str) -> bool:
    lowered = op_type.lower()
    return any(hint in lowered for hint in TRANSFER_HINTS)


def _collect_data_movement_hotspots(
    rows: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    hotspots = [row for row in rows if _is_data_movement_op(str(row["op_type"]))]
    return sorted(hotspots, key=lambda item: float(item["total_time_us"]), reverse=True)


def _find_summary_columns(fieldnames: list[str]) -> tuple[str | None, str | None]:
    op_name_column = next((name for name in OP_NAME_COLUMNS if name in fieldnames), None)
    duration_column = next((name for name in DURATION_COLUMNS if name in fieldnames), None)
    return op_name_column, duration_column


def _classify_operator_type(
    target_row: dict[str, float | str],
    ratio_avgs: dict[str, float],
    cube_utilization_avg: float | None,
) -> dict[str, Any]:
    vector_ratio = ratio_avgs.get("aiv_vec_ratio", 0.0)
    cube_ratio = max(ratio_avgs.get("aic_mac_ratio", 0.0), cube_utilization_avg or 0.0)
    signals: list[str] = []
    if vector_ratio >= 30.0:
        signals.append(f"high aiv_vec_ratio={_format_number(vector_ratio)}")
    if cube_ratio >= 20.0:
        signals.append(f"high cube-side activity={_format_number(cube_ratio)}")

    if vector_ratio >= 30.0 and cube_ratio >= 20.0:
        kind = "mix"
    elif cube_ratio >= 20.0:
        kind = "cube"
    elif vector_ratio >= 20.0:
        kind = "vector"
    else:
        kind = _normalize_core_type(str(target_row["core_type"]))
        signals.append(f"fallback to target core type {target_row['core_type']}")

    return {
        "kind": kind,
        "signals": signals,
        "source": "op_summary" if signals and "fallback" not in signals[-1] else "op_statistic",
    }


def _classify_bound(ratio_avgs: dict[str, float]) -> dict[str, Any]:
    compute_score = ratio_avgs.get("aic_mac_ratio", 0.0) + ratio_avgs.get("aiv_vec_ratio", 0.0)
    memory_score = (
        ratio_avgs.get("aic_mte1_ratio", 0.0)
        + ratio_avgs.get("aic_mte2_ratio", 0.0)
        + ratio_avgs.get("aic_mte3_ratio", 0.0)
        + ratio_avgs.get("aiv_mte2_ratio", 0.0)
        + ratio_avgs.get("aiv_mte3_ratio", 0.0)
    )
    scalar_score = ratio_avgs.get("aic_scalar_ratio", 0.0) + ratio_avgs.get("aiv_scalar_ratio", 0.0)

    if scalar_score >= max(compute_score, memory_score) and scalar_score >= 20.0:
        classification = "scalar-overhead"
        reasoning = ["scalar-side ratios dominate the visible pipeline ratios"]
    elif memory_score >= compute_score + 5.0:
        classification = "memory-bound"
        reasoning = ["MTE-side ratios exceed compute-side ratios"]
    elif compute_score >= 70.0 and scalar_score < 20.0 and memory_score < 25.0:
        classification = "compute-bound"
        reasoning = ["compute-side ratios dominate with limited scalar and MTE pressure"]
    else:
        classification = "mixed"
        reasoning = ["no single pipeline family dominates strongly enough"]

    return {
        "classification": classification,
        "scores": {
            "compute": compute_score,
            "memory": memory_score,
            "scalar": scalar_score,
        },
        "reasoning": reasoning,
    }


def _summarize_op_summary(
    csv_path: Path | None, target_op: str, target_row: dict[str, float | str]
) -> tuple[dict[str, float | int | str | None], dict[str, Any], dict[str, Any], dict[str, Any]]:
    summary_stats: dict[str, float | int | str | None] = {
        "path": None,
        "matched_rows": 0,
        "total_duration_us": None,
        "avg_duration_us": None,
        "min_duration_us": None,
        "max_duration_us": None,
        "note": "No op_summary CSV found." if csv_path is None else None,
    }
    operator_type_guess = {
        "kind": _normalize_core_type(str(target_row["core_type"])),
        "signals": [f"fallback to target core type {target_row['core_type']}"],
        "source": "op_statistic",
    }
    bound_analysis = {
        "classification": "unknown",
        "scores": {"compute": 0.0, "memory": 0.0, "scalar": 0.0},
        "reasoning": ["op_summary evidence unavailable"],
    }
    pipeline_signals: dict[str, Any] = {
        "matched_rows": 0,
        "ratios": {},
        "task_wait_time_us": None,
        "block_dim": None,
        "cube_utilization_percent": None,
    }

    if csv_path is None:
        return summary_stats, operator_type_guess, bound_analysis, pipeline_signals

    summary_stats["path"] = str(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        op_name_column, duration_column = _find_summary_columns(fieldnames)
        if op_name_column is None:
            summary_stats["note"] = "No operator-identifying column was recognized in op_summary."
            return summary_stats, operator_type_guess, bound_analysis, pipeline_signals

        ratio_values: dict[str, list[float]] = {key: [] for key in SUMMARY_RATIO_COLUMNS.values()}
        extra_values: dict[str, list[float]] = {key: [] for key in SUMMARY_EXTRA_COLUMNS.values()}
        duration_values: list[float] = []
        matched_rows = 0

        for row in reader:
            if row.get(op_name_column, "").strip() != target_op:
                continue
            matched_rows += 1
            if duration_column is not None:
                duration = _safe_float(row.get(duration_column))
                if duration is not None:
                    duration_values.append(duration)
            for column, key in SUMMARY_RATIO_COLUMNS.items():
                value = _safe_float(row.get(column))
                if value is not None:
                    ratio_values[key].append(value)
            for column, key in SUMMARY_EXTRA_COLUMNS.items():
                value = _safe_float(row.get(column))
                if value is not None:
                    extra_values[key].append(value)

    summary_stats["matched_rows"] = matched_rows
    pipeline_signals["matched_rows"] = matched_rows
    if matched_rows == 0:
        summary_stats["note"] = "Target operator did not match any op_summary rows."
        return summary_stats, operator_type_guess, bound_analysis, pipeline_signals

    if duration_values:
        summary_stats["total_duration_us"] = sum(duration_values)
        summary_stats["avg_duration_us"] = sum(duration_values) / len(duration_values)
        summary_stats["min_duration_us"] = min(duration_values)
        summary_stats["max_duration_us"] = max(duration_values)
    elif duration_column is None:
        summary_stats["note"] = "No duration column was recognized in op_summary."

    ratio_summaries: dict[str, Any] = {}
    ratio_avgs: dict[str, float] = {}
    for key, values in ratio_values.items():
        aggregated = _aggregate_numeric(values)
        if aggregated is not None:
            ratio_summaries[key] = aggregated
            ratio_avgs[key] = float(aggregated["avg"])
    pipeline_signals["ratios"] = ratio_summaries

    wait_summary = _aggregate_numeric(extra_values["task_wait_time_us"])
    if wait_summary is not None:
        pipeline_signals["task_wait_time_us"] = wait_summary
    block_dim_summary = _aggregate_numeric(extra_values["block_dim"])
    if block_dim_summary is not None:
        pipeline_signals["block_dim"] = {
            **block_dim_summary,
            "observed_values": sorted({int(value) for value in extra_values["block_dim"]}),
        }
    cube_summary = _aggregate_numeric(extra_values["cube_utilization_percent"])
    if cube_summary is not None:
        pipeline_signals["cube_utilization_percent"] = cube_summary

    operator_type_guess = _classify_operator_type(
        target_row,
        ratio_avgs,
        float(cube_summary["avg"]) if cube_summary is not None else None,
    )
    bound_analysis = _classify_bound(ratio_avgs)
    return summary_stats, operator_type_guess, bound_analysis, pipeline_signals


def _summarize_task_time(csv_path: Path | None, target_op: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(csv_path) if csv_path is not None else None,
        "matched_rows": 0,
        "stream_ids": [],
        "task_ids": [],
        "total_task_time_us": None,
        "span_us": None,
        "total_gap_us": 0.0,
        "max_gap_us": 0.0,
        "overlap_count": 0,
        "note": "No task_time CSV found." if csv_path is None else None,
    }
    if csv_path is None:
        return summary

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_column = next((name for name in TASK_TIME_NAME_COLUMNS if name in fieldnames), None)
        duration_column = next((name for name in TASK_TIME_DURATION_COLUMNS if name in fieldnames), None)
        start_column = next((name for name in TASK_TIME_START_COLUMNS if name in fieldnames), None)
        stop_column = next((name for name in TASK_TIME_STOP_COLUMNS if name in fieldnames), None)
        if name_column is None:
            summary["note"] = "No kernel-identifying column was recognized in task_time."
            return summary

        matched_rows: list[dict[str, Any]] = []
        for row in reader:
            if row.get(name_column, "").strip() != target_op:
                continue
            matched_rows.append(row)

    summary["matched_rows"] = len(matched_rows)
    if not matched_rows:
        summary["note"] = "Target operator did not match any task_time rows."
        return summary

    durations = [
        value
        for row in matched_rows
        if duration_column is not None
        for value in [_safe_float(row.get(duration_column))]
        if value is not None
    ]
    if durations:
        summary["total_task_time_us"] = sum(durations)

    starts = [
        value
        for row in matched_rows
        if start_column is not None
        for value in [_safe_float(row.get(start_column))]
        if value is not None
    ]
    stops = [
        value
        for row in matched_rows
        if stop_column is not None
        for value in [_safe_float(row.get(stop_column))]
        if value is not None
    ]
    if starts and stops:
        summary["span_us"] = max(stops) - min(starts)

    stream_ids = sorted({str(row.get("stream_id", "")).strip() for row in matched_rows if row.get("stream_id")})
    task_ids = sorted({str(row.get("task_id", "")).strip() for row in matched_rows if row.get("task_id")})
    summary["stream_ids"] = stream_ids
    summary["task_ids"] = task_ids

    if start_column is not None and stop_column is not None:
        ordered = []
        for row in matched_rows:
            start = _safe_float(row.get(start_column))
            stop = _safe_float(row.get(stop_column))
            if start is None or stop is None:
                continue
            ordered.append((start, stop))
        ordered.sort()
        previous_stop: float | None = None
        total_gap = 0.0
        max_gap = 0.0
        overlap_count = 0
        for start, stop in ordered:
            if previous_stop is not None:
                gap = start - previous_stop
                if gap > 0:
                    total_gap += gap
                    max_gap = max(max_gap, gap)
                elif gap < 0:
                    overlap_count += 1
            previous_stop = max(previous_stop, stop) if previous_stop is not None else stop
        summary["total_gap_us"] = total_gap
        summary["max_gap_us"] = max_gap
        summary["overlap_count"] = overlap_count

    return summary


def _summarize_api_statistic(csv_path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(csv_path) if csv_path is not None else None,
        "top_apis": [],
        "launch_related_present": False,
        "sync_related_present": False,
        "tiling_related_present": False,
        "note": "No api_statistic CSV found." if csv_path is None else None,
    }
    if csv_path is None:
        return summary

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_column = next((name for name in API_NAME_COLUMNS if name in fieldnames), None)
        time_column = next((name for name in API_TIME_COLUMNS if name in fieldnames), None)
        if name_column is None or time_column is None:
            summary["note"] = "No API name or time column was recognized in api_statistic."
            return summary

        apis = []
        for row in reader:
            api_name = row.get(name_column, "").strip()
            time_us = _safe_float(row.get(time_column))
            if not api_name or time_us is None:
                continue
            count = _safe_float(row.get("Count"))
            avg_us = _safe_float(row.get("Avg(us)"))
            apis.append(
                {
                    "api_name": api_name,
                    "time_us": time_us,
                    "count": count,
                    "avg_us": avg_us,
                }
            )

    apis.sort(key=lambda item: item["time_us"], reverse=True)
    summary["top_apis"] = apis[:5]
    names = " ".join(item["api_name"].lower() for item in apis)
    summary["launch_related_present"] = any(token in names for token in ("launch", "execute"))
    summary["sync_related_present"] = "sync" in names
    summary["tiling_related_present"] = "til" in names
    return summary


def _compute_max_overlap(intervals: list[tuple[float, float]]) -> int:
    events: list[tuple[float, int]] = []
    for start, stop in intervals:
        events.append((start, 1))
        events.append((stop, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        max_active = max(max_active, active)
    return max_active


def _summarize_msprof_json(json_path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(json_path) if json_path is not None else None,
        "event_count": 0,
        "complete_event_count": 0,
        "stream_like_tracks": 0,
        "max_overlap": 0,
        "note": "No msprof JSON found." if json_path is None else None,
    }
    if json_path is None:
        return summary

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        summary["note"] = "msprof JSON did not contain a top-level event list."
        return summary

    tracks = set()
    intervals: list[tuple[float, float]] = []
    complete_event_count = 0
    for item in data:
        if not isinstance(item, dict):
            continue
        summary["event_count"] += 1
        pid = item.get("pid")
        tid = item.get("tid")
        if pid is not None or tid is not None:
            tracks.add((pid, tid))
        if item.get("ph") == "X":
            ts = item.get("ts")
            dur = item.get("dur")
            if isinstance(ts, (int, float)) and isinstance(dur, (int, float)):
                complete_event_count += 1
                intervals.append((float(ts), float(ts) + float(dur)))

    summary["complete_event_count"] = complete_event_count
    summary["stream_like_tracks"] = len(tracks)
    if intervals:
        summary["max_overlap"] = _compute_max_overlap(intervals)
    return summary


def _load_parse_bin_module() -> Any | None:
    script = Path(__file__).resolve().parent / "parse_bin.py"
    spec = importlib.util.spec_from_file_location("profile_summary_parse_bin_module", script)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summarize_binary(profile_dir: Path) -> dict[str, Any]:
    bin_path = _find_newest_bin(profile_dir)
    if bin_path is None:
        return {"path": None, "available": False, "note": "No profiler binary found."}
    module = _load_parse_bin_module()
    if module is None or not hasattr(module, "summarize_file"):
        return {
            "path": str(bin_path),
            "available": False,
            "note": "Binary parser helper could not be loaded.",
        }
    try:
        signals = module.summarize_file(str(bin_path))
    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "path": str(bin_path),
            "available": False,
            "note": f"Binary parsing failed: {exc}",
        }
    return {
        "path": str(bin_path),
        "available": True,
        "signals": signals,
    }


def build_profile_payload(
    profile_path: str | Path,
    *,
    target_op: str | None = None,
    top_count: int = 5,
) -> dict[str, object]:
    profile_dir = resolve_profile_dir(profile_path)
    output_dir = profile_dir / "mindstudio_profiler_output"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Missing mindstudio_profiler_output directory in {profile_dir}")

    op_statistic_csv = _find_newest_csv(output_dir, "op_statistic")
    if op_statistic_csv is None:
        raise FileNotFoundError(f"No op_statistic_*.csv found in {output_dir}")
    op_summary_csv = _find_newest_csv(output_dir, "op_summary")
    task_time_csv = _find_newest_csv(output_dir, "task_time")
    api_statistic_csv = _find_newest_csv(output_dir, "api_statistic")
    msprof_json = _find_newest_json(output_dir, "msprof")

    statistic_rows = _load_statistic_rows(op_statistic_csv)
    target_row, inferred = _select_target_row(statistic_rows, target_op)
    top_rows = sorted(
        statistic_rows,
        key=lambda item: float(item["total_time_us"]),
        reverse=True,
    )[:top_count]
    core_type_totals = _aggregate_core_types(statistic_rows)
    data_movement_hotspots = _collect_data_movement_hotspots(statistic_rows)
    op_summary_stats, operator_type_guess, bound_analysis, pipeline_signals = _summarize_op_summary(
        op_summary_csv,
        str(target_row["op_type"]),
        target_row,
    )
    task_timeline_signals = _summarize_task_time(task_time_csv, str(target_row["op_type"]))
    host_api_signals = _summarize_api_statistic(api_statistic_csv)
    msprof_timeline_signals = _summarize_msprof_json(msprof_json)
    binary_signals = _summarize_binary(profile_dir)

    return {
        "profile_dir": str(profile_dir),
        "op_statistic_file": op_statistic_csv.name,
        "op_summary_file": op_summary_csv.name if op_summary_csv is not None else None,
        "task_time_file": task_time_csv.name if task_time_csv is not None else None,
        "api_statistic_file": api_statistic_csv.name if api_statistic_csv is not None else None,
        "msprof_json_file": msprof_json.name if msprof_json is not None else None,
        "target_operator": str(target_row["op_type"]),
        "selection": (
            "inferred from the hottest `op_statistic` row by `Total Time(us)`"
            if inferred
            else "matched the explicit `--target-op` value"
        ),
        "target_row": _serialize_row(target_row),
        "op_summary": op_summary_stats,
        "core_type_totals": core_type_totals,
        "data_movement_hotspots": [_serialize_row(row) for row in data_movement_hotspots],
        "top_ops": [_serialize_row(row) for row in top_rows],
        "operator_type_guess": operator_type_guess,
        "bound_analysis": bound_analysis,
        "pipeline_signals": pipeline_signals,
        "task_timeline_signals": task_timeline_signals,
        "host_api_signals": host_api_signals,
        "msprof_timeline_signals": msprof_timeline_signals,
        "binary_signals": binary_signals,
    }


def _render_markdown(payload: dict[str, object]) -> str:
    target_row = payload["target_row"]
    assert isinstance(target_row, dict)
    summary_stats = payload["op_summary"]
    assert isinstance(summary_stats, dict)
    top_ops = payload["top_ops"]
    assert isinstance(top_ops, list)
    core_type_totals = payload["core_type_totals"]
    assert isinstance(core_type_totals, dict)
    data_movement_hotspots = payload["data_movement_hotspots"]
    assert isinstance(data_movement_hotspots, list)
    pipeline_signals = payload["pipeline_signals"]
    assert isinstance(pipeline_signals, dict)
    operator_type_guess = payload["operator_type_guess"]
    assert isinstance(operator_type_guess, dict)
    bound_analysis = payload["bound_analysis"]
    assert isinstance(bound_analysis, dict)
    task_timeline_signals = payload["task_timeline_signals"]
    assert isinstance(task_timeline_signals, dict)
    host_api_signals = payload["host_api_signals"]
    assert isinstance(host_api_signals, dict)
    msprof_timeline_signals = payload["msprof_timeline_signals"]
    assert isinstance(msprof_timeline_signals, dict)
    binary_signals = payload["binary_signals"]
    assert isinstance(binary_signals, dict)

    top_rows = [
        [
            str(row["op_type"]),
            str(row["core_type"]),
            _format_number(float(row["count"])),
            _format_number(float(row["total_time_us"])),
            _format_number(float(row["avg_time_us"])),
            _format_number(float(row["ratio_percent"])),
        ]
        for row in top_ops
        if isinstance(row, dict)
    ]
    core_rows = [
        [
            bucket,
            ", ".join(sorted(str(item) for item in entry["raw_core_types"])),
            _format_number(float(entry["count"])),
            _format_number(float(entry["total_time_us"])),
            _format_number(float(entry["ratio_percent"])),
        ]
        for bucket, entry in core_type_totals.items()
        if isinstance(entry, dict)
    ]
    movement_rows = [
        [
            str(row["op_type"]),
            str(row["core_type"]),
            _format_number(float(row["total_time_us"])),
            _format_number(float(row["ratio_percent"])),
        ]
        for row in data_movement_hotspots
        if isinstance(row, dict)
    ]

    lines = [
        "# Ascend NPU Operator Profile Summary",
        "",
        f"- Profile directory: `{payload['profile_dir']}`",
        f"- `op_statistic` file: `{payload['op_statistic_file']}`",
        (
            f"- `op_summary` file: `{payload['op_summary_file']}`"
            if payload["op_summary_file"] is not None
            else "- `op_summary` file: `not found`"
        ),
        f"- Target operator: `{payload['target_operator']}`",
        f"- Selection: {payload['selection']}",
        "",
        "## Operator timing",
        "",
        f"- Core type: `{target_row['core_type']}`",
        f"- Invocation count: `{_format_number(float(target_row['count']))}`",
        f"- Total time: `{_format_number(float(target_row['total_time_us']))} us`",
        f"- Average time: `{_format_number(float(target_row['avg_time_us']))} us`",
        f"- Min time: `{_format_number(float(target_row['min_time_us']))} us`",
        f"- Max time: `{_format_number(float(target_row['max_time_us']))} us`",
        f"- Runtime ratio: `{_format_number(float(target_row['ratio_percent']))}%`",
        "",
        "## op_summary cross-check",
        "",
        f"- Matched op_summary rows: `{summary_stats['matched_rows']}`",
    ]

    if summary_stats["avg_duration_us"] is not None:
        lines.extend(
            [
                f"- Summed task duration: `{_format_number(float(summary_stats['total_duration_us']))} us`",
                f"- Average task duration: `{_format_number(float(summary_stats['avg_duration_us']))} us`",
                f"- Min task duration: `{_format_number(float(summary_stats['min_duration_us']))} us`",
                f"- Max task duration: `{_format_number(float(summary_stats['max_duration_us']))} us`",
            ]
        )
    if summary_stats["note"]:
        lines.append(f"- Note: {summary_stats['note']}")

    lines.extend(
        [
            f"- Operator type guess: `{operator_type_guess['kind']}`",
            f"- Bound analysis: `{bound_analysis['classification']}`",
        ]
    )

    if pipeline_signals.get("task_wait_time_us"):
        task_wait = pipeline_signals["task_wait_time_us"]
        assert isinstance(task_wait, dict)
        lines.append(
            f"- Avg task wait time: `{_format_number(float(task_wait['avg']))} us`"
        )

    lines.extend(
        [
            "",
            "## Core type totals",
            "",
            _markdown_table(
                ["Bucket", "Raw core types", "Count", "Total Time(us)", "Ratio(%)"],
                core_rows,
            ),
            "",
            "## Data movement hotspots",
            "",
            (
                _markdown_table(
                    ["OP Type", "Core Type", "Total Time(us)", "Ratio(%)"],
                    movement_rows,
                )
                if movement_rows
                else "_No transfer-like hotspots matched the default heuristics._"
            ),
            "",
            "## Layered profiler signals",
            "",
            f"- Task timeline matched rows: `{task_timeline_signals['matched_rows']}`",
            f"- Max task gap: `{_format_number(float(task_timeline_signals['max_gap_us']))} us`",
            f"- Host launch-related APIs present: `{host_api_signals['launch_related_present']}`",
            f"- msprof tracks: `{msprof_timeline_signals['stream_like_tracks']}`",
            f"- Binary signals available: `{binary_signals.get('available', False)}`",
            "",
            "## Top operators by total time",
            "",
            _markdown_table(
                ["OP Type", "Core Type", "Count", "Total Time(us)", "Avg Time(us)", "Ratio(%)"],
                top_rows,
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def build_profile_report(
    profile_path: str | Path,
    *,
    target_op: str | None = None,
    top_count: int = 5,
    output_format: str = "markdown",
) -> str:
    payload = build_profile_payload(profile_path, target_op=target_op, top_count=top_count)
    if output_format == "json":
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_format != "markdown":
        raise ValueError(f"Unsupported output format: {output_format}")
    return _render_markdown(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ascend msprof CSV output.")
    parser.add_argument("profile_path", help="A PROF_* directory or a parent directory containing one.")
    parser.add_argument("--target-op", help="Operator name to summarize from op_statistic/op_summary.")
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top operators to include in the hotspot table.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("markdown", "json"),
        default="markdown",
        help="Render either the default Markdown summary or a JSON payload.",
    )
    args = parser.parse_args()

    print(
        build_profile_report(
            args.profile_path,
            target_op=args.target_op,
            top_count=args.top,
            output_format=args.output_format,
        ),
        end="",
    )


if __name__ == "__main__":
    main()
