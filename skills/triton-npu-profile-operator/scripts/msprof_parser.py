from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from collections.abc import Sequence

from models import HostApiCall, KernelInvocation, OperatorStats, PipelineStage, TaskRecord
from parser_base import find_newest_csv, parse_api_statistic, parse_op_statistic

_OP_NAME_COLUMNS = ("Op Name", "Name", "OP Type")
_DURATION_COLUMNS = ("Task Duration(us)", "Duration(us)")
_WAIT_COLUMNS = ("Task Wait Time(us)", "Wait Time(us)")
_BLOCK_DIM_COLUMNS = ("Block Dim", "Mix Block Dim")

_PIPELINE_FIELDS = {
    "aic_mac_ratio": ["aic_mac_ratio", "aic_mac_ratio"],
    "aic_scalar_ratio": ["aic_scalar_ratio", "aic_scalar_ratio"],
    "aic_mte1_ratio": ["aic_mte1_ratio", "aic_mte1_ratio"],
    "aic_mte2_ratio": ["aic_mte2_ratio", "aic_mte2_ratio"],
    "aic_mte3_ratio": ["aic_mte3_ratio", "aic_mte3_ratio"],
    "aiv_vec_ratio": ["aiv_vec_ratio", "aiv_vec_ratio"],
    "aiv_scalar_ratio": ["aiv_scalar_ratio", "aiv_scalar_ratio"],
    "aiv_mte2_ratio": ["aiv_mte2_ratio", "aiv_mte2_ratio"],
    "aiv_mte3_ratio": ["aiv_mte3_ratio", "aiv_mte3_ratio"],
    "cube_utilization": ["cube_utilization(%)", "cube_utilization"],
    "block_dim": ["Block Dim"],
}

_TASK_NAME_COLUMNS = ("kernel_name", "Kernel Name", "Op Name", "OP Type")
_TASK_DURATION_COLUMNS = ("task_time(us)", "Task Duration(us)")
_TASK_START_COLUMNS = ("task_start(us)", "Task Start Time(us)")
_TASK_STOP_COLUMNS = ("task_stop(us)", "Task Stop Time(us)")
_TASK_TYPE_COLUMNS = ("kernel_type", "Kernel Type")


def _find_column(fieldnames: Sequence[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in fieldnames:
            return col
    return None


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


def parse_op_summary(csv_path: Path) -> list[KernelInvocation]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_col = _find_column(fieldnames, _OP_NAME_COLUMNS)
        duration_col = _find_column(fieldnames, _DURATION_COLUMNS)
        wait_col = _find_column(fieldnames, _WAIT_COLUMNS)
        block_dim_col = _find_column(fieldnames, _BLOCK_DIM_COLUMNS)

        invocations: list[KernelInvocation] = []
        for row in reader:
            op_name = (row.get(name_col, "") if name_col else "").strip()
            if not op_name:
                continue
            duration = _safe_float(row[duration_col]) if duration_col else 0.0
            wait_time = _safe_float(row[wait_col]) if wait_col else 0.0
            block_dim = int(_safe_float(row[block_dim_col]) or 0) if block_dim_col else 0

            pipeline = _build_pipeline_stage(fieldnames, row)
            invocations.append(
                KernelInvocation(
                    op_name=op_name,
                    duration_us=duration or 0.0,
                    wait_time_us=wait_time or 0.0,
                    block_dim=block_dim,
                    pipeline=pipeline,
                )
            )

    return invocations


def _build_pipeline_stage(fieldnames: Sequence[str], row: dict[str, str]) -> PipelineStage | None:
    has_any = False
    for candidates in _PIPELINE_FIELDS.values():
        for col in candidates:
            if col in fieldnames and row.get(col, "").strip():
                has_any = True
                break

    if not has_any:
        return None

    return PipelineStage(
        aic_mac_ratio=_safe_float(row.get("aic_mac_ratio", row.get("aic_mac_ratio"))) or 0.0,
        aic_scalar_ratio=_safe_float(row.get("aic_scalar_ratio", row.get("aic_scalar_ratio"))) or 0.0,
        aic_mte1_ratio=_safe_float(row.get("aic_mte1_ratio", row.get("aic_mte1_ratio"))) or 0.0,
        aic_mte2_ratio=_safe_float(row.get("aic_mte2_ratio", row.get("aic_mte2_ratio"))) or 0.0,
        aic_mte3_ratio=_safe_float(row.get("aic_mte3_ratio", row.get("aic_mte3_ratio"))) or 0.0,
        aiv_vec_ratio=_safe_float(row.get("aiv_vec_ratio", row.get("aiv_vec_ratio"))) or 0.0,
        aiv_scalar_ratio=_safe_float(row.get("aiv_scalar_ratio", row.get("aiv_scalar_ratio"))) or 0.0,
        aiv_mte2_ratio=_safe_float(row.get("aiv_mte2_ratio", row.get("aiv_mte2_ratio"))) or 0.0,
        aiv_mte3_ratio=_safe_float(row.get("aiv_mte3_ratio", row.get("aiv_mte3_ratio"))) or 0.0,
        cube_utilization=_safe_float(row.get("cube_utilization(%)", row.get("cube_utilization"))) or 0.0,
        block_dim=int(_safe_float(row.get("Block Dim", "0")) or 0),
    )


def parse_task_time(csv_path: Path) -> list[TaskRecord]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        name_col = _find_column(fieldnames, _TASK_NAME_COLUMNS)
        duration_col = _find_column(fieldnames, _TASK_DURATION_COLUMNS)
        start_col = _find_column(fieldnames, _TASK_START_COLUMNS)
        stop_col = _find_column(fieldnames, _TASK_STOP_COLUMNS)
        type_col = _find_column(fieldnames, _TASK_TYPE_COLUMNS)

        records: list[TaskRecord] = []
        for row in reader:
            kernel_name = (row.get(name_col, "") if name_col else "").strip()
            if not kernel_name or kernel_name == "N/A":
                continue
            records.append(
                TaskRecord(
                    kernel_name=kernel_name,
                    kernel_type=(row.get(type_col, "") if type_col else "").strip(),
                    task_time_us=_safe_float(row[duration_col]) or 0.0 if duration_col else 0.0,
                    task_start_us=_safe_float(row[start_col]) or 0.0 if start_col else 0.0,
                    task_stop_us=_safe_float(row[stop_col]) or 0.0 if stop_col else 0.0,
                )
            )

    return records


def parse_msprof_json(json_path: Path) -> list[dict[str, Any]]:
    data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    if isinstance(data, dict) and "traceEvents" in data:
        return cast(list[dict[str, Any]], data["traceEvents"])
    return []


class MsprofParser:
    """Parser for msprof PROF_*/mindstudio_profiler_output/ artifacts."""

    def parse(
        self,
        artifacts_dir: Path,
    ) -> tuple[
        list[OperatorStats],
        list[KernelInvocation],
        list[TaskRecord],
        list[HostApiCall],
        list[dict[str, Any]],
        dict[str, str | None],
    ]:
        op_statistic_csv = find_newest_csv(artifacts_dir, "op_statistic")
        if op_statistic_csv is None:
            raise FileNotFoundError(f"No op_statistic CSV found in {artifacts_dir}")

        op_summary_csv = find_newest_csv(artifacts_dir, "op_summary")
        task_time_csv = find_newest_csv(artifacts_dir, "task_time")
        api_statistic_csv = find_newest_csv(artifacts_dir, "api_statistic")
        msprof_json_files = sorted(artifacts_dir.glob("msprof_*.json"))

        operators = parse_op_statistic(op_statistic_csv)
        invocations = parse_op_summary(op_summary_csv) if op_summary_csv else []
        task_records = parse_task_time(task_time_csv) if task_time_csv else []
        host_api = parse_api_statistic(api_statistic_csv) if api_statistic_csv else []

        timeline: list[dict[str, Any]] = []
        if msprof_json_files:
            timeline = parse_msprof_json(msprof_json_files[-1])

        source_files: dict[str, str | None] = {
            "op_statistic": op_statistic_csv.name,
            "op_summary": op_summary_csv.name if op_summary_csv else None,
            "task_time": task_time_csv.name if task_time_csv else None,
            "api_statistic": api_statistic_csv.name if api_statistic_csv else None,
            "msprof_json": msprof_json_files[-1].name if msprof_json_files else None,
        }

        return operators, invocations, task_records, host_api, timeline, source_files
