from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, cast

from collections.abc import Sequence

from models import HostApiCall, KernelInvocation, OperatorStats, PipelineStage, StepTrace, TorchOpTiming
from parser_base import find_newest_csv, parse_api_statistic, parse_op_statistic


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


def parse_kernel_details(csv_path: Path) -> list[KernelInvocation]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        # torch-npu-profiler uses: Name, Type, Accelerator Core, Duration(us), Wait Time(us), Block Dim
        name_col = "Name" if "Name" in fieldnames else None
        duration_col = "Duration(us)" if "Duration(us)" in fieldnames else None
        wait_col = "Wait Time(us)" if "Wait Time(us)" in fieldnames else None
        block_dim_col = "Block Dim" if "Block Dim" in fieldnames else None

        invocations: list[KernelInvocation] = []
        for row in reader:
            op_name = (row.get(name_col, "") if name_col else "").strip()
            if not op_name:
                continue
            duration = _safe_float(row[duration_col]) if duration_col else None
            wait_time = _safe_float(row[wait_col]) if wait_col else None
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


_PIPELINE_STAGE_FIELDS = [
    "aic_mac_ratio",
    "aic_scalar_ratio",
    "aic_mte1_ratio",
    "aic_mte2_ratio",
    "aic_mte3_ratio",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte2_ratio",
    "aiv_mte3_ratio",
]


def _build_pipeline_stage(fieldnames: Sequence[str], row: dict[str, str]) -> PipelineStage | None:
    has_any = any(
        col in fieldnames and row.get(col, "").strip()
        for col in _PIPELINE_STAGE_FIELDS
    )
    if not has_any:
        return None

    return PipelineStage(
        aic_mac_ratio=_safe_float(row.get("aic_mac_ratio")) or 0.0,
        aic_scalar_ratio=_safe_float(row.get("aic_scalar_ratio")) or 0.0,
        aic_mte1_ratio=_safe_float(row.get("aic_mte1_ratio")) or 0.0,
        aic_mte2_ratio=_safe_float(row.get("aic_mte2_ratio")) or 0.0,
        aic_mte3_ratio=_safe_float(row.get("aic_mte3_ratio")) or 0.0,
        aiv_vec_ratio=_safe_float(row.get("aiv_vec_ratio")) or 0.0,
        aiv_scalar_ratio=_safe_float(row.get("aiv_scalar_ratio")) or 0.0,
        aiv_mte2_ratio=_safe_float(row.get("aiv_mte2_ratio")) or 0.0,
        aiv_mte3_ratio=_safe_float(row.get("aiv_mte3_ratio")) or 0.0,
        cube_utilization=_safe_float(row.get("cube_utilization(%)")) or 0.0,
        block_dim=int(_safe_float(row.get("Block Dim", "0")) or 0),
    )


def parse_operator_details(csv_path: Path) -> list[TorchOpTiming]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        has_device = "Device Self Duration(us)" in fieldnames
        has_aicore = "Device Self Duration With AICore(us)" in fieldnames

        ops: list[TorchOpTiming] = []
        for row in reader:
            name = row.get("Name", "").strip()
            if not name:
                continue
            host_self = _safe_float(row.get("Host Self Duration(us)")) or 0.0
            host_total = _safe_float(row.get("Host Total Duration(us)")) or 0.0
            device_self = _safe_float(row.get("Device Self Duration(us)")) if has_device else 0.0
            device_total = _safe_float(row.get("Device Total Duration(us)")) if has_device else 0.0
            if has_aicore:
                aicore_self = _safe_float(row.get("Device Self Duration With AICore(us)")) or 0.0
                aicore_total = _safe_float(row.get("Device Total Duration With AICore(us)")) or 0.0
                if aicore_self > 0:
                    device_self = aicore_self
                    device_total = aicore_total

            ops.append(
                TorchOpTiming(
                    name=name,
                    host_self_us=host_self,
                    host_total_us=host_total,
                    device_self_us=device_self or 0.0,
                    device_total_us=device_total or 0.0,
                )
            )

    return ops


def parse_step_trace(csv_path: Path) -> list[StepTrace]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        traces: list[StepTrace] = []
        for row in reader:
            traces.append(
                StepTrace(
                    step=int(row.get("Step", "0")),
                    computing_us=_safe_float(row.get("Computing")) or 0.0,
                    communication_not_overlapped_us=_safe_float(
                        row.get("Communication(Not Overlapped)")
                    ) or 0.0,
                    overlapped_us=_safe_float(row.get("Overlapped")) or 0.0,
                    communication_us=_safe_float(row.get("Communication")) or 0.0,
                    free_us=_safe_float(row.get("Free")) or 0.0,
                    stage_us=_safe_float(row.get("Stage")) or 0.0,
                    bubble_us=_safe_float(row.get("Bubble")) or 0.0,
                    communication_not_overlapped_exclude_receive_us=_safe_float(
                        row.get("Communication(Not Overlapped and Exclude Receive)")
                    ) or 0.0,
                    preparing_us=_safe_float(row.get("Preparing")) or 0.0,
                )
            )
    return traces


def parse_trace_view(json_path: Path) -> list[dict[str, Any]]:
    data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return cast(list[dict[str, Any]], data)
    if isinstance(data, dict) and "traceEvents" in data:
        return cast(list[dict[str, Any]], data["traceEvents"])
    return []


class TorchNpuProfilerParser:
    """Parser for torch_npu.profiler ASCEND_PROFILER_OUTPUT/ artifacts."""

    def parse(
        self,
        artifacts_dir: Path,
    ) -> tuple[
        list[OperatorStats],
        list[KernelInvocation],
        list[TorchOpTiming],
        list[StepTrace],
        list[HostApiCall],
        list[dict[str, Any]],
        dict[str, str | None],
    ]:
        op_statistic_csv = find_newest_csv(artifacts_dir, "op_statistic")
        if op_statistic_csv is None:
            raise FileNotFoundError(f"No op_statistic CSV found in {artifacts_dir}")

        kernel_details_csv = find_newest_csv(artifacts_dir, "kernel_details")
        operator_details_csv = find_newest_csv(artifacts_dir, "operator_details")
        step_trace_csv = find_newest_csv(artifacts_dir, "step_trace_time")
        api_statistic_csv = find_newest_csv(artifacts_dir, "api_statistic")
        trace_view_files = sorted(artifacts_dir.glob("trace_view.json"))

        operators = parse_op_statistic(op_statistic_csv)
        invocations = parse_kernel_details(kernel_details_csv) if kernel_details_csv else []
        torch_ops = parse_operator_details(operator_details_csv) if operator_details_csv else []
        step_traces = parse_step_trace(step_trace_csv) if step_trace_csv else []
        host_api = parse_api_statistic(api_statistic_csv) if api_statistic_csv else []

        timeline: list[dict[str, Any]] = []
        if trace_view_files:
            timeline = parse_trace_view(trace_view_files[-1])

        source_files: dict[str, str | None] = {
            "op_statistic": op_statistic_csv.name,
            "kernel_details": kernel_details_csv.name if kernel_details_csv else None,
            "operator_details": operator_details_csv.name if operator_details_csv else None,
            "step_trace_time": step_trace_csv.name if step_trace_csv else None,
            "api_statistic": api_statistic_csv.name if api_statistic_csv else None,
            "trace_view": trace_view_files[-1].name if trace_view_files else None,
        }

        return operators, invocations, torch_ops, step_traces, host_api, timeline, source_files
