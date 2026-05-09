import json
from pathlib import Path
from typing import Any, cast

from models import (
    BoundClassificationKind,
    CoreTypeAggregate,
    HostApiCall,
    HostApiSummary,
    KernelInvocation,
    OperatorStats,
    OperatorTypeKind,
    ParsedProfile,
    TaskTimelineSummary,
)
from parser_base import detect_profile_mode
from msprof_parser import MsprofParser
from standalone_parser import StandaloneParser

_TRANSFER_HINTS = (
    "copy", "memcpy", "transdata", "dma", "load", "store", "move",
)


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


def _normalize_core_type(core_type: str) -> str:
    lowered = core_type.strip().lower()
    if "scalar" in lowered:
        return "scalar"
    if "vector" in lowered:
        return "vector"
    if "cube" in lowered or "aicore" in lowered or "ai_core" in lowered:
        return "cube"
    return "other"


def _is_data_movement_op(op_type: str) -> bool:
    lowered = op_type.lower()
    return any(hint in lowered for hint in _TRANSFER_HINTS)


def _compute_data_movement_hotspots(operators: list[OperatorStats]) -> list[OperatorStats]:
    hotspots = [op for op in operators if _is_data_movement_op(op.op_type)]
    return sorted(hotspots, key=lambda op: op.total_time_us, reverse=True)


def _classify_operator_type(
    core_type: str,
    ratio_avgs: dict[str, float],
    cube_utilization_avg: float | None,
) -> tuple[OperatorTypeKind, list[str], str]:
    vector_ratio = ratio_avgs.get("aiv_vec_ratio", 0.0)
    cube_ratio = max(ratio_avgs.get("aic_mac_ratio", 0.0), cube_utilization_avg or 0.0)
    signals: list[str] = []
    if vector_ratio >= 30.0:
        signals.append(f"high aiv_vec_ratio={_format_number(vector_ratio)}")
    if cube_ratio >= 20.0:
        signals.append(f"high cube-side activity={_format_number(cube_ratio)}")

    if vector_ratio >= 30.0 and cube_ratio >= 20.0:
        kind: OperatorTypeKind = "mix"
    elif cube_ratio >= 20.0:
        kind = "cube"
    elif vector_ratio >= 20.0:
        kind = "vector"
    else:
        kind: OperatorTypeKind = cast(OperatorTypeKind, _normalize_core_type(core_type))
        signals.append(f"fallback to target core type {core_type}")

    source = "op_summary" if signals and "fallback" not in signals[-1] else "op_statistic"
    return kind, signals, source


def _classify_bound(ratio_avgs: dict[str, float]) -> tuple[BoundClassificationKind, dict[str, float], list[str]]:
    compute_score = ratio_avgs.get("aic_mac_ratio", 0.0) + ratio_avgs.get("aiv_vec_ratio", 0.0)
    memory_score = (
        ratio_avgs.get("aic_mte1_ratio", 0.0)
        + ratio_avgs.get("aic_mte2_ratio", 0.0)
        + ratio_avgs.get("aic_mte3_ratio", 0.0)
        + ratio_avgs.get("aiv_mte2_ratio", 0.0)
        + ratio_avgs.get("aiv_mte3_ratio", 0.0)
    )
    scalar_score = ratio_avgs.get("aic_scalar_ratio", 0.0) + ratio_avgs.get("aiv_scalar_ratio", 0.0)

    scores = {"compute": compute_score, "memory": memory_score, "scalar": scalar_score}

    if scalar_score >= max(compute_score, memory_score) and scalar_score >= 20.0:
        classification: BoundClassificationKind = "scalar-overhead"
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

    return classification, scores, reasoning


def _select_target(operators: list[OperatorStats], target_op: str | None) -> tuple[OperatorStats, bool]:
    if target_op is not None:
        for op in operators:
            if op.op_type == target_op:
                return op, False
        raise ValueError(f"Target operator not found in op_statistic: {target_op}")
    return max(operators, key=lambda op: op.total_time_us), True


def _aggregate_core_types(operators: list[OperatorStats]) -> CoreTypeAggregate:
    agg = CoreTypeAggregate()
    for op in operators:
        bucket = _normalize_core_type(op.core_type)
        agg.raw_core_types.setdefault(bucket, [])
        if op.core_type not in agg.raw_core_types[bucket]:
            agg.raw_core_types[bucket].append(op.core_type)
        if bucket == "cube":
            agg.cube_total_us += op.total_time_us
            agg.cube_ratio_pct += op.ratio_percent
        elif bucket == "vector":
            agg.vector_total_us += op.total_time_us
            agg.vector_ratio_pct += op.ratio_percent
        elif bucket == "scalar":
            agg.scalar_total_us += op.total_time_us
            agg.scalar_ratio_pct += op.ratio_percent
        else:
            agg.other_total_us += op.total_time_us
            agg.other_ratio_pct += op.ratio_percent
    return agg


def _compute_pipeline_averages(invocations: list[KernelInvocation], target_op: str) -> dict[str, float]:
    ratio_values: dict[str, list[float]] = {
        "aic_mac_ratio": [],
        "aic_scalar_ratio": [],
        "aic_mte1_ratio": [],
        "aic_mte2_ratio": [],
        "aic_mte3_ratio": [],
        "aiv_vec_ratio": [],
        "aiv_scalar_ratio": [],
        "aiv_mte2_ratio": [],
        "aiv_mte3_ratio": [],
    }
    for inv in invocations:
        if inv.op_name != target_op or inv.pipeline is None:
            continue
        ratio_values["aic_mac_ratio"].append(inv.pipeline.aic_mac_ratio)
        ratio_values["aic_scalar_ratio"].append(inv.pipeline.aic_scalar_ratio)
        ratio_values["aic_mte1_ratio"].append(inv.pipeline.aic_mte1_ratio)
        ratio_values["aic_mte2_ratio"].append(inv.pipeline.aic_mte2_ratio)
        ratio_values["aic_mte3_ratio"].append(inv.pipeline.aic_mte3_ratio)
        ratio_values["aiv_vec_ratio"].append(inv.pipeline.aiv_vec_ratio)
        ratio_values["aiv_scalar_ratio"].append(inv.pipeline.aiv_scalar_ratio)
        ratio_values["aiv_mte2_ratio"].append(inv.pipeline.aiv_mte2_ratio)
        ratio_values["aiv_mte3_ratio"].append(inv.pipeline.aiv_mte3_ratio)

    return {key: sum(vals) / len(vals) if vals else 0.0 for key, vals in ratio_values.items()}


def _check_launch_related(host_api: list[HostApiCall]) -> bool:
    launch_keywords = ("launch", "memcpy", "synchronize", "binary", "stream")
    return any(
        any(kw in call.api_name.lower() for kw in launch_keywords)
        for call in host_api
    )


class ProfileReporter:
    def load_profile(
        self,
        profile_path: Path,
        target_op: str | None = None,
        top_count: int = 5,
    ) -> ParsedProfile:
        mode, artifacts_dir = detect_profile_mode(profile_path)

        if mode == "msprof":
            parser = MsprofParser()
            operators, invocations, task_records, host_api, _timeline, source_files = parser.parse(artifacts_dir)
            torch_ops = None
            step_traces = None
        else:
            parser = StandaloneParser()
            operators, invocations, torch_ops, step_traces, host_api, _timeline, source_files = parser.parse(artifacts_dir)
            task_records = None

        target, inferred = _select_target(operators, target_op)
        top_ops = sorted(operators, key=lambda op: op.total_time_us, reverse=True)[:top_count]
        core_type_agg = _aggregate_core_types(operators)

        ratio_avgs = _compute_pipeline_averages(invocations, target.op_type)

        cube_utilization_values = [
            inv.pipeline.cube_utilization
            for inv in invocations
            if inv.op_name == target.op_type and inv.pipeline is not None
        ]
        cube_util_avg = sum(cube_utilization_values) / len(cube_utilization_values) if cube_utilization_values else None

        op_type_kind, op_type_signals, op_type_source = _classify_operator_type(
            target.core_type, ratio_avgs, cube_util_avg
        )
        bound_kind, bound_scores, bound_reasoning = _classify_bound(ratio_avgs)

        task_timeline = None
        if task_records:
            target_tasks = [t for t in task_records if t.kernel_name == target.op_type]
            if target_tasks:
                total_time = sum(t.task_time_us for t in target_tasks)
                starts = [t.task_start_us for t in target_tasks]
                stops = [t.task_stop_us for t in target_tasks]
                gap = 0.0
                max_gap = 0.0
                overlap = 0
                for i in range(1, len(target_tasks)):
                    current_gap = starts[i] - stops[i - 1]
                    if current_gap < 0:
                        overlap += 1
                    else:
                        gap += current_gap
                        if current_gap > max_gap:
                            max_gap = current_gap
                task_timeline = TaskTimelineSummary(
                    matched_rows=len(target_tasks),
                    total_task_time_us=total_time,
                    span_us=stops[-1] - starts[0] if starts and stops else None,
                    total_gap_us=gap,
                    max_gap_us=max_gap,
                    overlap_count=overlap,
                )

        host_api_summary = HostApiSummary(launch_related_present=_check_launch_related(host_api))
        stream_like_tracks = len({e.get("tid") for e in _timeline if e.get("tid") is not None}) if _timeline else 0

        return ParsedProfile(
            bench_mode=mode,
            profile_dir=str(profile_path.resolve()),
            source_files=source_files,
            operators=operators,
            invocations=invocations,
            host_api_calls=host_api,
            task_records=task_records,
            torch_op_timing=torch_ops,
            step_trace=step_traces,
            target=target,
            target_inferred=inferred,
            top_operators=top_ops,
            core_type_aggregate=core_type_agg,
            bound_classification=bound_kind,
            bound_scores=bound_scores,
            bound_reasoning=bound_reasoning,
            operator_type=op_type_kind,
            operator_type_signals=op_type_signals,
            operator_type_source=op_type_source,
            task_timeline=task_timeline,
            host_api_summary=host_api_summary,
            stream_like_tracks=stream_like_tracks,
        )

    def build_report(
        self,
        profile_path: str | Path,
        target_op: str | None = None,
        top_count: int = 5,
        output_format: str = "markdown",
    ) -> str:
        profile = self.load_profile(
            Path(profile_path), target_op=target_op, top_count=top_count
        )
        if output_format == "json":
            return self._render_json(profile)
        if output_format != "markdown":
            raise ValueError(f"Unsupported output format: {output_format}")
        return self._render_markdown(profile)

    def _render_json(self, profile: ParsedProfile) -> str:
        return json.dumps(self._build_json_payload(profile), indent=2, sort_keys=True) + "\n"

    def _build_json_payload(self, profile: ParsedProfile) -> dict[str, Any]:
        target = profile.target
        assert target is not None

        core_type_totals: dict[str, Any] = {}
        agg = profile.core_type_aggregate
        if agg:
            for bucket, count_val, total_val, ratio_val, raw_types in [
                ("cube", 0, agg.cube_total_us, agg.cube_ratio_pct, agg.raw_core_types.get("cube", [])),
                ("vector", 0, agg.vector_total_us, agg.vector_ratio_pct, agg.raw_core_types.get("vector", [])),
                ("scalar", 0, agg.scalar_total_us, agg.scalar_ratio_pct, agg.raw_core_types.get("scalar", [])),
                ("other", 0, agg.other_total_us, agg.other_ratio_pct, agg.raw_core_types.get("other", [])),
            ]:
                if total_val > 0 or raw_types:
                    core_type_totals[bucket] = {
                        "total_time_us": total_val,
                        "ratio_percent": ratio_val,
                        "count": count_val,
                        "raw_core_types": raw_types,
                    }

        data_movement = _compute_data_movement_hotspots(profile.operators)
        invocations_for_target = [inv for inv in profile.invocations if inv.op_name == target.op_type]
        duration_values = [inv.duration_us for inv in invocations_for_target if inv.duration_us > 0]

        op_summary_stats: dict[str, Any] = {
            "matched_rows": len(invocations_for_target),
            "total_duration_us": sum(duration_values) if duration_values else None,
            "avg_duration_us": (sum(duration_values) / len(duration_values)) if duration_values else None,
            "min_duration_us": min(duration_values) if duration_values else None,
            "max_duration_us": max(duration_values) if duration_values else None,
            "note": None,
        }

        task_timeline_signals: dict[str, Any] = {"matched_rows": 0, "max_gap_us": 0.0}
        if profile.task_timeline:
            tt = profile.task_timeline
            task_timeline_signals = {
                "matched_rows": tt.matched_rows,
                "max_gap_us": tt.max_gap_us,
            }

        host_api_signals: dict[str, Any] = {
            "launch_related_present": False,
            "top_apis": [],
        }
        if profile.host_api_summary:
            host_api_signals["launch_related_present"] = profile.host_api_summary.launch_related_present
        if profile.host_api_calls:
            top_apis = sorted(profile.host_api_calls, key=lambda c: c.time_us, reverse=True)[:10]
            host_api_signals["top_apis"] = [
                {"api_name": c.api_name, "time_us": c.time_us, "count": c.count, "avg_us": c.avg_us}
                for c in top_apis
            ]

        pipeline_signals: dict[str, Any] = {}
        target_invocations = [inv for inv in profile.invocations if inv.op_name == target.op_type]
        if target_invocations:
            ratio_collect: dict[str, list[float]] = {}
            wait_values: list[float] = []
            cube_values: list[float] = []
            block_dims: set[int] = set()
            for inv in target_invocations:
                if inv.wait_time_us > 0:
                    wait_values.append(inv.wait_time_us)
                if inv.block_dim > 0:
                    block_dims.add(inv.block_dim)
                if inv.pipeline is not None:
                    p = inv.pipeline
                    for name, val in [
                        ("aic_mac_ratio", p.aic_mac_ratio),
                        ("aic_scalar_ratio", p.aic_scalar_ratio),
                        ("aic_mte1_ratio", p.aic_mte1_ratio),
                        ("aic_mte2_ratio", p.aic_mte2_ratio),
                        ("aic_mte3_ratio", p.aic_mte3_ratio),
                        ("aiv_vec_ratio", p.aiv_vec_ratio),
                        ("aiv_scalar_ratio", p.aiv_scalar_ratio),
                        ("aiv_mte2_ratio", p.aiv_mte2_ratio),
                        ("aiv_mte3_ratio", p.aiv_mte3_ratio),
                    ]:
                        ratio_collect.setdefault(name, []).append(val)
                    cube_values.append(p.cube_utilization)
            if ratio_collect:
                pipeline_signals["ratios"] = {
                    name: {
                        "avg": sum(vals) / len(vals),
                        "count": len(vals),
                        "min": min(vals),
                        "max": max(vals),
                        "total": sum(vals),
                    }
                    for name, vals in ratio_collect.items()
                }
            if wait_values:
                pipeline_signals["task_wait_time_us"] = {
                    "avg": sum(wait_values) / len(wait_values),
                    "count": len(wait_values),
                    "min": min(wait_values),
                    "max": max(wait_values),
                    "total": sum(wait_values),
                }
            if cube_values:
                pipeline_signals["cube_utilization_percent"] = {
                    "avg": sum(cube_values) / len(cube_values),
                    "count": len(cube_values),
                    "min": min(cube_values),
                    "max": max(cube_values),
                    "total": sum(cube_values),
                }
            if block_dims:
                pipeline_signals["block_dim"] = {
                    "observed_values": sorted(block_dims),
                }

        msprof_timeline_signals: dict[str, Any] = {
            "stream_like_tracks": profile.stream_like_tracks,
        }

        return {
            "profile_dir": profile.profile_dir,
            "op_statistic_file": profile.source_files.get("op_statistic"),
            "op_summary_file": profile.source_files.get("op_summary") or profile.source_files.get("kernel_details"),
            "task_time_file": profile.source_files.get("task_time"),
            "api_statistic_file": profile.source_files.get("api_statistic"),
            "msprof_json_file": profile.source_files.get("msprof_json") or profile.source_files.get("trace_view"),
            "target_operator": target.op_type,
            "selection": (
                "inferred from the hottest `op_statistic` row by `Total Time(us)`"
                if profile.target_inferred
                else "matched the explicit `--target-op` value"
            ),
            "target_row": {
                "op_type": target.op_type,
                "core_type": target.core_type,
                "count": float(target.count),
                "total_time_us": target.total_time_us,
                "min_time_us": target.min_time_us,
                "avg_time_us": target.avg_time_us,
                "max_time_us": target.max_time_us,
                "ratio_percent": target.ratio_percent,
            },
            "op_summary": op_summary_stats,
            "core_type_totals": core_type_totals,
            "data_movement_hotspots": [
                {"op_type": op.op_type, "core_type": op.core_type,
                 "total_time_us": op.total_time_us, "ratio_percent": op.ratio_percent,
                 "count": float(op.count), "min_time_us": op.min_time_us,
                 "avg_time_us": op.avg_time_us, "max_time_us": op.max_time_us}
                for op in data_movement
            ],
            "top_ops": [
                {"op_type": op.op_type, "core_type": op.core_type,
                 "count": float(op.count), "total_time_us": op.total_time_us,
                 "min_time_us": op.min_time_us, "avg_time_us": op.avg_time_us,
                 "max_time_us": op.max_time_us, "ratio_percent": op.ratio_percent}
                for op in profile.top_operators
            ],
            "operator_type_guess": {
                "kind": profile.operator_type,
                "signals": profile.operator_type_signals,
                "source": profile.operator_type_source,
            },
            "bound_analysis": {
                "classification": profile.bound_classification,
                "scores": profile.bound_scores,
                "reasoning": profile.bound_reasoning,
            },
            "pipeline_signals": pipeline_signals,
            "task_timeline_signals": task_timeline_signals,
            "host_api_signals": host_api_signals,
            "msprof_timeline_signals": msprof_timeline_signals,
            "binary_signals": {},
        }

    def _render_markdown(self, profile: ParsedProfile) -> str:
        target = profile.target
        assert target is not None
        invocations_for_target = [inv for inv in profile.invocations if inv.op_name == target.op_type]
        duration_values = [inv.duration_us for inv in invocations_for_target if inv.duration_us > 0]

        top_rows = [
            [op.op_type, op.core_type,
             _format_number(float(op.count)), _format_number(op.total_time_us),
             _format_number(op.avg_time_us), _format_number(op.ratio_percent)]
            for op in profile.top_operators
        ]

        agg = profile.core_type_aggregate
        core_rows: list[list[str]] = []
        if agg:
            for bucket, total_val, ratio_val, raw_types in [
                ("cube", agg.cube_total_us, agg.cube_ratio_pct, agg.raw_core_types.get("cube", [])),
                ("vector", agg.vector_total_us, agg.vector_ratio_pct, agg.raw_core_types.get("vector", [])),
                ("scalar", agg.scalar_total_us, agg.scalar_ratio_pct, agg.raw_core_types.get("scalar", [])),
                ("other", agg.other_total_us, agg.other_ratio_pct, agg.raw_core_types.get("other", [])),
            ]:
                if total_val > 0 or raw_types:
                    count = sum(1 for op in profile.operators if _normalize_core_type(op.core_type) == bucket)
                    core_rows.append([
                        bucket,
                        ", ".join(sorted(raw_types)),
                        _format_number(float(count)),
                        _format_number(total_val),
                        _format_number(ratio_val),
                    ])

        movement = _compute_data_movement_hotspots(profile.operators)
        movement_rows = [
            [op.op_type, op.core_type, _format_number(op.total_time_us), _format_number(op.ratio_percent)]
            for op in movement
        ]

        op_summary_file = profile.source_files.get("op_summary") or profile.source_files.get("kernel_details")
        matched_rows = len(invocations_for_target)

        source_files = profile.source_files
        lines = [
            "# Ascend NPU Operator Profile Summary",
            "",
            f"- Profile directory: `{profile.profile_dir}`",
            f"- `op_statistic` file: `{source_files.get('op_statistic', 'not found')}`",
            (
                f"- `op_summary` file: `{op_summary_file}`"
                if op_summary_file
                else "- `op_summary` file: `not found`"
            ),
            f"- Target operator: `{target.op_type}`",
            f"- Selection: {('inferred from the hottest `op_statistic` row by `Total Time(us)`' if profile.target_inferred else 'matched the explicit `--target-op` value')}",
            "",
            "## Operator timing",
            "",
            f"- Core type: `{target.core_type}`",
            f"- Invocation count: `{_format_number(float(target.count))}`",
            f"- Total time: `{_format_number(target.total_time_us)} us`",
            f"- Average time: `{_format_number(target.avg_time_us)} us`",
            f"- Min time: `{_format_number(target.min_time_us)} us`",
            f"- Max time: `{_format_number(target.max_time_us)} us`",
            f"- Runtime ratio: `{_format_number(target.ratio_percent)}%`",
            "",
            "## op_summary cross-check",
            "",
            f"- Matched op_summary rows: `{matched_rows}`",
        ]

        if duration_values:
            lines.extend([
                f"- Summed task duration: `{_format_number(sum(duration_values))} us`",
                f"- Average task duration: `{_format_number(sum(duration_values) / len(duration_values))} us`",
                f"- Min task duration: `{_format_number(min(duration_values))} us`",
                f"- Max task duration: `{_format_number(max(duration_values))} us`",
            ])
        elif not invocations_for_target:
            lines.append("- Note: No invocations matched the target operator.")

        lines.extend([
            f"- Operator type guess: `{profile.operator_type}`",
            f"- Bound analysis: `{profile.bound_classification}`",
        ])

        if profile.task_timeline:
            lines.append(f"- Avg task wait time: `N/A`")

        task_matched = profile.task_timeline.matched_rows if profile.task_timeline else 0
        task_max_gap = profile.task_timeline.max_gap_us if profile.task_timeline else 0.0
        launch_present = profile.host_api_summary.launch_related_present if profile.host_api_summary else False

        lines.extend([
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
            f"- Task timeline matched rows: `{task_matched}`",
            f"- Max task gap: `{_format_number(task_max_gap)} us`",
            f"- Host launch-related APIs present: `{launch_present}`",
            f"- msprof tracks: `0`",
            f"- Binary signals available: `False`",
            "",
            "## Top operators by total time",
            "",
            _markdown_table(
                ["OP Type", "Core Type", "Count", "Total Time(us)", "Avg Time(us)", "Ratio(%)"],
                top_rows,
            ),
        ])

        return "\n".join(lines) + "\n"


def build_report(
    profile_path: str | Path,
    target_op: str | None = None,
    top_count: int = 5,
    output_format: str = "markdown",
) -> str:
    return ProfileReporter().build_report(
        profile_path,
        target_op=target_op,
        top_count=top_count,
        output_format=output_format,
    )
