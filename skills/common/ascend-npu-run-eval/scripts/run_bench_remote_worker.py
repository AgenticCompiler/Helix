"""Self-contained remote worker for benchmark runtime actions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import shutil
from pathlib import Path

import run_bench_execution
from profile_csv_parser import find_latest_op_statistic_csv, parse_op_statistic_csv, resolve_perf_metrics
from perf_artifacts import PerfCaseRecord


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument(
        "action",
        choices=(
            "case-plan",
            "profile-all",
            "perf-counter-all",
            "profile-case",
            "perf-counter-case",
            "msprof-metrics",
        ),
    )
    parser.add_argument("--bench-file")
    parser.add_argument("--operator-file")
    parser.add_argument("--output")
    parser.add_argument("--case-id")
    parser.add_argument("--preserved-run-dir")
    parser.add_argument("--warmup-cap", type=int)
    parser.add_argument("--repeats-cap", type=int)
    parser.add_argument("--metrics-root")
    parser.add_argument("--kernel-names")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "msprof-metrics":
        return _emit_msprof_metrics(args)
    bench_file, operator_file = _require_inputs(args)
    if args.action == "case-plan":
        return _emit_case_plan(bench_file, operator_file)
    if args.action == "profile-all":
        return _profile_all(args, bench_file, operator_file)
    if args.action == "perf-counter-all":
        return _perf_counter_all(args, bench_file, operator_file)
    if args.action == "profile-case":
        return _profile_case(args, bench_file, operator_file)
    return _perf_counter_case(args, bench_file, operator_file)


def _require_inputs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.bench_file is None or args.operator_file is None:
        raise ValueError("--bench-file and --operator-file are required for benchmark execution")
    return Path(args.bench_file), Path(args.operator_file)


def _profile_all(args: argparse.Namespace, bench_file: Path, operator_file: Path) -> int:
    preloaded = None
    if args.warmup_cap is not None or args.repeats_cap is not None:
        if args.warmup_cap is None or args.repeats_cap is None:
            raise ValueError("Both execution limits are required together")
        cases, resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
        preloaded = (
            [
                replace(case, warmup=min(case.warmup, args.warmup_cap), repeats=min(case.repeats, args.repeats_cap))
                for case in cases
            ],
            resolution,
        )
    result, perf_path = run_bench_execution.profile_all_bench_cases(
        bench_file, operator_file, preloaded=preloaded, verbose=args.verbose
    )
    _copy_output(perf_path, args.output)
    return int(result["return_code"])


def _emit_case_plan(bench_file: Path, operator_file: Path) -> int:
    cases, resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
    print(
        json.dumps(
            {
                "case_ids": [case.case_id for case in cases],
                "iterations_by_case": {case.case_id: case.warmup + case.repeats for case in cases},
                "kernel_names": resolution.kernel_names,
                "kernel_source": resolution.kernel_source,
            },
            separators=(",", ":"),
        )
    )
    return 0


def _perf_counter_all(args: argparse.Namespace, bench_file: Path, operator_file: Path) -> int:
    result, perf_path = run_bench_execution.time_all_bench_cases(
        bench_file, operator_file, bench_mode="perf-counter"
    )
    _copy_output(perf_path, args.output)
    return int(result["return_code"])


def _profile_case(args: argparse.Namespace, bench_file: Path, operator_file: Path) -> int:
    if args.case_id is None:
        raise ValueError("--case-id is required for profile-case")
    preserved = None if args.preserved_run_dir in {None, "__NONE__"} else Path(args.preserved_run_dir)
    record = run_bench_execution.profile_bench_case(
        bench_file, operator_file, args.case_id, preserved_run_dir=preserved, verbose=args.verbose
    )
    _emit_case_record(record)
    return 0


def _perf_counter_case(args: argparse.Namespace, bench_file: Path, operator_file: Path) -> int:
    if args.case_id is None:
        raise ValueError("--case-id is required for perf-counter-case")
    cases, resolution = run_bench_execution.load_bench_cases(bench_file, operator_file)
    case = run_bench_execution.select_bench_case(cases, args.case_id)
    record = run_bench_execution.time_bench_case(case, resolution, bench_mode="perf-counter")
    _emit_case_record(record)
    return 0


def _emit_case_record(record: PerfCaseRecord) -> None:
    print(
        json.dumps(
            {
                "case_label": record.case_label,
                "kernel_names": record.kernel_names,
                "kernel_source": record.kernel_source,
                "metrics": record.metrics,
                "error_message": record.error_message,
                "case_wall_clock_seconds": record.case_wall_clock_seconds,
                "bench_mode": getattr(record, "bench_mode", None),
            },
            separators=(",", ":"),
        )
    )


def _emit_msprof_metrics(args: argparse.Namespace) -> int:
    if args.metrics_root is None or args.kernel_names is None:
        raise ValueError("--metrics-root and --kernel-names are required for msprof-metrics")
    csv_path = find_latest_op_statistic_csv(Path(args.metrics_root))
    if csv_path is None:
        raise FileNotFoundError(f"No op_statistic_*.csv or op_statistic.csv found under {args.metrics_root}")
    rows = parse_op_statistic_csv(csv_path)
    metrics = resolve_perf_metrics(
        rows.ops,
        json.loads(args.kernel_names),
        total_op_avg_time_us=rows.total_op_avg_time_us,
    )
    print(json.dumps(metrics, separators=(",", ":")))
    return 0


def _copy_output(perf_path: Path, output: str | None) -> None:
    if output is None:
        return
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if perf_path != target:
        shutil.copyfile(perf_path, target)


if __name__ == "__main__":
    raise SystemExit(main())
