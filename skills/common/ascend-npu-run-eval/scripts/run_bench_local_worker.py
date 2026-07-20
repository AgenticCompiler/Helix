"""Isolated local worker for canonical benchmark execution."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

from run_bench_modes import execute_local_bench
from perf_artifacts import PerfCaseRecord
from result_payload import ResultPayload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    parser.add_argument("action", choices=("run-all", "profile-case", "perf-counter-case"), nargs="?", default="run-all")
    parser.add_argument("--bench-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--bench-mode")
    parser.add_argument("--result-file")
    parser.add_argument("--case-id")
    parser.add_argument("--preserved-run-dir")
    parser.add_argument("--npu-devices")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--warmup-cap", type=int)
    parser.add_argument("--repeats-cap", type=int)
    args = parser.parse_args(argv)
    bench_file = Path(args.bench_file).expanduser().resolve()
    operator_file = Path(args.operator_file).expanduser().resolve()
    if args.action == "profile-case":
        _run_profile_case(args, bench_file, operator_file)
        return 0
    if args.action == "perf-counter-case":
        _run_perf_counter_case(args, bench_file, operator_file)
        return 0
    if args.bench_mode is None or args.result_file is None:
        parser.error("run-all requires --bench-mode and --result-file")
    result, perf_path = execute_local_bench(
        bench_file,
        operator_file,
        args.bench_mode,
        args.npu_devices,
        verbose=bool(args.verbose),
        output=args.output,
        execution_limits=_execution_limits(args),
    )
    _write_payload(Path(args.result_file), result, perf_path)
    return 0


def _execution_limits(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.warmup_cap is None and args.repeats_cap is None:
        return None
    if args.warmup_cap is None or args.repeats_cap is None:
        raise ValueError("Both execution limits are required together")
    return args.warmup_cap, args.repeats_cap


def _run_profile_case(args: argparse.Namespace, bench_file: Path, operator_file: Path) -> None:
    if args.case_id is None:
        raise ValueError("--case-id is required for profile-case")
    preserved = None if args.preserved_run_dir in {None, "__NONE__"} else Path(args.preserved_run_dir)
    record = _workspace_run_bench_execution().profile_bench_case(
        bench_file,
        operator_file,
        args.case_id,
        preserved_run_dir=preserved,
        verbose=bool(args.verbose),
    )
    _emit_case_record(record)


def _run_perf_counter_case(args: argparse.Namespace, bench_file: Path, operator_file: Path) -> None:
    if args.case_id is None:
        raise ValueError("--case-id is required for perf-counter-case")
    runtime = _workspace_run_bench_execution()
    cases, resolution = runtime.load_bench_cases(bench_file, operator_file)
    case = runtime.select_bench_case(cases, args.case_id)
    _emit_case_record(runtime.time_bench_case(case, resolution, bench_mode="perf-counter"))


def _workspace_run_bench_execution():
    workdir = str(Path.cwd())
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    return importlib.import_module("run_bench_execution")


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


def _write_payload(result_file: Path, result: ResultPayload, perf_path: Path | None) -> None:
    result_file.write_text(
        json.dumps(
            {
                "result": {
                    "return_code": int(result["return_code"]),
                    "stdout": str(result["stdout"]),
                    "stderr": str(result["stderr"]),
                    "stalled": bool(result["stalled"]),
                    "session_id": result["session_id"],
                },
                "perf_path": None if perf_path is None else str(perf_path.resolve()),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
