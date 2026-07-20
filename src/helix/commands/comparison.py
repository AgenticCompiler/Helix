from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from helix.remote.env import resolve_remote_execution
from helix.skill_bridges import run_eval_comparison


def compare_result_files(
    ref_result: Path,
    new_result: Path,
    *,
    accuracy_mode: str | None = None,
) -> int:
    return run_eval_comparison.compare_result_files(
        ref_result,
        new_result,
        accuracy_mode=accuracy_mode,
    )


def compare_result_payload_objects(
    ref_payload: object,
    new_payload: object,
    *,
    accuracy_mode: str | None = None,
) -> int:
    return run_eval_comparison.compare_result_payload_objects(
        ref_payload,
        new_payload,
        accuracy_mode=accuracy_mode,
    )


def load_case_result_payload(
    ref_result: Path,
    case_id: str,
) -> object:
    return run_eval_comparison.load_case_result_payload(ref_result, case_id)


def find_case_result_payload(
    ref_result: Path,
    case_id: str,
) -> object | None:
    return run_eval_comparison.find_case_result_payload(ref_result, case_id)


def compare_remote_result_files(
    ref_result: Path,
    new_result: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    accuracy_mode: str | None = None,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return run_eval_comparison.compare_remote_result_files(
        ref_result,
        new_result,
        remote,
        remote_workdir,
        accuracy_mode=accuracy_mode,
        verbose=verbose,
        stderr=stderr,
    )


def compare_perf_files(
    baseline_perf: Path,
    compare_perf: Path,
    *,
    skip_latency_errors: bool = False,
    metric_source: str = "auto",
) -> int:
    return run_eval_comparison.compare_perf_files(
        baseline_perf,
        compare_perf,
        skip_latency_errors=skip_latency_errors,
        metric_source=metric_source,
    )


def handle_compare_result(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    ref_result = Path(args.ref_result).expanduser().resolve()
    if not ref_result.exists():
        parser.error(f"Reference result path does not exist: {ref_result}")
    new_result = Path(args.new_result).expanduser().resolve()
    if not new_result.exists():
        parser.error(f"New result path does not exist: {new_result}")
    remote, remote_workdir = resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )
    if remote is not None:
        try:
            return compare_remote_result_files(
                ref_result,
                new_result,
                remote,
                remote_workdir,
                accuracy_mode=args.accuracy_mode,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return compare_result_files(
        ref_result,
        new_result,
        accuracy_mode=args.accuracy_mode,
    )


def handle_compare_perf(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    baseline_perf = Path(args.baseline).expanduser().resolve()
    if not baseline_perf.exists():
        parser.error(f"Baseline perf path does not exist: {baseline_perf}")
    compare_perf = Path(args.compare).expanduser().resolve()
    if not compare_perf.exists():
        parser.error(f"Compare perf path does not exist: {compare_perf}")
    return compare_perf_files(
        baseline_perf,
        compare_perf,
        skip_latency_errors=args.skip_latency_errors,
        metric_source=args.metric_source,
    )
