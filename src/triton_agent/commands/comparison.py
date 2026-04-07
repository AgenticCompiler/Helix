from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.comparison import (
    compare_perf_files,
    compare_remote_result_files,
    compare_result_files,
)


def handle_compare_result(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    oracle_result = Path(args.oracle_result).expanduser().resolve()
    if not oracle_result.exists():
        parser.error(f"Oracle result path does not exist: {oracle_result}")
    new_result = Path(args.new_result).expanduser().resolve()
    if not new_result.exists():
        parser.error(f"New result path does not exist: {new_result}")
    if args.remote:
        try:
            return compare_remote_result_files(
                oracle_result,
                new_result,
                args.compare_level,
                args.remote,
                args.remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return compare_result_files(oracle_result, new_result, args.compare_level)


def handle_compare_perf(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    baseline_perf = Path(args.baseline).expanduser().resolve()
    if not baseline_perf.exists():
        parser.error(f"Baseline perf path does not exist: {baseline_perf}")
    compare_perf = Path(args.compare).expanduser().resolve()
    if not compare_perf.exists():
        parser.error(f"Compare perf path does not exist: {compare_perf}")
    return compare_perf_files(baseline_perf, compare_perf)
