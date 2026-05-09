from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol, TextIO, cast

from triton_agent.skill_loader import load_operator_eval_script_module


class CompareResultModule(Protocol):
    def compare_result_files(
        self,
        oracle_result: Path,
        new_result: Path,
        compare_level: str,
    ) -> int: ...

    def compare_remote_result_files(
        self,
        oracle_result: Path,
        new_result: Path,
        compare_level: str,
        remote: str,
        remote_workdir: str | None,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> int: ...


class ComparePerfModule(Protocol):
    def compare_perf_files(self, baseline_perf: Path, compare_perf: Path) -> int: ...


def _load_compare_result() -> CompareResultModule:
    return cast(CompareResultModule, load_operator_eval_script_module("compare_result"))


def _load_compare_perf() -> ComparePerfModule:
    return cast(ComparePerfModule, load_operator_eval_script_module("bench_runner"))


def compare_result_files(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    return _load_compare_result().compare_result_files(oracle_result, new_result, compare_level)


def compare_remote_result_files(
    oracle_result: Path,
    new_result: Path,
    compare_level: str,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return _load_compare_result().compare_remote_result_files(
        oracle_result,
        new_result,
        compare_level,
        remote,
        remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    return _load_compare_perf().compare_perf_files(baseline_perf, compare_perf)


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
