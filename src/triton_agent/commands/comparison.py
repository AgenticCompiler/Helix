from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Protocol, TextIO, cast

from triton_agent.remote_execution_env import resolve_remote_execution
from triton_agent.skill_loader import load_operator_eval_script_module


class CompareResultModule(Protocol):
    def compare_result_files(
        self,
        ref_result: Path,
        new_result: Path,
    ) -> int: ...

    def compare_remote_result_files(
        self,
        ref_result: Path,
        new_result: Path,
        remote: str,
        remote_workdir: str | None,
        verbose: bool = False,
        stderr: TextIO | None = None,
    ) -> int: ...


class ComparePerfModule(Protocol):
    def compare_perf_files(
        self,
        baseline_perf: Path,
        compare_perf: Path,
        *,
        skip_latency_errors: bool = False,
        metric_source: str = "auto",
    ) -> int: ...


def _load_compare_result() -> CompareResultModule:
    return cast(CompareResultModule, load_operator_eval_script_module("compare_result"))


def _load_compare_perf() -> ComparePerfModule:
    return cast(ComparePerfModule, load_operator_eval_script_module("perf_artifacts"))


def compare_result_files(ref_result: Path, new_result: Path) -> int:
    return _load_compare_result().compare_result_files(ref_result, new_result)


def compare_remote_result_files(
    ref_result: Path,
    new_result: Path,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return _load_compare_result().compare_remote_result_files(
        ref_result,
        new_result,
        remote,
        remote_workdir,
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
    return _load_compare_perf().compare_perf_files(
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
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        except (RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
    return compare_result_files(ref_result, new_result)


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
