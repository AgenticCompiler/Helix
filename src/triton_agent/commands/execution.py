from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.commands.comparison import compare_result_files
from triton_agent.execution import (
    resolve_bench_mode_from_metadata,
    resolve_test_mode_from_metadata,
    run_local_bench,
    run_local_simulator,
    run_local_test,
    run_remote_bench,
    run_remote_test,
)
from triton_agent.output import render_result
from triton_agent.remote_execution_env import resolve_remote_execution

_RUN_BENCH_HINT = "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly."
_RUN_TEST_HINT = "Hint: use `compare-result` to inspect this archived result instead of reading it directly."


def handle_run_test(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    test_file, operator_file, baseline_result, baseline_operator_file = resolve_run_test_paths(parser, args)
    resolved_test_mode = args.test_mode or resolve_test_mode_from_metadata(test_file)
    remote, remote_workdir = resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )
    compare_level, baseline_result = resolve_run_test_comparison_inputs(
        parser,
        args,
        resolved_test_mode,
        baseline_result,
        baseline_operator_file,
        test_file,
        remote=remote,
        remote_workdir=remote_workdir,
    )
    remote_workspace: str | None = None
    try:
        if remote is not None:
            result, archived_result, remote_workspace = run_remote_test(
                test_file,
                operator_file,
                resolved_test_mode,
                remote,
                remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result, archived_result = run_local_test(
                test_file,
                operator_file,
                resolved_test_mode,
                verbose=args.verbose,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    render_result(result, show_output=True)
    print(f"Return code: {result.return_code}")
    final_code = result.return_code
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
        if baseline_result is not None:
            final_code = compare_result_files(baseline_result, archived_result, compare_level)
        else:
            print(_RUN_TEST_HINT)
    elif baseline_result is not None:
        print(
            "Differential run-test did not produce an archived result required for automatic comparison.",
            file=sys.stderr,
        )
        final_code = 1
    if remote is not None and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    return final_code


def handle_run_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file = resolve_run_bench_paths(parser, args)
    resolved_bench_mode = args.bench_mode or resolve_bench_mode_from_metadata(bench_file)
    remote, remote_workdir = resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )
    output: str | None = getattr(args, "output", None)
    remote_workspace: str | None = None
    try:
        if remote is not None:
            result, perf_path, remote_workspace = run_remote_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
                remote,
                remote_workdir,
                args.npu_devices,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
                output=output,
            )
        else:
            result, perf_path = run_local_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
                args.npu_devices,
                verbose=args.verbose,
                output=output,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose or result.return_code != 0:
        render_result(result, show_output=False)
    if remote is not None and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
        print(_RUN_BENCH_HINT)
    return result.return_code


def handle_run_simulator(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file = resolve_run_bench_paths(parser, args)
    try:
        result = run_local_simulator(
            bench_file,
            operator_file,
            case_id=getattr(args, "case_id", None),
            kernel_name=getattr(args, "kernel_name", None),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return result.return_code


def resolve_run_test_paths(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path | None]:
    test_file = Path(args.test_file).expanduser().resolve()
    if not test_file.exists():
        parser.error(f"Test file path does not exist: {test_file}")
    operator_file = Path(args.operator_file).expanduser().resolve()
    if not operator_file.exists():
        parser.error(f"Operator file path does not exist: {operator_file}")
    baseline_result: Path | None = None
    if getattr(args, "baseline_result", None) is not None:
        baseline_result = Path(args.baseline_result).expanduser().resolve()
        if not baseline_result.exists():
            parser.error(f"Baseline result path does not exist: {baseline_result}")
    baseline_operator_file: Path | None = None
    if getattr(args, "baseline_operator_file", None) is not None:
        baseline_operator_file = Path(args.baseline_operator_file).expanduser().resolve()
        if not baseline_operator_file.exists():
            parser.error(f"Baseline operator file path does not exist: {baseline_operator_file}")
    return test_file, operator_file, baseline_result, baseline_operator_file


def _derived_result_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def resolve_run_test_comparison_inputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    resolved_test_mode: str,
    baseline_result: Path | None,
    baseline_operator_file: Path | None,
    test_file: Path,
    *,
    remote: str | None,
    remote_workdir: str | None,
) -> tuple[str, Path | None]:
    if baseline_result is not None and baseline_operator_file is not None:
        parser.error("run-test differential mode accepts at most one of --baseline-result or --baseline-operator-file")
    if args.compare_level is not None and baseline_result is None and baseline_operator_file is None:
        parser.error("--compare-level requires --baseline-result or --baseline-operator-file")
    if baseline_result is not None and resolved_test_mode != "differential":
        parser.error("--baseline-result is supported only with --test-mode differential")
    if baseline_operator_file is not None and resolved_test_mode != "differential":
        parser.error("--baseline-operator-file is supported only with --test-mode differential")
    compare_level = args.compare_level or "balanced"
    if baseline_operator_file is None:
        return compare_level, baseline_result
    derived_baseline_result = _derived_result_path(baseline_operator_file)
    if derived_baseline_result.exists():
        return compare_level, derived_baseline_result

    try:
        if remote is not None:
            baseline_run_result, archived_result, remote_workspace = run_remote_test(
                test_file,
                baseline_operator_file,
                resolved_test_mode,
                remote,
                remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
            render_result(baseline_run_result, show_output=True)
            print(f"Return code: {baseline_run_result.return_code}")
            if archived_result is not None:
                print(f"Archived result: {archived_result}")
            if args.keep_remote_workdir:
                print(f"Remote workspace: {remote_workspace}")
        else:
            baseline_run_result, archived_result = run_local_test(
                test_file,
                baseline_operator_file,
                resolved_test_mode,
                verbose=args.verbose,
            )
            render_result(baseline_run_result, show_output=True)
            print(f"Return code: {baseline_run_result.return_code}")
            if archived_result is not None:
                print(f"Archived result: {archived_result}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    if baseline_run_result.return_code != 0 or archived_result is None:
        raise SystemExit(1)
    return compare_level, derived_baseline_result


def resolve_run_bench_paths(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    bench_file = Path(args.bench_file).expanduser().resolve()
    if not bench_file.exists():
        parser.error(f"Bench file path does not exist: {bench_file}")
    operator_file = Path(args.operator_file).expanduser().resolve()
    if not operator_file.exists():
        parser.error(f"Operator file path does not exist: {operator_file}")
    return bench_file, operator_file
