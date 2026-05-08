from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.execution import (
    resolve_bench_mode_from_metadata,
    resolve_test_mode_from_metadata,
    run_local_bench,
    run_local_test,
    run_remote_bench,
    run_remote_test,
)
from triton_agent.output import render_result

_RUN_BENCH_HINT = "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly."
_RUN_TEST_HINT = "Hint: use `compare-result` to inspect this archived result instead of reading it directly."


def handle_run_test(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    test_file, operator_file = resolve_run_test_paths(parser, args)
    resolved_test_mode = args.test_mode or resolve_test_mode_from_metadata(test_file)
    remote_workspace: str | None = None
    try:
        if args.remote:
            result, archived_result, remote_workspace = run_remote_test(
                test_file,
                operator_file,
                resolved_test_mode,
                args.remote,
                args.remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result, archived_result = run_local_test(
                test_file,
                operator_file,
                resolved_test_mode,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    render_result(result, show_output=True)
    print(f"Return code: {result.return_code}")
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
        print(_RUN_TEST_HINT)
    if args.remote and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    return result.return_code


def handle_run_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file = resolve_run_bench_paths(parser, args)
    resolved_bench_mode = args.bench_mode or resolve_bench_mode_from_metadata(bench_file)
    remote_workspace: str | None = None
    try:
        if args.remote:
            result, perf_path, remote_workspace = run_remote_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
                args.remote,
                args.remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result, perf_path = run_local_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if result.return_code != 0:
        render_result(result, show_output=False)
    if args.remote and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
        print(_RUN_BENCH_HINT)
    return result.return_code


def resolve_run_test_paths(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    test_file = Path(args.test_file).expanduser().resolve()
    if not test_file.exists():
        parser.error(f"Test file path does not exist: {test_file}")
    operator_file = Path(args.operator_file).expanduser().resolve()
    if not operator_file.exists():
        parser.error(f"Operator file path does not exist: {operator_file}")
    return test_file, operator_file


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
