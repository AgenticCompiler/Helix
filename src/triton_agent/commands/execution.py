from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.commands.comparison import compare_result_files
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
    test_file, operator_file, oracle_result = resolve_run_test_paths(parser, args)
    resolved_test_mode = args.test_mode or resolve_test_mode_from_metadata(test_file)
    compare_level = resolve_run_test_compare_level(parser, args, resolved_test_mode, oracle_result)
    force_recompile: bool = getattr(args, "force_recompile", False)
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
                force_recompile=force_recompile,
            )
        else:
            result, archived_result = run_local_test(
                test_file,
                operator_file,
                resolved_test_mode,
                verbose=args.verbose,
                force_recompile=force_recompile,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    render_result(result, show_output=True)
    print(f"Return code: {result.return_code}")
    final_code = result.return_code
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
        if oracle_result is not None:
            final_code = compare_result_files(oracle_result, archived_result, compare_level)
        else:
            print(_RUN_TEST_HINT)
    elif oracle_result is not None:
        print(
            "Differential run-test did not produce an archived result required for automatic comparison.",
            file=sys.stderr,
        )
        final_code = 1
    if args.remote and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    return final_code


def handle_run_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file = resolve_run_bench_paths(parser, args)
    resolved_bench_mode = args.bench_mode or resolve_bench_mode_from_metadata(bench_file)
    force_recompile: bool = getattr(args, "force_recompile", False)
    output: str | None = getattr(args, "output", None)
    remote_workspace: str | None = None
    try:
        if args.remote:
            result, perf_path, remote_workspace = run_remote_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
                args.remote,
                args.remote_workdir,
                args.npu_devices,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
                force_recompile=force_recompile,
                output=output,
            )
        else:
            result, perf_path = run_local_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
                args.npu_devices,
                verbose=args.verbose,
                force_recompile=force_recompile,
                output=output,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose or result.return_code != 0:
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
) -> tuple[Path, Path, Path | None]:
    test_file = Path(args.test_file).expanduser().resolve()
    if not test_file.exists():
        parser.error(f"Test file path does not exist: {test_file}")
    operator_file = Path(args.operator_file).expanduser().resolve()
    if not operator_file.exists():
        parser.error(f"Operator file path does not exist: {operator_file}")
    oracle_result: Path | None = None
    if getattr(args, "oracle_result", None) is not None:
        oracle_result = Path(args.oracle_result).expanduser().resolve()
        if not oracle_result.exists():
            parser.error(f"Oracle result path does not exist: {oracle_result}")
    return test_file, operator_file, oracle_result


def resolve_run_test_compare_level(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    resolved_test_mode: str,
    oracle_result: Path | None,
) -> str:
    if args.compare_level is not None and oracle_result is None:
        parser.error("--compare-level requires --oracle-result")
    if oracle_result is not None and resolved_test_mode != "differential":
        parser.error("--oracle-result is supported only with --test-mode differential")
    return args.compare_level or "balanced"


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
