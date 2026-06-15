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
    test_file, operator_file, ref_result, ref_operator_file = resolve_run_test_paths(parser, args)
    resolved_test_mode = args.test_mode or resolve_test_mode_from_metadata(test_file)
    remote, remote_workdir = resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )
    ref_result = resolve_run_test_comparison_inputs(
        parser,
        args,
        resolved_test_mode,
        ref_result,
        ref_operator_file,
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
        if ref_result is not None:
            final_code = compare_result_files(ref_result, archived_result)
        else:
            print(_RUN_TEST_HINT)
    elif ref_result is not None:
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
    ref_result: Path | None = None
    if getattr(args, "ref_result", None) is not None:
        ref_result = Path(args.ref_result).expanduser().resolve()
        if not ref_result.exists():
            parser.error(f"Reference result path does not exist: {ref_result}")
    ref_operator_file: Path | None = None
    if getattr(args, "ref_operator_file", None) is not None:
        ref_operator_file = Path(args.ref_operator_file).expanduser().resolve()
        if not ref_operator_file.exists():
            parser.error(f"Reference operator file path does not exist: {ref_operator_file}")
    return test_file, operator_file, ref_result, ref_operator_file


def _derived_result_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_result.pt"


def resolve_run_test_comparison_inputs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    resolved_test_mode: str,
    ref_result: Path | None,
    ref_operator_file: Path | None,
    test_file: Path,
    *,
    remote: str | None,
    remote_workdir: str | None,
) -> Path | None:
    if ref_result is not None and ref_operator_file is not None:
        parser.error("run-test differential mode accepts at most one of --ref-result or --ref-operator-file")
    if ref_result is not None and resolved_test_mode != "differential":
        parser.error("--ref-result is supported only with --test-mode differential")
    if ref_operator_file is not None and resolved_test_mode != "differential":
        parser.error("--ref-operator-file is supported only with --test-mode differential")
    if ref_operator_file is None:
        return ref_result
    derived_ref_result = _derived_result_path(ref_operator_file)
    if derived_ref_result.exists():
        return derived_ref_result

    try:
        if remote is not None:
            ref_run_result, archived_result, remote_workspace = run_remote_test(
                test_file,
                ref_operator_file,
                resolved_test_mode,
                remote,
                remote_workdir,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
            render_result(ref_run_result, show_output=True)
            print(f"Return code: {ref_run_result.return_code}")
            if archived_result is not None:
                print(f"Archived result: {archived_result}")
            if args.keep_remote_workdir:
                print(f"Remote workspace: {remote_workspace}")
        else:
            ref_run_result, archived_result = run_local_test(
                test_file,
                ref_operator_file,
                resolved_test_mode,
                verbose=args.verbose,
            )
            render_result(ref_run_result, show_output=True)
            print(f"Return code: {ref_run_result.return_code}")
            if archived_result is not None:
                print(f"Archived result: {archived_result}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    if ref_run_result.return_code != 0 or archived_result is None:
        raise SystemExit(1)
    return derived_ref_result


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
