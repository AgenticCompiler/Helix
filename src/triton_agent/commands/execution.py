from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TextIO

from triton_agent.commands.comparison import compare_perf_files
from triton_agent.commands.comparison import compare_remote_result_files, compare_result_files
from triton_agent.eval.runners import (
    AgentResult,
    resolve_bench_mode_default,
    resolve_test_mode_from_metadata,
    run_local_bench,
    run_local_probe_bench,
    run_local_simulator,
    run_local_test,
    run_remote_bench,
    run_remote_probe_bench,
    run_remote_test,
)
from triton_agent.optimize.pt_cleanup import (
    cleanup_run_test_pt_files,
)
from triton_agent.terminal.render import render_result
from triton_agent.remote.env import resolve_remote_execution

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
    accuracy_mode = args.accuracy_mode
    remote_workspace: str | None = None
    try:
        if remote is not None:
            result, archived_result, remote_workspace = run_remote_test(
                test_file,
                operator_file,
                resolved_test_mode,
                remote,
                remote_workdir,
                accuracy_mode=accuracy_mode,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result, archived_result = run_local_test(
                test_file,
                operator_file,
                resolved_test_mode,
                accuracy_mode=accuracy_mode,
                verbose=args.verbose,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    render_result(result, skip_stdout=remote is not None)
    print(f"Return code: {result.return_code}")
    final_code = result.return_code
    if archived_result is not None:
        print(f"Archived result: {archived_result}")
        if ref_result is not None:
            final_code = _compare_run_test_result(
                ref_result,
                archived_result,
                remote,
                remote_workdir,
                accuracy_mode=accuracy_mode,
                verbose=args.verbose,
            )
        cleaned_pt = cleanup_run_test_pt_files((archived_result,))
        if ref_result is None and resolved_test_mode == "differential" and not cleaned_pt:
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


def _compare_run_test_result(
    ref_result: Path,
    archived_result: Path,
    remote: str | None,
    remote_workdir: str | None,
    *,
    accuracy_mode: str,
    verbose: bool,
) -> int:
    if remote is None:
        return compare_result_files(
            ref_result,
            archived_result,
            accuracy_mode=accuracy_mode,
        )
    try:
        return compare_remote_result_files(
            ref_result,
            archived_result,
            remote,
            remote_workdir,
            accuracy_mode=accuracy_mode,
            verbose=verbose,
            stderr=sys.stderr,
        )
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def handle_run_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file, baseline_operator_file = resolve_run_bench_paths(parser, args)
    resolved_bench_mode = args.bench_mode or resolve_bench_mode_default()
    remote, remote_workdir = resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )
    output: str | None = getattr(args, "output", None)
    remote_workspace: str | None = None
    baseline_remote_workspace: str | None = None
    baseline_perf_path = _derived_perf_path(baseline_operator_file) if baseline_operator_file is not None else None
    if baseline_perf_path is not None and baseline_perf_path.exists():
        print(f"Baseline perf file: {baseline_perf_path}")
    try:
        if baseline_operator_file is not None and baseline_perf_path is not None and not baseline_perf_path.exists():
            baseline_result, baseline_generated_perf, baseline_remote_workspace = _run_bench_once(
                bench_file,
                baseline_operator_file,
                resolved_bench_mode,
                remote,
                remote_workdir,
                npu_devices=args.npu_devices,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
                output=None,
            )
            if args.verbose or baseline_result.return_code != 0:
                render_result(baseline_result, skip_stdout=remote is not None and args.verbose)
            if baseline_generated_perf is not None:
                baseline_perf_path = baseline_generated_perf
                print(f"Baseline perf file: {baseline_perf_path}")
            if baseline_result.return_code != 0 or baseline_generated_perf is None:
                if remote is not None and args.keep_remote_workdir and baseline_remote_workspace is not None:
                    print(f"Remote workspace: {baseline_remote_workspace}")
                return baseline_result.return_code if baseline_result.return_code != 0 else 1
            if remote is not None and args.keep_remote_workdir and baseline_remote_workspace is not None:
                print(f"Remote workspace: {baseline_remote_workspace}")

        result, perf_path, remote_workspace = _run_bench_once(
            bench_file,
            operator_file,
            resolved_bench_mode,
            remote,
            remote_workdir,
            npu_devices=args.npu_devices,
            keep_remote_workdir=args.keep_remote_workdir,
            verbose=args.verbose,
            stderr=sys.stderr,
            output=output,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.verbose or result.return_code != 0:
        render_result(result, skip_stdout=remote is not None and args.verbose)
    if remote is not None and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
        if baseline_perf_path is None:
            print(_RUN_BENCH_HINT)
        else:
            return compare_perf_files(
                baseline_perf_path,
                perf_path,
                skip_latency_errors=args.skip_latency_errors,
                metric_source=args.metric_source,
            )
    return result.return_code


def handle_probe_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file, baseline_operator_file = resolve_probe_bench_paths(parser, args)
    resolved_bench_mode = args.bench_mode or resolve_bench_mode_default()
    remote, remote_workdir = resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )
    try:
        if remote is not None:
            result = run_remote_probe_bench(
                bench_file,
                operator_file,
                baseline_operator_file,
                resolved_bench_mode,
                remote,
                remote_workdir,
                metric_source=args.metric_source,
                npu_devices=args.npu_devices,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
        else:
            result = run_local_probe_bench(
                bench_file,
                operator_file,
                baseline_operator_file,
                resolved_bench_mode,
                metric_source=args.metric_source,
                npu_devices=args.npu_devices,
                verbose=args.verbose,
            )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for line in result.default_lines:
        print(line)
    if args.verbose:
        for line in result.verbose_lines:
            print(line)
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    remote_workspace = getattr(result, "remote_workspace", None)
    if remote is not None and remote_workspace is not None and (args.verbose or args.keep_remote_workdir):
        print(f"Remote workspace: {remote_workspace}")
    return result.return_code


def handle_run_simulator(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    bench_file, operator_file, _baseline_operator_file = resolve_run_bench_paths(parser, args)
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
                accuracy_mode=args.accuracy_mode,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
            )
            render_result(ref_run_result, skip_stdout=True)
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
                accuracy_mode=args.accuracy_mode,
                verbose=args.verbose,
            )
            render_result(ref_run_result, skip_stdout=False)
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
) -> tuple[Path, Path, Path | None]:
    bench_file = Path(args.bench_file).expanduser().resolve()
    if not bench_file.exists():
        parser.error(f"Bench file path does not exist: {bench_file}")
    operator_file = Path(args.operator_file).expanduser().resolve()
    if not operator_file.exists():
        parser.error(f"Operator file path does not exist: {operator_file}")
    baseline_operator_file: Path | None = None
    if getattr(args, "baseline_operator_file", None) is not None:
        baseline_operator_file = Path(args.baseline_operator_file).expanduser().resolve()
        if not baseline_operator_file.exists():
            parser.error(f"Baseline operator file path does not exist: {baseline_operator_file}")
    return bench_file, operator_file, baseline_operator_file


def _derived_perf_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"


def _run_bench_once(
    bench_file: Path,
    operator_file: Path,
    resolved_bench_mode: str,
    remote: str | None,
    remote_workdir: str | None,
    *,
    npu_devices: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
    stderr: TextIO | None,
    output: str | None,
) -> tuple[AgentResult, Path | None, str | None]:
    if remote is not None:
        result, perf_path, remote_workspace = run_remote_bench(
            bench_file,
            operator_file,
            resolved_bench_mode,
            remote,
            remote_workdir,
            npu_devices,
            keep_remote_workdir=keep_remote_workdir,
            verbose=verbose,
            stderr=stderr,
            output=output,
        )
        return result, perf_path, remote_workspace
    result, perf_path = run_local_bench(
        bench_file,
        operator_file,
        resolved_bench_mode,
        npu_devices,
        verbose=verbose,
        output=output,
    )
    return result, perf_path, None


def resolve_probe_bench_paths(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    bench_file = Path(args.bench_file).expanduser().resolve()
    if not bench_file.exists():
        parser.error(f"Bench file path does not exist: {bench_file}")
    operator_file = Path(args.operator_file).expanduser().resolve()
    if not operator_file.exists():
        parser.error(f"Operator file path does not exist: {operator_file}")
    baseline_operator_file = Path(args.baseline_operator_file).expanduser().resolve()
    if not baseline_operator_file.exists():
        parser.error(f"Baseline operator file path does not exist: {baseline_operator_file}")
    return bench_file, operator_file, baseline_operator_file
