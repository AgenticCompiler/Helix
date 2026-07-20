"""CLI orchestration for the canonical run-bench command."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TextIO

from result_payload import ResultPayload


RunLocalBench = Callable[..., tuple[ResultPayload, Optional[Path]]]
RunRemoteBench = Callable[..., tuple[ResultPayload, Optional[Path], str]]
LoadBenchFunctions = Callable[[], tuple[object, RunLocalBench, RunRemoteBench]]
ResolvePath = Callable[[argparse.ArgumentParser, str, str], Path]
ResolveOptionalPath = Callable[[argparse.ArgumentParser, Optional[str], str], Optional[Path]]
DerivedPerfPath = Callable[[Path], Path]
RenderResult = Callable[[ResultPayload, bool], None]
ComparePerf = Callable[..., int]
LoadComparePerf = Callable[[], ComparePerf]
TimingContext = object
ActiveTimingContext = Callable[[Path, Path], TimingContext]
AppendTimingEvent = Callable[..., None]


@dataclass(frozen=True)
class RunBenchDependencies:
    load_bench_functions: LoadBenchFunctions
    resolve_existing_path: ResolvePath
    resolve_optional_existing_path: ResolveOptionalPath
    derived_perf_path: DerivedPerfPath
    render_result: RenderResult
    load_compare_perf: LoadComparePerf
    active_optimize_round_context: ActiveTimingContext
    append_optimize_timing_event: AppendTimingEvent
    hint: str


def handle_run_bench_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    remote: str | None,
    remote_workdir: str | None,
    dependencies: RunBenchDependencies,
) -> int:
    bench_file = dependencies.resolve_existing_path(parser, args.bench_file, "Bench file")
    operator_file = dependencies.resolve_existing_path(parser, args.operator_file, "Operator file")
    timing_context = dependencies.active_optimize_round_context(bench_file, operator_file)
    baseline_operator_file = dependencies.resolve_optional_existing_path(
        parser,
        getattr(args, "baseline_operator_file", None),
        "Baseline operator file",
    )
    _parse_metadata, run_local_bench, run_remote_bench = dependencies.load_bench_functions()
    bench_mode = args.bench_mode or "torch-npu-profiler"
    baseline_perf_path = (
        dependencies.derived_perf_path(baseline_operator_file)
        if baseline_operator_file is not None
        else None
    )
    if baseline_perf_path is not None and baseline_perf_path.exists():
        print(f"Baseline perf file: {baseline_perf_path}")
    dependencies.append_optimize_timing_event(
        timing_context,
        event="run_bench_start",
        command=args.command,
        bench_file=bench_file,
        operator_file=operator_file,
    )
    try:
        if baseline_operator_file is not None and baseline_perf_path is not None and not baseline_perf_path.exists():
            baseline_result, generated_perf, baseline_workspace = _run_bench_once(
                run_local_bench,
                run_remote_bench,
                bench_file,
                baseline_operator_file,
                bench_mode,
                remote,
                remote_workdir,
                npu_devices=args.npu_devices,
                keep_remote_workdir=args.keep_remote_workdir,
                verbose=args.verbose,
                stderr=sys.stderr,
                output=None,
            )
            if int(baseline_result["return_code"]) != 0:
                dependencies.render_result(baseline_result, remote is not None and args.verbose)
            if generated_perf is not None:
                baseline_perf_path = generated_perf
                print(f"Baseline perf file: {baseline_perf_path}")
            if int(baseline_result["return_code"]) != 0 or generated_perf is None:
                if remote is not None and args.keep_remote_workdir and baseline_workspace is not None:
                    print(f"Remote workspace: {baseline_workspace}")
                return _finish(
                    dependencies,
                    timing_context,
                    args.command,
                    bench_file,
                    operator_file,
                    int(baseline_result["return_code"]) or 1,
                )
            if remote is not None and args.keep_remote_workdir and baseline_workspace is not None:
                print(f"Remote workspace: {baseline_workspace}")

        result, perf_path, workspace = _run_bench_once(
            run_local_bench,
            run_remote_bench,
            bench_file,
            operator_file,
            bench_mode,
            remote,
            remote_workdir,
            npu_devices=args.npu_devices,
            keep_remote_workdir=args.keep_remote_workdir,
            verbose=args.verbose,
            stderr=sys.stderr,
            output=args.output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return _finish(dependencies, timing_context, args.command, bench_file, operator_file, 1)

    if int(result["return_code"]) != 0:
        dependencies.render_result(result, remote is not None and args.verbose)
    if remote is not None and args.keep_remote_workdir and workspace is not None:
        print(f"Remote workspace: {workspace}")
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
        if baseline_perf_path is None:
            print(dependencies.hint)
        else:
            return _finish(
                dependencies,
                timing_context,
                args.command,
                bench_file,
                operator_file,
                dependencies.load_compare_perf()(
                    baseline_perf_path,
                    perf_path,
                    skip_latency_errors=args.skip_latency_errors,
                    metric_source=args.metric_source,
                ),
            )
    return _finish(
        dependencies,
        timing_context,
        args.command,
        bench_file,
        operator_file,
        int(result["return_code"]),
    )


def _run_bench_once(
    run_local_bench: RunLocalBench,
    run_remote_bench: RunRemoteBench,
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str | None,
    remote_workdir: str | None,
    *,
    npu_devices: str | None,
    keep_remote_workdir: bool,
    verbose: bool,
    stderr: TextIO | None,
    output: str | None,
) -> tuple[ResultPayload, Path | None, str | None]:
    if remote is not None:
        result, perf_path, workspace = run_remote_bench(
            bench_file,
            operator_file,
            bench_mode,
            remote,
            remote_workdir,
            npu_devices,
            keep_remote_workdir=keep_remote_workdir,
            verbose=verbose,
            stderr=stderr,
            output=output,
        )
        return result, perf_path, workspace
    result, perf_path = run_local_bench(
        bench_file,
        operator_file,
        bench_mode,
        npu_devices,
        output=output,
    )
    return result, perf_path, None


def _finish(
    dependencies: RunBenchDependencies,
    timing_context: TimingContext,
    command: str,
    bench_file: Path,
    operator_file: Path,
    return_code: int,
) -> int:
    dependencies.append_optimize_timing_event(
        timing_context,
        event="run_bench_end",
        command=command,
        return_code=return_code,
        bench_file=bench_file,
        operator_file=operator_file,
    )
    return return_code
