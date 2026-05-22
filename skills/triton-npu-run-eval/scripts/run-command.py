from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterator, Protocol, cast

from result_payload import ResultPayload

SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_BENCH_HINT = "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly."
_RUN_TEST_HINT = "Hint: use `compare-result` to inspect this archived result instead of reading it directly."

class ParseMetadataFn(Protocol):
    def __call__(self, path: Path) -> dict[str, str]: ...


class RunLocalTestFn(Protocol):
    def __call__(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        *,
        verbose: bool = False,
    ) -> tuple[ResultPayload, Path | None]: ...


class RunRemoteTestFn(Protocol):
    def __call__(
        self,
        test_file: Path,
        operator_file: Path,
        test_mode: str,
        remote: str,
        remote_workdir: str | None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: object | None = None,
    ) -> tuple[ResultPayload, Path | None, str]: ...


class RunLocalBenchFn(Protocol):
    def __call__(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        npu_devices: str | None = None,
    ) -> tuple[ResultPayload, Path | None]: ...


class RunRemoteBenchFn(Protocol):
    def __call__(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        remote: str,
        remote_workdir: str | None,
        npu_devices: str | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: object | None = None,
    ) -> tuple[ResultPayload, Path | None, str]: ...


class CompareResultFn(Protocol):
    def __call__(self, oracle_result: Path, new_result: Path, compare_level: str) -> int: ...


class CompareRemoteResultFn(Protocol):
    def __call__(
        self,
        oracle_result: Path,
        new_result: Path,
        compare_level: str,
        remote: str,
        remote_workdir: str | None,
        verbose: bool = False,
        stderr: object | None = None,
    ) -> int: ...


class ComparePerfFn(Protocol):
    def __call__(
        self,
        baseline_perf: Path,
        compare_perf: Path,
        *,
        skip_latency_errors: bool = False,
        metric_source: str = "auto",
    ) -> int: ...


class RunLocalProfileBenchFn(Protocol):
    def __call__(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        bench_case: int | None = None,
        case_id: str | None = None,
        kernel_name: str | None = None,
    ) -> tuple[ResultPayload, Path | None]: ...


class RunRemoteProfileBenchFn(Protocol):
    def __call__(
        self,
        bench_file: Path,
        operator_file: Path,
        bench_mode: str,
        remote: str,
        remote_workdir: str | None,
        bench_case: int | None = None,
        case_id: str | None = None,
        kernel_name: str | None = None,
        keep_remote_workdir: bool = False,
        verbose: bool = False,
        stderr: object | None = None,
    ) -> tuple[ResultPayload, Path | None, str]: ...


class ProfileSummaryModule(Protocol):
    def build_report(
        self,
        profile_path: str | Path,
        target_op: str | None = None,
        top_count: int = 5,
        output_format: str = "markdown",
    ) -> str: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_test = subparsers.add_parser("run-test")
    run_test.add_argument("--test-file", required=True)
    run_test.add_argument("--operator-file", required=True)
    run_test.add_argument("--oracle-result")
    run_test.add_argument("--compare-level", choices=["strict", "balanced", "relaxed"])
    run_test.add_argument("--remote")
    run_test.add_argument("--remote-workdir")
    run_test.add_argument("--keep-remote-workdir", action="store_true")
    run_test.add_argument("--verbose", action="store_true")
    run_test.add_argument("--test-mode", choices=["standalone", "differential"])

    run_bench = subparsers.add_parser("run-bench")
    run_bench.add_argument("--bench-file", required=True)
    run_bench.add_argument("--operator-file", required=True)
    run_bench.add_argument("--remote")
    run_bench.add_argument("--remote-workdir")
    run_bench.add_argument("--keep-remote-workdir", action="store_true")
    run_bench.add_argument("--verbose", action="store_true")
    run_bench.add_argument("--bench-mode", choices=["standalone", "msprof"])
    run_bench.add_argument("--npu-devices")

    profile_bench = subparsers.add_parser("profile-bench")
    profile_bench.add_argument("--bench-file", required=True)
    profile_bench.add_argument("--operator-file", required=True)
    profile_bench.add_argument("--bench-mode", choices=["standalone", "msprof"])
    profile_bench.add_argument("--case-id")
    profile_bench.add_argument("--bench", type=int)
    profile_bench.add_argument("--kernel-name", help=argparse.SUPPRESS)
    profile_bench.add_argument("--target-op")
    profile_bench.add_argument("--remote")
    profile_bench.add_argument("--remote-workdir")
    profile_bench.add_argument("--keep-remote-workdir", action="store_true")
    profile_bench.add_argument("--verbose", action="store_true")

    profile_report = subparsers.add_parser("profile-report")
    profile_report.add_argument("--profile-dir", required=True)
    profile_report.add_argument("--target-op")
    profile_report.add_argument("--format", choices=["markdown", "json"], default="markdown")
    profile_report.add_argument("--top", type=int, default=5)

    compare_result = subparsers.add_parser("compare-result")
    compare_result.add_argument("--oracle-result", required=True)
    compare_result.add_argument("--new-result", required=True)
    compare_result.add_argument(
        "--compare-level",
        default="balanced",
        choices=["strict", "balanced", "relaxed"],
    )
    compare_result.add_argument("--remote")
    compare_result.add_argument("--remote-workdir")
    compare_result.add_argument("--verbose", action="store_true")

    compare_perf = subparsers.add_parser("compare-perf")
    compare_perf.add_argument("--baseline", required=True)
    compare_perf.add_argument("--compare", required=True)
    compare_perf.add_argument("--skip-latency-errors", action="store_true")
    compare_perf.add_argument(
        "--metric-source",
        default="auto",
        choices=["auto", "kernel", "total-op", "all"],
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "compare-result":
        compare_result_files, compare_remote_result_files = _load_compare_result_functions()
        oracle_result = _resolve_existing_path(parser, args.oracle_result, "Oracle result")
        new_result = _resolve_existing_path(parser, args.new_result, "New result")
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

    if args.command == "compare-perf":
        compare_perf_files = _load_compare_perf_function()
        baseline_perf = _resolve_existing_path(parser, args.baseline, "Baseline perf")
        compare_perf = _resolve_existing_path(parser, args.compare, "Compare perf")
        return compare_perf_files(
            baseline_perf,
            compare_perf,
            skip_latency_errors=args.skip_latency_errors,
            metric_source=args.metric_source,
        )

    if args.command == "run-test":
        _parse_test_metadata, run_local_test, run_remote_test = _load_test_functions()
        test_file = _resolve_existing_path(parser, args.test_file, "Test file")
        operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
        oracle_result = (
            _resolve_existing_path(parser, args.oracle_result, "Oracle result")
            if args.oracle_result is not None
            else None
        )
        resolved_test_mode = args.test_mode or _resolve_test_mode_from_metadata(test_file)
        compare_level = _resolve_run_test_compare_level(
            parser,
            args,
            resolved_test_mode,
            oracle_result,
        )
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
                    verbose=args.verbose,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _render_result(result, show_output=True)
        print(f"Return code: {result['return_code']}")
        final_code = int(result["return_code"])
        if archived_result is not None:
            print(f"Archived result: {archived_result}")
            if oracle_result is not None:
                compare_result_files = _load_compare_result_functions()[0]
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

    if args.command == "profile-bench":
        run_local_profile_bench, run_remote_profile_bench = _load_profile_functions()
        bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
        operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
        resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(bench_file)
        remote_workspace: str | None = None
        try:
            if args.remote:
                result, profile_dir, remote_workspace = run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    resolved_bench_mode,
                    args.remote,
                    args.remote_workdir,
                    bench_case=args.bench,
                    case_id=args.case_id,
                    kernel_name=args.kernel_name,
                    keep_remote_workdir=args.keep_remote_workdir,
                    verbose=args.verbose,
                    stderr=sys.stderr,
                )
            else:
                result, profile_dir = run_local_profile_bench(
                    bench_file,
                    operator_file,
                    resolved_bench_mode,
                    bench_case=args.bench,
                    case_id=args.case_id,
                    kernel_name=args.kernel_name,
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _render_result(result, show_output=True)
        print(f"Return code: {result['return_code']}")
        if profile_dir is not None:
            print(f"Profile directory: {profile_dir}")
            print(_build_profile_report(profile_dir, args.target_op))
        if args.remote and args.keep_remote_workdir and remote_workspace is not None:
            print(f"Remote workspace: {remote_workspace}")
        return int(result["return_code"])

    if args.command == "profile-report":
        report = _build_profile_report(
            _resolve_existing_path(parser, args.profile_dir, "Profile directory"),
            args.target_op,
            top_count=args.top,
            output_format=args.format,
        )
        print(report)
        return 0

    bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    _parse_bench_metadata, run_local_bench, run_remote_bench = _load_bench_functions()
    resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(bench_file)
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
            )
        else:
            result, perf_path = run_local_bench(
                bench_file,
                operator_file,
                resolved_bench_mode,
                args.npu_devices,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if result["return_code"] != 0:
        _render_result(result, show_output=False)
    if args.remote and args.keep_remote_workdir and remote_workspace is not None:
        print(f"Remote workspace: {remote_workspace}")
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
        print(_RUN_BENCH_HINT)
    return int(result["return_code"])


def _resolve_existing_path(
    parser: argparse.ArgumentParser,
    raw_path: str,
    label: str,
) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        parser.error(f"{label} path does not exist: {path}")
    return path


def _resolve_test_mode_from_metadata(test_file: Path) -> str:
    parse_test_metadata = _load_test_functions()[0]
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        raise ValueError(f"Test metadata is missing required 'test-mode' entry: {test_file}")
    return mode


def _resolve_run_test_compare_level(
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


def _resolve_bench_mode_from_metadata(bench_file: Path) -> str:
    parse_bench_metadata = _load_bench_functions()[0]
    metadata = parse_bench_metadata(bench_file)
    mode = metadata.get("bench-mode")
    if mode not in {"standalone", "msprof"}:
        raise ValueError(f"Benchmark metadata is missing required 'bench-mode' entry: {bench_file}")
    return mode


def _render_result(result: ResultPayload, show_output: bool) -> None:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if stdout and not show_output:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def _load_test_functions() -> tuple[ParseMetadataFn, RunLocalTestFn, RunRemoteTestFn]:
    with _script_dir_on_path():
        module = importlib.import_module("test_runner")

    return (
        cast(ParseMetadataFn, getattr(module, "parse_test_metadata")),
        cast(RunLocalTestFn, getattr(module, "run_local_test")),
        cast(RunRemoteTestFn, getattr(module, "run_remote_test")),
    )


def _load_bench_functions() -> tuple[ParseMetadataFn, RunLocalBenchFn, RunRemoteBenchFn]:
    with _script_dir_on_path():
        from bench_contract import parse_bench_metadata
        from bench_runner import run_local_bench, run_remote_bench

    return (
        cast(ParseMetadataFn, parse_bench_metadata),
        cast(RunLocalBenchFn, run_local_bench),
        cast(RunRemoteBenchFn, run_remote_bench),
    )


def _load_compare_result_functions() -> tuple[CompareResultFn, CompareRemoteResultFn]:
    with _script_dir_on_path():
        module = importlib.import_module("compare_result")

    return (
        cast(CompareResultFn, getattr(module, "compare_result_files")),
        cast(CompareRemoteResultFn, getattr(module, "compare_remote_result_files")),
    )


def _load_compare_perf_function() -> ComparePerfFn:
    with _script_dir_on_path():
        from perf_artifacts import compare_perf_files

    return cast(ComparePerfFn, compare_perf_files)


def _load_profile_functions() -> tuple[RunLocalProfileBenchFn, RunRemoteProfileBenchFn]:
    with _script_dir_on_path():
        module = importlib.import_module("profile_runner")

    return (
        cast(RunLocalProfileBenchFn, getattr(module, "run_local_profile_bench")),
        cast(RunRemoteProfileBenchFn, getattr(module, "run_remote_profile_bench")),
    )


def _build_profile_report(
    profile_dir: Path,
    target_op: str | None = None,
    top_count: int = 5,
    output_format: str = "markdown",
) -> str:
    script = SCRIPT_DIR.parents[1] / "triton-npu-profile-operator" / "scripts" / "reporter.py"
    with _temporary_sys_path_entry(str(script.parent)):
        spec = importlib.util.spec_from_file_location("profile_reporter_runtime", script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load profile reporter script: {script}")
        loaded_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded_module)
    module = cast(ProfileSummaryModule, loaded_module)
    return module.build_report(
        profile_dir,
        target_op=target_op,
        top_count=top_count,
        output_format=output_format,
    )


@contextlib.contextmanager
def _temporary_sys_path_entry(path: str) -> Iterator[None]:
    added = False
    if path not in sys.path:
        sys.path.insert(0, path)
        added = True
    try:
        yield
    finally:
        if added:
            sys.path.remove(path)


@contextlib.contextmanager
def _script_dir_on_path() -> Iterator[None]:
    with _temporary_sys_path_entry(str(SCRIPT_DIR)):
        yield


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
