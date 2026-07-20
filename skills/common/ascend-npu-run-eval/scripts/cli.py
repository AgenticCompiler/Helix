from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterator, Protocol, cast

from execution_lifecycle import (
    active_optimize_round_context as _active_optimize_round_context,
    append_optimize_timing_event as _append_optimize_timing_event,
    guard_operator_execution_env as _guard_operator_execution_env,
)
from result_payload import ResultPayload

SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_BENCH_HINT = "Hint: use `compare-perf` to inspect this perf artifact instead of reading it directly."


def _profile_bench_hint(profile_dir: Path) -> str:
    return (
        "Hint: rerun the bundled `profile-report` helper for this "
        f"`--profile-dir {profile_dir}` if you need the summary again; "
        "if that is not enough, inspect the raw files in this profile directory directly."
    )


class ParseMetadataFn(Protocol):
    def __call__(self, path: Path) -> dict[str, str]: ...


class ResolveRemoteExecutionFn(Protocol):
    def __call__(
        self,
        explicit_remote: str | None,
        explicit_remote_workdir: str | None,
    ) -> tuple[str | None, str | None]: ...


class ComparePerfFn(Protocol):
    def __call__(
        self,
        baseline_perf: Path,
        compare_perf: Path,
        *,
        skip_latency_errors: bool = False,
        metric_source: str = "auto",
    ) -> int: ...


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

    run_test_baseline = subparsers.add_parser("run-test-baseline")
    _add_run_test_arguments(run_test_baseline)

    run_test_convert = subparsers.add_parser("run-test-convert")
    _add_run_test_arguments(run_test_convert)

    run_test_optimize = subparsers.add_parser("run-test-optimize")
    _add_run_test_arguments(run_test_optimize)

    run_bench = subparsers.add_parser("run-bench")
    run_bench.add_argument("--bench-file", required=True)
    run_bench.add_argument("--operator-file", required=True)
    run_bench.add_argument("--baseline-operator-file")
    run_bench.add_argument("--skip-latency-errors", "--skip-error", dest="skip_latency_errors", action="store_true")
    run_bench.add_argument(
        "-m",
        "--metric-source",
        default="auto",
        choices=["auto", "kernel", "total-op", "all"],
    )
    run_bench.add_argument("--output")
    run_bench.add_argument("--remote")
    run_bench.add_argument("--remote-workdir")
    run_bench.add_argument("--keep-remote-workdir", action="store_true")
    run_bench.add_argument("--verbose", action="store_true")
    run_bench.add_argument("--bench-mode", choices=["torch-npu-profiler", "msprof", "perf-counter"])
    run_bench.add_argument("--npu-devices")

    profile_bench = subparsers.add_parser("profile-bench")
    profile_bench.add_argument("--bench-file", required=True)
    profile_bench.add_argument("--operator-file", required=True)
    profile_bench.add_argument("--case-id")
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

    compare_perf = subparsers.add_parser("compare-perf")
    compare_perf.add_argument("--baseline", required=True)
    compare_perf.add_argument("--compare", required=True)
    compare_perf.add_argument("--skip-latency-errors", "--skip-error", dest="skip_latency_errors", action="store_true")
    compare_perf.add_argument(
        "-m",
        "--metric-source",
        default="auto",
        choices=["auto", "kernel", "total-op", "all"],
    )

    return parser


def _add_run_test_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--operator-file", required=True)
    parser.add_argument("--ref-result", "--baseline-result", dest="ref_result")
    parser.add_argument("--ref-operator-file", "--baseline-operator-file", dest="ref_operator_file")
    parser.add_argument("--case-id")
    parser.add_argument("--remote")
    parser.add_argument("--remote-workdir")
    parser.add_argument("--keep-remote-workdir", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--test-mode", choices=["standalone", "differential"])
    parser.add_argument(
        "--accuracy-mode",
        choices=["npu-contract", "dtype-close"],
        # Leave the policy unset so managed MCP invocations can inherit the
        # HELIX_RUN_TEST_* controls supplied by their parent process.
        default=None,
    )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    with _guard_operator_execution_env(args.command):
        return _dispatch_command(parser, args)


def _dispatch_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    remote, remote_workdir = _resolve_remote_execution(args)

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

    if args.command in {"run-test-baseline", "run-test-convert", "run-test-optimize"}:
        from run_test_command import RunTestDependencies, handle_run_test_command

        parse_test_metadata, run_local_test, run_remote_test = _load_test_functions()
        run_local_payload, run_remote_payload = _load_test_payload_functions()
        load_case_payload, find_case_payload, compare_payloads = _load_compare_result_payload_functions()
        compare_result, compare_remote_result = _load_compare_result_functions()
        # Keep CLI dependencies late-bound to the active skill-script environment;
        # run_test_command uses direct imports for standalone calls and test injection.
        return handle_run_test_command(
            parser,
            args,
            remote,
            remote_workdir,
            RunTestDependencies(
                parse_test_metadata=parse_test_metadata,
                run_local_test=run_local_test,
                run_remote_test=run_remote_test,
                run_remote_differential_comparison=_load_remote_differential_comparison_function(),
                run_local_test_case_payload=run_local_payload,
                run_remote_test_case_payload=run_remote_payload,
                load_case_result_payload=load_case_payload,
                find_case_result_payload=find_case_payload,
                compare_result_payload_objects=compare_payloads,
                compare_result_files=compare_result,
                compare_remote_result_files=compare_remote_result,
            ),
        )

    if args.command == "profile-bench":
        from run_profile_command import RunProfileDependencies, handle_run_profile_command

        return handle_run_profile_command(
            parser,
            args,
            remote,
            remote_workdir,
            RunProfileDependencies(
                load_profile_functions=_load_profile_functions,
                resolve_existing_path=_resolve_existing_path,
                render_result=_render_result,
                build_profile_report=_build_profile_report,
                profile_hint=_profile_bench_hint,
            ),
        )

    if args.command == "profile-report":
        report = _build_profile_report(
            _resolve_existing_path(parser, args.profile_dir, "Profile directory"),
            args.target_op,
            top_count=args.top,
            output_format=args.format,
        )
        print(report)
        return 0

    from run_bench_command import RunBenchDependencies, handle_run_bench_command

    return handle_run_bench_command(
        parser,
        args,
        remote,
        remote_workdir,
        RunBenchDependencies(
            load_bench_functions=_load_bench_functions,
            resolve_existing_path=_resolve_existing_path,
            resolve_optional_existing_path=_resolve_optional_existing_path,
            derived_perf_path=_derived_perf_path,
            render_result=_render_result,
            load_compare_perf=_load_compare_perf_function,
            active_optimize_round_context=_active_optimize_round_context,
            append_optimize_timing_event=_append_optimize_timing_event,
            hint=_RUN_BENCH_HINT,
        ),
    )


def _resolve_existing_path(
    parser: argparse.ArgumentParser,
    raw_path: str,
    label: str,
) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        parser.error(f"{label} path does not exist: {path}")
    return path


def _resolve_optional_existing_path(
    parser: argparse.ArgumentParser,
    raw_path: str | None,
    label: str,
) -> Path | None:
    if raw_path is None:
        return None
    return _resolve_existing_path(parser, raw_path, label)


def _derived_perf_path(operator_file: Path) -> Path:
    return operator_file.parent / f"{operator_file.stem}_perf.txt"
def _render_result(result: ResultPayload, skip_stdout: bool) -> None:
    stdout = result["stdout"]
    stderr = result["stderr"]
    if stdout and not skip_stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def _resolve_remote_execution(args: argparse.Namespace) -> tuple[str | None, str | None]:
    resolve_remote_execution = _load_remote_execution_function()
    return resolve_remote_execution(
        getattr(args, "remote", None),
        getattr(args, "remote_workdir", None),
    )


def _load_test_functions() -> tuple[Any, Any, Any]:
    with _script_dir_on_path():
        module = importlib.import_module("run_test_api")
    return (
        getattr(module, "parse_test_metadata"),
        getattr(module, "run_local_test"),
        getattr(module, "run_remote_test"),
    )


def _load_remote_differential_comparison_function() -> Any:
    with _script_dir_on_path():
        module = importlib.import_module("run_test_api")
    return getattr(module, "run_remote_differential_comparison")


def _load_test_payload_functions() -> tuple[Any, Any]:
    with _script_dir_on_path():
        module = importlib.import_module("run_test_api")
    return (
        getattr(module, "run_local_test_case_payload"),
        getattr(module, "run_remote_test_case_payload"),
    )


def _load_compare_result_functions() -> tuple[Any, Any]:
    with _script_dir_on_path():
        module = importlib.import_module("compare_result")
    return (
        getattr(module, "compare_result_files"),
        getattr(module, "compare_remote_result_files"),
    )


def _load_compare_result_payload_functions() -> tuple[Any, Any, Any]:
    with _script_dir_on_path():
        module = importlib.import_module("compare_result")
    return (
        getattr(module, "load_case_result_payload"),
        getattr(module, "find_case_result_payload"),
        getattr(module, "compare_result_payload_objects"),
    )


def _load_bench_functions() -> tuple[ParseMetadataFn, Any, Any]:
    with _script_dir_on_path():
        from bench_contract import parse_bench_metadata
        from run_bench_api import run_local_bench, run_remote_bench

    return (
        cast(ParseMetadataFn, parse_bench_metadata),
        run_local_bench,
        run_remote_bench,
    )


def _load_compare_perf_function() -> ComparePerfFn:
    with _script_dir_on_path():
        from perf_artifacts import compare_perf_files

    return cast(ComparePerfFn, compare_perf_files)


def _load_remote_execution_function() -> ResolveRemoteExecutionFn:
    with _script_dir_on_path():
        from remote_execution_env import resolve_remote_execution

    return cast(ResolveRemoteExecutionFn, resolve_remote_execution)


def _load_profile_functions() -> tuple[Any, Any]:
    with _script_dir_on_path():
        module = importlib.import_module("run_profile_api")

    return (
        getattr(module, "run_local_profile_bench"),
        getattr(module, "run_remote_profile_bench"),
    )


def _build_profile_report(
    profile_dir: Path,
    target_op: str | None = None,
    top_count: int = 5,
    output_format: str = "markdown",
) -> str:
    script = SCRIPT_DIR.parents[1] / "ascend-npu-profile-operator" / "scripts" / "reporter.py"
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
