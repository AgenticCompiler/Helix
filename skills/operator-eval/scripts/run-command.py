from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triton-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_test = subparsers.add_parser("run-test")
    run_test.add_argument("--test-file", required=True)
    run_test.add_argument("--operator-file", required=True)
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

    profile_bench = subparsers.add_parser("profile-bench")
    profile_bench.add_argument("--bench-file", required=True)
    profile_bench.add_argument("--operator-file", required=True)
    profile_bench.add_argument("--bench-mode", choices=["standalone", "msprof"])
    profile_bench.add_argument("--bench", type=int)
    profile_bench.add_argument("--target-op")
    profile_bench.add_argument("--remote")
    profile_bench.add_argument("--remote-workdir")
    profile_bench.add_argument("--keep-remote-workdir", action="store_true")
    profile_bench.add_argument("--verbose", action="store_true")

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
        return compare_perf_files(baseline_perf, compare_perf)

    if args.command == "run-test":
        parse_test_metadata, run_local_test, run_remote_test = _load_test_functions()
        test_file = _resolve_existing_path(parser, args.test_file, "Test file")
        operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
        resolved_test_mode = args.test_mode or _resolve_test_mode_from_metadata(test_file)
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
                result, archived_result = run_local_test(test_file, operator_file, resolved_test_mode)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _render_result(result, show_output=True)
        print(f"Return code: {result['return_code']}")
        if archived_result is not None:
            print(f"Archived result: {archived_result}")
        if args.remote and args.keep_remote_workdir:
            print(f"Remote workspace: {remote_workspace}")
        return int(result["return_code"])

    if args.command == "profile-bench":
        run_local_profile_bench, run_remote_profile_bench = _load_profile_functions()
        bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
        operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
        resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(bench_file)
        try:
            if args.remote:
                result, profile_dir, remote_workspace = run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    resolved_bench_mode,
                    args.remote,
                    args.remote_workdir,
                    bench_case=args.bench,
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
                )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        _render_result(result, show_output=True)
        print(f"Return code: {result['return_code']}")
        if profile_dir is not None:
            print(f"Profile directory: {profile_dir}")
            print(_build_profile_report(profile_dir, args.target_op))
        if args.remote and args.keep_remote_workdir:
            print(f"Remote workspace: {remote_workspace}")
        return int(result["return_code"])

    bench_file = _resolve_existing_path(parser, args.bench_file, "Bench file")
    operator_file = _resolve_existing_path(parser, args.operator_file, "Operator file")
    parse_bench_metadata, run_local_bench, run_remote_bench = _load_bench_functions()
    resolved_bench_mode = args.bench_mode or _resolve_bench_mode_from_metadata(bench_file)
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
            result, perf_path = run_local_bench(bench_file, operator_file, resolved_bench_mode)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    _render_result(result, show_output=True)
    print(f"Return code: {result['return_code']}")
    if perf_path is not None:
        print(f"Perf file: {perf_path}")
    if args.remote and args.keep_remote_workdir:
        print(f"Remote workspace: {remote_workspace}")
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
    parse_test_metadata, _, _ = _load_test_functions()
    metadata = parse_test_metadata(test_file)
    mode = metadata.get("test-mode")
    if mode not in {"standalone", "differential"}:
        raise ValueError(f"Test metadata is missing required 'test-mode' entry: {test_file}")
    return mode


def _resolve_bench_mode_from_metadata(bench_file: Path) -> str:
    parse_bench_metadata, _, _ = _load_bench_functions()
    metadata = parse_bench_metadata(bench_file)
    mode = metadata.get("bench-mode")
    if mode not in {"standalone", "msprof"}:
        raise ValueError(f"Benchmark metadata is missing required 'bench-mode' entry: {bench_file}")
    return mode


def _render_result(result, show_output: bool) -> None:
    stdout = str(result["stdout"])
    stderr = str(result["stderr"])
    if stdout and not show_output:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")


def _load_test_functions():
    _ensure_script_dir_on_path()
    from test_runner import parse_test_metadata, run_local_test, run_remote_test

    return parse_test_metadata, run_local_test, run_remote_test


def _load_bench_functions():
    _ensure_script_dir_on_path()
    from bench_runner import parse_bench_metadata, run_local_bench, run_remote_bench

    return parse_bench_metadata, run_local_bench, run_remote_bench


def _load_compare_result_functions():
    _ensure_script_dir_on_path()
    from compare_result import compare_remote_result_files, compare_result_files

    return compare_result_files, compare_remote_result_files


def _load_compare_perf_function():
    _ensure_script_dir_on_path()
    from compare_perf import compare_perf_files

    return compare_perf_files


def _load_profile_functions():
    _ensure_script_dir_on_path()
    from profile_runner import run_local_profile_bench, run_remote_profile_bench

    return run_local_profile_bench, run_remote_profile_bench


def _build_profile_report(profile_dir: Path, target_op: str | None) -> str:
    script = SCRIPT_DIR.parents[1] / "ascend-npu-operator-profiler" / "scripts" / "profile_summary.py"
    spec = importlib.util.spec_from_file_location("profile_summary_runtime", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load profiler summary script: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_profile_report(profile_dir, target_op=target_op)


def _ensure_script_dir_on_path() -> None:
    script_dir = str(SCRIPT_DIR)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
