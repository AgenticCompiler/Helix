from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, TextIO

from triton_agent.agent import AgentRunner
from triton_agent.commands.comparison import handle_compare_perf, handle_compare_result
from triton_agent.commands.execution import handle_run_bench, handle_run_test
from triton_agent.commands.generation import handle_gen_bench, handle_gen_test
from triton_agent.commands.optimize import (
    handle_optimize,
    handle_optimize_batch,
    handle_optimize_status,
)
from triton_agent.comparison import (
    compare_perf_files as _compare_perf_files,
    compare_remote_result_files as _compare_remote_result_files,
    compare_result_files as _compare_result_files,
)
from triton_agent.execution import run_local_bench as _run_local_bench
from triton_agent.execution import run_local_test as _run_local_test
from triton_agent.execution import run_remote_bench as _run_remote_bench
from triton_agent.execution import run_remote_test as _run_remote_test
from triton_agent.generation import prepare_generation_target as _prepare_generation_target
from triton_agent.models import AgentResult, CommandKind
from triton_agent.output import render_result as _render_result
from triton_agent.runner_factory import create_runner as _create_runner


def run_local_test(test_file: Path, operator_file: Path, test_mode: str) -> tuple[AgentResult, Path | None]:
    return _run_local_test(test_file, operator_file, test_mode)


def run_remote_test(
    test_file: Path,
    operator_file: Path,
    test_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[AgentResult, Path | None, str]:
    return _run_remote_test(
        test_file,
        operator_file,
        test_mode,
        remote,
        remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def compare_result_files(oracle_result: Path, new_result: Path, compare_level: str) -> int:
    return _compare_result_files(oracle_result, new_result, compare_level)


def compare_remote_result_files(
    oracle_result: Path,
    new_result: Path,
    compare_level: str,
    remote: str,
    remote_workdir: str | None,
    *,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> int:
    return _compare_remote_result_files(
        oracle_result,
        new_result,
        compare_level,
        remote,
        remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def run_local_bench(
    bench_file: Path, operator_file: Path, bench_mode: str
) -> tuple[AgentResult, Path | None]:
    return _run_local_bench(bench_file, operator_file, bench_mode)


def run_remote_bench(
    bench_file: Path,
    operator_file: Path,
    bench_mode: str,
    remote: str,
    remote_workdir: str | None,
    *,
    keep_remote_workdir: bool = False,
    verbose: bool = False,
    stderr: TextIO | None = None,
) -> tuple[AgentResult, Path | None, str]:
    return _run_remote_bench(
        bench_file,
        operator_file,
        bench_mode,
        remote,
        remote_workdir,
        keep_remote_workdir=keep_remote_workdir,
        verbose=verbose,
        stderr=stderr,
    )


def parse_perf_file(path: Path) -> dict[str, float]:
    from triton_agent.bench_runner import parse_perf_file as _parse_perf_file

    return _parse_perf_file(path)


def compare_perf_files(baseline_perf: Path, compare_perf: Path) -> int:
    return _compare_perf_files(baseline_perf, compare_perf)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triton-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_kind in CommandKind:
        subparser = subparsers.add_parser(command_kind.value)
        subparser.set_defaults(command_kind=command_kind)
        if command_kind == CommandKind.RUN_TEST:
            subparser.add_argument("--test-file", required=True)
            subparser.add_argument("--operator-file", required=True)
        elif command_kind == CommandKind.RUN_BENCH:
            subparser.add_argument("--bench-file", required=True)
            subparser.add_argument("--operator-file", required=True)
        elif command_kind == CommandKind.COMPARE_RESULT:
            subparser.add_argument("--oracle-result", required=True)
            subparser.add_argument("--new-result", required=True)
            subparser.add_argument(
                "--compare-level",
                default="balanced",
                choices=["strict", "balanced", "relaxed"],
            )
        elif command_kind == CommandKind.COMPARE_PERF:
            subparser.add_argument("--baseline", required=True)
            subparser.add_argument("--compare", required=True)
        else:
            subparser.add_argument("-i", "--input", required=True)
        if command_kind in {
            CommandKind.GEN_TEST,
            CommandKind.GEN_BENCH,
            CommandKind.OPTIMIZE,
            CommandKind.OPTIMIZE_BATCH,
            CommandKind.RUN_TEST,
            CommandKind.RUN_BENCH,
            CommandKind.COMPARE_RESULT,
        }:
            subparser.add_argument("--remote")
            subparser.add_argument("--remote-workdir")
            if command_kind in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument("--keep-remote-workdir", action="store_true")
        if command_kind not in {
            CommandKind.COMPARE_RESULT,
            CommandKind.COMPARE_PERF,
            CommandKind.OPTIMIZE_STATUS,
            CommandKind.OPTIMIZE_BATCH,
        }:
            subparser.add_argument("-o", "--output")
        if command_kind != CommandKind.COMPARE_PERF:
            subparser.add_argument("--verbose", action="store_true")
        if command_kind not in {
            CommandKind.COMPARE_RESULT,
            CommandKind.COMPARE_PERF,
            CommandKind.OPTIMIZE_STATUS,
        }:
            if command_kind not in {
                CommandKind.RUN_TEST,
                CommandKind.RUN_BENCH,
            }:
                if command_kind != CommandKind.OPTIMIZE_BATCH:
                    subparser.add_argument("--interact", action="store_true")
                subparser.add_argument("--show-output", action="store_true")
            if command_kind not in {CommandKind.RUN_TEST, CommandKind.RUN_BENCH}:
                subparser.add_argument(
                    "--agent", default="codex", choices=["codex", "opencode", "pi", "claude"]
                )
        if command_kind in {
            CommandKind.GEN_TEST,
            CommandKind.RUN_TEST,
            CommandKind.OPTIMIZE,
            CommandKind.OPTIMIZE_BATCH,
        }:
            subparser.add_argument(
                "--test-mode",
                default=(
                    "standalone"
                    if command_kind == CommandKind.GEN_TEST
                    else None
                ),
                choices=["standalone", "differential"],
            )
        if command_kind in {
            CommandKind.GEN_BENCH,
            CommandKind.RUN_BENCH,
            CommandKind.OPTIMIZE,
            CommandKind.OPTIMIZE_BATCH,
        }:
            subparser.add_argument(
                "--bench-mode",
                default=(
                    "standalone" if command_kind == CommandKind.GEN_BENCH else None
                ),
                choices=["standalone", "msprof"],
            )
        if command_kind in {CommandKind.OPTIMIZE, CommandKind.OPTIMIZE_BATCH}:
            subparser.add_argument("--min-rounds", type=int)
            subparser.add_argument("--continue", dest="continue_optimize", action="store_true")
            subparser.add_argument("--no-agent-session", action="store_true")
        if command_kind == CommandKind.OPTIMIZE_BATCH:
            subparser.add_argument("--max-concurrency", type=int, default=2)
        if command_kind in {CommandKind.GEN_TEST, CommandKind.GEN_BENCH}:
            subparser.add_argument("--force-overwrite", action="store_true")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))

    command_kind: CommandKind = args.command_kind
    if command_kind == CommandKind.GEN_TEST:
        return handle_gen_test(parser, args)
    if command_kind == CommandKind.GEN_BENCH:
        return handle_gen_bench(parser, args)
    if command_kind == CommandKind.RUN_TEST:
        return handle_run_test(parser, args)
    if command_kind == CommandKind.RUN_BENCH:
        return handle_run_bench(parser, args)
    if command_kind == CommandKind.COMPARE_RESULT:
        return handle_compare_result(parser, args)
    if command_kind == CommandKind.COMPARE_PERF:
        return handle_compare_perf(parser, args)
    if command_kind == CommandKind.OPTIMIZE:
        return handle_optimize(parser, args)
    if command_kind == CommandKind.OPTIMIZE_BATCH:
        return handle_optimize_batch(parser, args)
    if command_kind == CommandKind.OPTIMIZE_STATUS:
        return handle_optimize_status(parser, args)
    raise AssertionError(f"Unhandled command kind: {command_kind}")


def _normalize_command_aliases(argv: Optional[list[str]]) -> Optional[list[str]]:
    if argv is None or not argv:
        return argv
    aliases = {
        "gen_test": "gen-test",
        "run_test": "run-test",
        "gen_bench": "gen-bench",
        "run_bench": "run-bench",
        "compare_result": "compare-result",
        "compare_perf": "compare-perf",
        "optimize_status": "optimize-status",
        "optimize_batch": "optimize-batch",
    }
    normalized = list(argv)
    normalized[0] = aliases.get(normalized[0], normalized[0])
    return normalized


def prepare_generation_target(
    command_kind: CommandKind, output_path: Path | None, force_overwrite: bool
) -> list[str]:
    return _prepare_generation_target(command_kind, output_path, force_overwrite)


def render_result(
    result: AgentResult,
    show_output: bool,
    stdout: Optional[TextIO] = None,
    stderr: Optional[TextIO] = None,
) -> None:
    _render_result(result, show_output=show_output, stdout=stdout, stderr=stderr)


def create_runner(agent_name: str) -> AgentRunner:
    return _create_runner(agent_name)


if __name__ == "__main__":
    raise SystemExit(main())
