from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from triton_agent.commands.comparison import handle_compare_perf, handle_compare_result
from triton_agent.commands.execution import handle_run_bench, handle_run_test
from triton_agent.commands.generation import handle_gen_bench, handle_gen_test
from triton_agent.commands.generation import handle_gen_eval
from triton_agent.commands.generation import handle_gen_eval_batch
from triton_agent.commands.optimize import (
    handle_optimize,
    handle_optimize_batch,
    handle_optimize_status,
)
from triton_agent.models import CommandKind


_Handler = Callable[[argparse.ArgumentParser, argparse.Namespace], int]
_AGENT_CHOICES = ("codex", "opencode", "pi", "claude")
_COMPARE_LEVEL_CHOICES = ("strict", "balanced", "relaxed")
_FORMAT_CHOICES = ("text", "markdown")
_TEST_MODE_CHOICES = ("standalone", "differential")
_BENCH_MODE_CHOICES = ("standalone", "msprof")
_RESUME_CHOICES = ("auto", "continue", "fresh")
_SUPERVISE_CHOICES = ("on", "off")


@dataclass(frozen=True)
class _CommandSpec:
    handler: _Handler
    input_mode: str = "input"
    has_output: bool = True
    has_verbose: bool = True
    has_remote: bool = False
    keep_remote_workdir: bool = False
    has_agent: bool = False
    has_interact: bool = False
    has_show_output: bool = False
    has_test_mode: bool = False
    test_mode_default: str | None = None
    has_bench_mode: bool = False
    bench_mode_default: str | None = None
    has_optimize_options: bool = False
    max_concurrency_default: int | None = None
    has_force_overwrite: bool = False
    has_format: bool = False


_COMMAND_SPECS: dict[CommandKind, _CommandSpec] = {
    CommandKind.GEN_EVAL: _CommandSpec(
        handler=handle_gen_eval,
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="differential",
        has_bench_mode=True,
        bench_mode_default="standalone",
        has_force_overwrite=True,
    ),
    CommandKind.GEN_EVAL_BATCH: _CommandSpec(
        handler=handle_gen_eval_batch,
        has_output=False,
        has_remote=True,
        has_agent=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="differential",
        has_bench_mode=True,
        bench_mode_default="standalone",
        max_concurrency_default=2,
    ),
    CommandKind.GEN_TEST: _CommandSpec(
        handler=handle_gen_test,
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        test_mode_default="standalone",
        has_force_overwrite=True,
    ),
    CommandKind.RUN_TEST: _CommandSpec(
        handler=handle_run_test,
        input_mode="run-test",
        has_remote=True,
        keep_remote_workdir=True,
        has_test_mode=True,
    ),
    CommandKind.GEN_BENCH: _CommandSpec(
        handler=handle_gen_bench,
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_bench_mode=True,
        bench_mode_default="standalone",
        has_force_overwrite=True,
    ),
    CommandKind.RUN_BENCH: _CommandSpec(
        handler=handle_run_bench,
        input_mode="run-bench",
        has_remote=True,
        keep_remote_workdir=True,
        has_bench_mode=True,
    ),
    CommandKind.COMPARE_RESULT: _CommandSpec(
        handler=handle_compare_result,
        input_mode="compare-result",
        has_output=False,
        has_remote=True,
    ),
    CommandKind.COMPARE_PERF: _CommandSpec(
        handler=handle_compare_perf,
        input_mode="compare-perf",
        has_output=False,
        has_verbose=False,
    ),
    CommandKind.OPTIMIZE_STATUS: _CommandSpec(
        handler=handle_optimize_status,
        has_output=False,
        has_format=True,
    ),
    CommandKind.OPTIMIZE: _CommandSpec(
        handler=handle_optimize,
        has_remote=True,
        has_agent=True,
        has_interact=True,
        has_show_output=True,
        has_test_mode=True,
        has_bench_mode=True,
        has_optimize_options=True,
    ),
    CommandKind.OPTIMIZE_BATCH: _CommandSpec(
        handler=handle_optimize_batch,
        has_output=False,
        has_remote=True,
        has_agent=True,
        has_show_output=True,
        has_test_mode=True,
        has_bench_mode=True,
        has_optimize_options=True,
        max_concurrency_default=2,
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="triton-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_kind in CommandKind:
        spec = _COMMAND_SPECS[command_kind]
        subparser = subparsers.add_parser(command_kind.value)
        subparser.set_defaults(command_kind=command_kind)
        _add_primary_arguments(subparser, spec)
        if spec.has_format:
            subparser.add_argument("--format", default="text", choices=_FORMAT_CHOICES)
        if spec.has_remote:
            subparser.add_argument("--remote")
            subparser.add_argument("--remote-workdir")
            if spec.keep_remote_workdir:
                subparser.add_argument("--keep-remote-workdir", action="store_true")
        if spec.has_output:
            subparser.add_argument("-o", "--output")
        if spec.has_verbose:
            subparser.add_argument("--verbose", action="store_true")
        if spec.has_interact:
            subparser.add_argument("--interact", action="store_true")
        if spec.has_show_output:
            subparser.add_argument("--show-output", action="store_true")
        if spec.has_agent:
            subparser.add_argument("--agent", default="codex", choices=_AGENT_CHOICES)
        if spec.has_test_mode:
            subparser.add_argument(
                "--test-mode",
                default=spec.test_mode_default,
                choices=_TEST_MODE_CHOICES,
            )
        if spec.has_bench_mode:
            subparser.add_argument(
                "--bench-mode",
                default=spec.bench_mode_default,
                choices=_BENCH_MODE_CHOICES,
            )
        if spec.has_optimize_options:
            subparser.add_argument("--min-rounds", type=int)
            subparser.add_argument("--resume", default="auto", choices=_RESUME_CHOICES)
            subparser.add_argument("--require-analysis", action="store_true")
            subparser.add_argument("--no-agent-session", action="store_true")
            subparser.add_argument("--supervise", default="off", choices=_SUPERVISE_CHOICES)
        if spec.max_concurrency_default is not None:
            subparser.add_argument("--max-concurrency", type=int, default=spec.max_concurrency_default)
        if spec.has_force_overwrite:
            subparser.add_argument("--force-overwrite", action="store_true")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_command_aliases(argv))
    command_kind = args.command_kind
    return _COMMAND_SPECS[command_kind].handler(parser, args)


def _add_primary_arguments(subparser: argparse.ArgumentParser, spec: _CommandSpec) -> None:
    if spec.input_mode == "run-test":
        subparser.add_argument("--test-file", required=True)
        subparser.add_argument("--operator-file", required=True)
        return
    if spec.input_mode == "run-bench":
        subparser.add_argument("--bench-file", required=True)
        subparser.add_argument("--operator-file", required=True)
        return
    if spec.input_mode == "compare-result":
        subparser.add_argument("--oracle-result", required=True)
        subparser.add_argument("--new-result", required=True)
        subparser.add_argument("--compare-level", default="balanced", choices=_COMPARE_LEVEL_CHOICES)
        return
    if spec.input_mode == "compare-perf":
        subparser.add_argument("--baseline", required=True)
        subparser.add_argument("--compare", required=True)
        return
    subparser.add_argument("-i", "--input", required=True)


def _normalize_command_aliases(argv: Optional[list[str]]) -> Optional[list[str]]:
    if argv is None or not argv:
        return argv
    aliases = {
        "gen_eval": "gen-eval",
        "gen_eval_batch": "gen-eval-batch",
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


if __name__ == "__main__":
    raise SystemExit(main())
