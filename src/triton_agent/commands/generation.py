from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.commands.input_resolution import resolve_single_operator_input
from triton_agent.generation.batch import run_gen_eval_batch
from triton_agent.generation.batch import resolve_batch_gen_eval_operator_file
from triton_agent.generation.models import GenerationOptions
from triton_agent.generation.outputs import prepare_generation_targets
from triton_agent.generation.orchestration import build_generation_request, run_generation_request
from triton_agent.models import CommandKind
from triton_agent.output import render_result
from triton_agent.verbose import emit_verbose_lines


def handle_gen_eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_generation_command(parser, args, CommandKind.GEN_EVAL)


def handle_gen_eval_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    max_concurrency = args.max_concurrency
    if max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    return run_gen_eval_batch(root, generation_options_from_args(args), max_concurrency=max_concurrency)


def handle_gen_test(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_generation_command(parser, args, CommandKind.GEN_TEST)


def handle_gen_bench(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_generation_command(parser, args, CommandKind.GEN_BENCH)


def _handle_generation_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    command_kind: CommandKind,
) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    _validate_agent_options(parser, args)
    try:
        operator_path, workdir = resolve_single_operator_input(
            input_path,
            resolve_operator_file=resolve_batch_gen_eval_operator_file,
        )
    except ValueError as exc:
        parser.error(str(exc))
    options = generation_options_from_args(args)
    request = build_generation_request(
        command_kind,
        operator_path,
        operator_path,
        workdir,
        options,
    )
    try:
        file_messages = prepare_generation_targets(
            command_kind,
            operator_path,
            request.output_path,
            test_mode=request.test_mode,
            force_overwrite=options.force_overwrite,
        )
    except (FileExistsError, IsADirectoryError) as exc:
        parser.exit(2, f"{exc}\n")
    if options.verbose:
        emit_verbose_lines(sys.stderr, "files", file_messages)
    try:
        result = run_generation_request(request)
    except FileNotFoundError as exc:
        parser.error(
            f"Agent executable not found: {exc}. "
            f"Make sure the '{options.agent_name}' CLI is installed and available in PATH."
        )
    render_result(result, show_output=request.show_output)
    return result.return_code


def _validate_agent_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")


def generation_options_from_args(args: argparse.Namespace) -> GenerationOptions:
    return GenerationOptions(
        interact=bool(getattr(args, "interact", False)),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        force_overwrite=bool(getattr(args, "force_overwrite", False)),
        agent_name=args.agent,
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        min_rounds=getattr(args, "min_rounds", None),
        continue_optimize=bool(getattr(args, "continue_optimize", False)),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        prompt=getattr(args, "prompt", None),
    )
