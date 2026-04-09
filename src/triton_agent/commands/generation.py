from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.generation import (
    GenerationOptions,
    build_generation_request,
    prepare_generation_target,
    run_generation_request,
)
from triton_agent.models import CommandKind
from triton_agent.output import render_result
from triton_agent.verbose import emit_verbose_lines


def handle_gen_eval(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    return _handle_generation_command(parser, args, CommandKind.GEN_EVAL)


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
    workdir = input_path.parent
    options = generation_options_from_args(args)
    request = build_generation_request(
        command_kind,
        input_path,
        input_path,
        workdir,
        options,
    )
    try:
        file_messages = prepare_generation_target(
            command_kind,
            request.output_path,
            options.force_overwrite,
        )
    except (FileExistsError, IsADirectoryError) as exc:
        parser.exit(2, f"{exc}\n")
    if options.verbose:
        emit_verbose_lines(sys.stderr, "files", file_messages)
    result = run_generation_request(request)
    render_result(result, show_output=request.show_output)
    return result.return_code


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
    )
