from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.convert.batch import resolve_batch_convert_operator_file, run_convert_batch
from triton_agent.convert.models import ConvertOptions
from triton_agent.convert.orchestration import build_convert_request, run_convert_request
from triton_agent.convert.outputs import prepare_convert_target
from triton_agent.output import render_result
from triton_agent.verbose import emit_verbose_lines


def handle_convert(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    _validate_agent_options(parser, args)
    if input_path.is_dir():
        try:
            operator_path = resolve_batch_convert_operator_file(input_path)
        except ValueError as exc:
            parser.error(str(exc))
        workdir = input_path
    else:
        operator_path = input_path
        workdir = input_path.parent
    options = convert_options_from_args(args)
    request = build_convert_request(
        operator_path,
        operator_path,
        workdir,
        options,
    )
    output_path = request.output_path
    if output_path is None:
        parser.error("Internal error: convert request did not resolve an output path.")
    try:
        file_messages = prepare_convert_target(
            output_path,
            force_overwrite=options.force_overwrite,
        )
    except (FileExistsError, IsADirectoryError) as exc:
        parser.exit(2, f"{exc}\n")
    if options.verbose:
        emit_verbose_lines(sys.stderr, "files", file_messages)
    try:
        result = run_convert_request(request)
    except FileNotFoundError as exc:
        parser.error(
            f"Agent executable not found: {exc}. "
            f"Make sure the '{options.agent_name}' CLI is installed and available in PATH."
        )
    render_result(result, show_output=request.show_output)
    return result.return_code


def handle_convert_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    if args.max_concurrency < 1:
        parser.error("--max-concurrency must be at least 1")
    return run_convert_batch(
        root,
        convert_options_from_args(args),
        max_concurrency=args.max_concurrency,
    )


def _validate_agent_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")


def convert_options_from_args(args: argparse.Namespace) -> ConvertOptions:
    return ConvertOptions(
        interact=bool(getattr(args, "interact", False)),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        force_overwrite=bool(getattr(args, "force_overwrite", False)),
        agent_name=args.agent,
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        prompt=getattr(args, "prompt", None),
    )
