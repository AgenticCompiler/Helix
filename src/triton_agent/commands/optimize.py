from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from triton_agent.models import CommandKind
from triton_agent.optimize.batch import resolve_batch_optimize_operator_file, run_optimize_batch
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request
from triton_agent.optimize.validation import validate_optimize_options
from triton_agent.output import render_result


def handle_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = optimize_run_options_from_args(args)
    _validate_agent_options(parser, args)
    try:
        validate_optimize_options(
            CommandKind.OPTIMIZE,
            min_rounds=options.min_rounds,
            max_concurrency=None,
            resume_mode=options.resume_mode,
            reset_optimize=options.reset_optimize,
            test_mode=options.test_mode,
            bench_mode=options.bench_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    if input_path.is_dir():
        try:
            input_path = resolve_batch_optimize_operator_file(input_path)
        except ValueError as exc:
            parser.error(str(exc))
        workdir = input_path.parent
    else:
        workdir = input_path.parent
    try:
        request = build_optimize_request(input_path, workdir, options)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = run_optimize_request(request)
    except FileNotFoundError as exc:
        parser.error(
            f"Agent executable not found: {exc}. "
            f"Make sure the '{options.agent_name}' CLI is installed and available in PATH."
        )
    render_result(result, show_output=request.show_output)
    return result.return_code


def handle_optimize_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = optimize_run_options_from_args(args)
    try:
        validate_optimize_options(
            CommandKind.OPTIMIZE_BATCH,
            min_rounds=options.min_rounds,
            max_concurrency=args.max_concurrency,
            resume_mode=options.resume_mode,
            reset_optimize=options.reset_optimize,
            test_mode=options.test_mode,
            bench_mode=options.bench_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    return run_optimize_batch(root, options, max_concurrency=args.max_concurrency)


def _validate_supervise_mode(args: argparse.Namespace) -> Literal["on", "off"]:
    value = getattr(args, "supervise", "off")
    supervise = str(value)
    if supervise == "on":
        return "on"
    if supervise == "off":
        return "off"
    raise ValueError(f"--supervise must be 'on' or 'off', got {value!r}")


def optimize_run_options_from_args(args: argparse.Namespace) -> OptimizeRunOptions:
    target_chip = cast(Literal["A3", "A5"], getattr(args, "target_chip", "A5"))
    compiler_source_enabled = bool(getattr(args, "enable_compiler_source_analysis", False))
    return OptimizeRunOptions(
        agent_name=args.agent,
        interact=bool(getattr(args, "interact", False)),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        min_rounds=getattr(args, "min_rounds", None),
        resume_mode=str(getattr(args, "resume", "auto")),
        reset_optimize=bool(getattr(args, "reset_optimize", False)),
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
        supervise=_validate_supervise_mode(args),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        prompt=getattr(args, "prompt", None),
        target_chip=target_chip,
        compiler_source_analysis="auto" if compiler_source_enabled else "off",
    )


def _validate_agent_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")
