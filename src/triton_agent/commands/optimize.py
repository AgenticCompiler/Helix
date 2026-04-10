from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.models import CommandKind
from triton_agent.optimize.batch import run_optimize_batch
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.render import render_optimize_status_results
from triton_agent.optimize.runtime import build_optimize_request, run_optimize_request
from triton_agent.optimize.status import scan_optimize_status_workspaces
from triton_agent.optimize.validation import validate_optimize_options
from triton_agent.output import render_result


def handle_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = optimize_run_options_from_args(args)
    try:
        validate_optimize_options(
            CommandKind.OPTIMIZE,
            min_rounds=options.min_rounds,
            max_concurrency=None,
            resume_mode=options.resume_mode,
            test_mode=options.test_mode,
            bench_mode=options.bench_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
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


def handle_optimize_status(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspace_candidates:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1
    results = scan_optimize_status_workspaces(root, verbose=bool(getattr(args, "verbose", False)))
    return render_optimize_status_results(results, output_format=str(getattr(args, "format", "text")))


def optimize_run_options_from_args(args: argparse.Namespace) -> OptimizeRunOptions:
    return OptimizeRunOptions(
        agent_name=args.agent,
        interact=bool(getattr(args, "interact", False)),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        min_rounds=getattr(args, "min_rounds", None),
        resume_mode=str(getattr(args, "resume", "auto")),
        require_analysis=bool(getattr(args, "require_analysis", False)),
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
    )
