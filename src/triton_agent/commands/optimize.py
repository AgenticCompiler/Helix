from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from triton_agent.models import CommandKind
from triton_agent.optimize.batch import resolve_batch_optimize_operator_file, run_optimize_batch
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.render import render_optimize_status_results
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request
from triton_agent.optimize.status import inspect_optimize_status_workspace, scan_optimize_status_workspaces, workspace_has_optimize_artifacts
from triton_agent.optimize.validation import validate_optimize_options
from triton_agent.optimize.verify_batch import run_optimize_verify_batch
from triton_agent.optimize.verify import OptimizeVerifyOptions, prepare_optimize_verify_target, run_optimize_verify
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


def handle_optimize_status(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    if workspace_has_optimize_artifacts(root):
        results = [inspect_optimize_status_workspace(root, verbose=bool(getattr(args, "verbose", False)))]
        return render_optimize_status_results(
            results,
            output_format=str(getattr(args, "format", "text")),
        )
    workspace_candidates = sorted(path for path in root.iterdir() if path.is_dir())
    if not workspace_candidates:
        print(f"No operator workspaces found under {root}", file=sys.stderr)
        return 1
    results = scan_optimize_status_workspaces(root, verbose=bool(getattr(args, "verbose", False)))
    return render_optimize_status_results(results, output_format=str(getattr(args, "format", "text")))


def handle_optimize_verify(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    workspace = Path(args.input).expanduser().resolve()
    if not workspace.exists():
        parser.error(f"Input path does not exist: {workspace}")
    if not workspace.is_dir():
        parser.error(f"Input path is not a directory: {workspace}")

    options = OptimizeVerifyOptions(
        phase=cast(Literal["all", "test", "bench"], str(getattr(args, "phase", "all"))),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        keep_remote_workdir=bool(getattr(args, "keep_remote_workdir", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )
    try:
        target = prepare_optimize_verify_target(workspace)
        result = run_optimize_verify(target, options)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Verification directory: {result.verify_dir}")
    print(f"State file: {result.state_path}")
    print(f"Return code: {result.return_code}")
    return result.return_code


def handle_optimize_verify_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    return run_optimize_verify_batch(
        root,
        force_verify=bool(getattr(args, "force_verify", False)),
        options=OptimizeVerifyOptions(verbose=bool(getattr(args, "verbose", False))),
    )


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
        require_analysis=bool(getattr(args, "require_analysis", False)),
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
        supervise=_validate_supervise_mode(args),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        prompt=getattr(args, "prompt", None),
        target_chip=target_chip,
    )


def _validate_agent_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")
