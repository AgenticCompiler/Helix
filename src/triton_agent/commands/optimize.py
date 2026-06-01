from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from triton_agent.commands.input_resolution import resolve_single_operator_input
from triton_agent.models import CommandKind
from triton_agent.optimize.batch import resolve_batch_optimize_operator_file, run_optimize_batch
from triton_agent.optimize.models import OptimizeRunOptions
from triton_agent.optimize.orchestration import build_optimize_request, run_optimize_request
from triton_agent.optimize.validation import validate_optimize_options
from triton_agent.optimize_upload.client import UploadUrlMissingError
from triton_agent.optimize_upload.workflow import upload_optimize_workspace
from triton_agent.output import render_result
from triton_agent.report.workspace import generate_workspace_report
from triton_agent.verbose import emit_verbose


def handle_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    options = optimize_run_options_from_args(args)
    _validate_agent_options(parser, args)
    _validate_interactive_round_mode(parser, options)
    try:
        validate_optimize_options(
            CommandKind.OPTIMIZE,
            min_rounds=options.min_rounds,
            max_concurrency=None,
            resume_mode=options.resume_mode,
            reset_optimize=options.reset_optimize,
            test_mode=options.test_mode,
            bench_mode=options.bench_mode,
            target_chip=options.target_chip,
            enable_cann_ext_api=options.enable_cann_ext_api,
        )
    except ValueError as exc:
        parser.error(str(exc))

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")
    try:
        input_path, workdir = resolve_single_operator_input(
            input_path,
            resolve_operator_file=resolve_batch_optimize_operator_file,
        )
    except ValueError as exc:
        parser.error(str(exc))
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
    if result.succeeded:
        if options.upload_enabled:
            _maybe_upload_workspace(workdir, verbose=options.verbose)
        if options.report:
            _maybe_generate_report(workdir, options)
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
            target_chip=options.target_chip,
            enable_cann_ext_api=options.enable_cann_ext_api,
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    return run_optimize_batch(root, options, max_concurrency=args.max_concurrency)


def _validate_round_mode(
    args: argparse.Namespace,
) -> Literal["continuous", "checked", "supervised"]:
    value = str(getattr(args, "round_mode", "continuous"))
    if value not in {"continuous", "checked", "supervised"}:
        raise ValueError(
            "--round-mode must be one of 'continuous', 'checked', or 'supervised'",
        )
    return cast(Literal["continuous", "checked", "supervised"], value)


def _validate_interactive_round_mode(
    parser: argparse.ArgumentParser,
    options: OptimizeRunOptions,
) -> None:
    if options.interact and options.round_mode != "continuous":
        parser.error("--interact is only supported with --round-mode continuous")


def optimize_run_options_from_args(args: argparse.Namespace) -> OptimizeRunOptions:
    target_chip = cast(Literal["A3", "A5"], getattr(args, "target_chip", "A5"))
    optimize_target = cast(
        Literal["kernel", "operator"],
        getattr(args, "optimize_target", "kernel"),
    )
    optimize_knowledge = cast(
        Literal["v1", "v2", "v3"],
        getattr(args, "optimize_knowledge", "v1"),
    )
    compiler_source_enabled = bool(getattr(args, "enable_compiler_source_analysis", False))
    cann_ext_api_enabled = bool(getattr(args, "enable_cann_ext_api", False))
    agent_hooks_enabled = bool(getattr(args, "enable_agent_hooks", False))
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
        round_mode=_validate_round_mode(args),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        prompt=getattr(args, "prompt", None),
        target_chip=target_chip,
        optimize_target=optimize_target,
        optimize_knowledge=optimize_knowledge,
        compiler_source_analysis="auto" if compiler_source_enabled else "off",
        enable_cann_ext_api=cann_ext_api_enabled,
        enable_agent_hooks=agent_hooks_enabled,
        log_tools=bool(getattr(args, "log_tools", False)),
        upload_enabled=not bool(getattr(args, "no_upload", False)),
        report=not bool(getattr(args, "no_report", False)),
        skills_source_dir=(
            Path(args.skills_source_dir).expanduser().resolve()
            if getattr(args, "skills_source_dir", None)
            else None
        ),
    )


def _validate_agent_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")


def _maybe_upload_workspace(workspace: Path, *, verbose: bool) -> None:
    try:
        upload_optimize_workspace(workspace, verbose=verbose)
    except UploadUrlMissingError:
        if verbose:
            emit_verbose(sys.stderr, "upload", "Auto-upload skipped: URL not set.")
    except (ValueError, RuntimeError) as exc:
        if verbose:
            emit_verbose(sys.stderr, "upload", f"Auto-upload warning: {exc}")


def _maybe_generate_report(workspace: Path, options: OptimizeRunOptions) -> None:
    try:
        if options.verbose:
            emit_verbose(sys.stderr, "report", "Auto-report: generating report.md...")
        report_ok, report_msg = generate_workspace_report(
            workspace=workspace,
            agent_name=options.agent_name,
            show_output=options.show_output,
        )
        if options.verbose:
            status = "completed" if report_ok else f"warning: {report_msg}"
            emit_verbose(sys.stderr, "report", f"Auto-report: {status}")
    except Exception as exc:
        if options.verbose:
            emit_verbose(sys.stderr, "report", f"Auto-report warning: {exc}")
