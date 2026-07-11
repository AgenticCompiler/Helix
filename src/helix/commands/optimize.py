from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast

from helix.commands.input_resolution import resolve_single_operator_input
from helix.report.workspace import generate_workspace_report
from helix.models import CommandKind
from helix.batch.affinity import resolve_batch_concurrency
from helix.optimize.batch import resolve_batch_optimize_operator_file, run_optimize_batch
from helix.optimize.models import OptimizeRunOptions
from helix.optimize.orchestration import build_optimize_request, run_optimize_request
from helix.optimize.validation import validate_optimize_options
from helix.optimize_upload.client import UploadUrlMissingError
from helix.optimize_upload.workflow import upload_optimize_workspace
from helix.terminal.render import render_result


def handle_optimize(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if getattr(args, "concurrency", None) is not None:
        return handle_optimize_batch(parser, args)
    args.system_prompt = _resolve_system_prompt_argument(parser, args)
    options = optimize_run_options_from_args(args)
    _validate_agent_options(parser, args, options)
    try:
        validate_optimize_options(
            CommandKind.OPTIMIZE,
            min_rounds=options.min_rounds,
            min_speedup=options.min_speedup,
            round_batch_size=options.round_batch_size,
            max_concurrency=None,
            resume_mode=options.resume_mode,
            reset_optimize=options.reset_optimize,
            test_mode=options.test_mode,
            bench_mode=options.bench_mode,
            target_chip=options.target_chip,
            enable_cann_ext_api=options.enable_cann_ext_api,
            language=options.language,
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
    render_result(result, skip_stdout=request.stream_output)

    # Auto-upload after successful optimize
    if result.return_code == 0 and options.upload_enabled:
        try:
            if options.verbose:
                print("Auto-upload enabled. Uploading workspace...", file=sys.stderr)
            upload_optimize_workspace(workdir, verbose=options.verbose)
            if options.verbose:
                print("Auto-upload completed successfully.", file=sys.stderr)
        except UploadUrlMissingError:
            if options.verbose:
                print("Auto-upload skipped: HELIX_OPTIMIZE_UPLOAD_URL not set.", file=sys.stderr)
        except (ValueError, RuntimeError) as exc:
            if options.verbose:
                print(f"Auto-upload warning: {exc}", file=sys.stderr)
        # Upload failure does NOT change the optimize exit code

    # Auto-report after successful optimize
    if result.return_code == 0 and options.report:
        try:
            if options.verbose:
                print("Auto-report enabled. Generating report.md...", file=sys.stderr)
            report_ok, report_msg = generate_workspace_report(
                workspace=workdir,
                agent_name=options.agent_name,
                show_output=options.stream_output,
            )
            if options.verbose:
                if report_ok:
                    print(f"Auto-report completed: {report_msg}", file=sys.stderr)
                else:
                    print(f"Auto-report warning: {report_msg}", file=sys.stderr)
        except Exception as exc:
            if options.verbose:
                print(f"Auto-report warning: {exc}", file=sys.stderr)

    return result.return_code


def handle_optimize_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    args.system_prompt = _resolve_system_prompt_argument(parser, args)
    options = optimize_run_options_from_args(args)
    _validate_agent_options(parser, args, options)
    try:
        max_concurrency = resolve_batch_concurrency(
            args.concurrency,
            getattr(args, "npu_devices", None),
            getattr(args, "workers_per_npu", None),
            ignore_workers_per_npu=bool(getattr(args, "enable_mcp", False)),
        )
    except ValueError as exc:
        parser.error(str(exc))
    try:
        validate_optimize_options(
            CommandKind.OPTIMIZE_BATCH,
            min_rounds=options.min_rounds,
            min_speedup=options.min_speedup,
            round_batch_size=options.round_batch_size,
            max_concurrency=max_concurrency,
            resume_mode=options.resume_mode,
            reset_optimize=options.reset_optimize,
            test_mode=options.test_mode,
            bench_mode=options.bench_mode,
            target_chip=options.target_chip,
            enable_cann_ext_api=options.enable_cann_ext_api,
            language=options.language,
        )
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    return run_optimize_batch(
        root,
        options,
        max_concurrency=max_concurrency,
        operator_filter=getattr(args, "operator_filter", None),
    )


def _validate_round_mode(args: argparse.Namespace) -> Literal["checked", "supervised"]:
    value = getattr(args, "round_mode", "checked")
    round_mode = str(value)
    if round_mode not in {"checked", "supervised"}:
        raise ValueError(f"--round-mode must be one of 'checked' or 'supervised', got {value!r}")
    return cast(Literal["checked", "supervised"], round_mode)


def _resolve_system_prompt_argument(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str | None:
    value = getattr(args, "system_prompt", None)
    if value is None:
        return None
    raw_value = str(value)
    if raw_value.startswith("@"):
        path_text = raw_value[1:].strip()
        if not path_text:
            parser.error("--system-prompt file reference must be in the form @path")
        path = Path(path_text).expanduser().resolve()
        if not path.exists():
            parser.error(f"--system-prompt file does not exist: {path}")
        if not path.is_file():
            parser.error(f"--system-prompt path is not a file: {path}")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            parser.error(f"--system-prompt file is not valid UTF-8: {path}: {exc}")
        except OSError as exc:
            parser.error(f"Failed to read --system-prompt file {path}: {exc}")
        stripped_content = content.strip()
        return stripped_content or None
    stripped_value = raw_value.strip()
    return stripped_value or None


def optimize_run_options_from_args(args: argparse.Namespace) -> OptimizeRunOptions:
    interact = bool(getattr(args, "interact", False))
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
    subagent_enabled = bool(getattr(args, "enable_subagent", False))
    upload_enabled = not bool(getattr(args, "no_upload", False))
    log_tools_enabled = bool(getattr(args, "log_tools", False))
    round_batch_size = 99 if interact else getattr(args, "round_batch_size", 5)
    post_optimize_command_value = getattr(args, "post_optimize_command", None)
    post_optimize_command = (
        post_optimize_command_value
        if isinstance(post_optimize_command_value, str) and post_optimize_command_value.strip()
        else None
    )
    return OptimizeRunOptions(
        agent_name=args.agent,
        interact=interact,
        language=getattr(args, "lang", "triton"),
        verbose=bool(getattr(args, "verbose", False)),
        stream_output=bool(getattr(args, "stream_output", True)),
        remote=getattr(args, "remote", None),
        remote_workdir=getattr(args, "remote_workdir", None),
        min_rounds=getattr(args, "min_rounds", 5),
        min_speedup=getattr(args, "min_speedup", None),
        round_batch_size=round_batch_size,
        resume_mode=str(getattr(args, "resume", "auto")),
        reset_optimize=bool(getattr(args, "reset_optimize", False)),
        no_agent_session=bool(getattr(args, "no_agent_session", False)),
        round_mode=_validate_round_mode(args),
        output=getattr(args, "output", None),
        test_mode=getattr(args, "test_mode", None),
        bench_mode=getattr(args, "bench_mode", None),
        npu_devices=getattr(args, "npu_devices", None),
        workers_per_npu=getattr(args, "workers_per_npu", None),
        prompt=getattr(args, "prompt", None),
        system_prompt=getattr(args, "system_prompt", None),
        post_optimize_command=post_optimize_command,
        target_chip=target_chip,
        optimize_target=optimize_target,
        optimize_knowledge=optimize_knowledge,
        compiler_source_analysis="auto" if compiler_source_enabled else "off",
        enable_cann_ext_api=cann_ext_api_enabled,
        enable_subagent=subagent_enabled,
        enable_agent_hooks=agent_hooks_enabled,
        upload_enabled=upload_enabled,
        report=bool(getattr(args, "enable_report", False)) and not interact,
        log_tools=log_tools_enabled,
        enable_mcp=bool(getattr(args, "enable_mcp", False)),
    )


def _validate_agent_options(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    options: OptimizeRunOptions,
) -> None:
    if getattr(args, "agent", None) == "openhands" and bool(getattr(args, "interact", False)):
        parser.error("OpenHands backend does not support --interact yet.")
    if options.enable_subagent and options.agent_name not in {"codex", "opencode", "claude"}:
        parser.error("--enable-subagent only supports --agent codex, opencode, or claude.")
