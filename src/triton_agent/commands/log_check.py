from __future__ import annotations

import argparse
from pathlib import Path

from triton_agent.log_check.batch import run_log_check_batch
from triton_agent.log_check.log_check_launcher import run_log_check


def handle_log_check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    target_path = Path(args.input).expanduser().resolve()
    if not target_path.exists():
        parser.error(f"Input path does not exist: {target_path}")
    if not target_path.is_dir():
        parser.error(f"Input path is not a directory: {target_path}")
    return run_log_check(
        target_path=target_path,
        output_json=str(getattr(args, "check_result_file", "log_check_result.json")),
        agent_name=str(getattr(args, "agent", "codex")),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
    )


def handle_log_check_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")
    return run_log_check_batch(
        root,
        output_file=str(getattr(args, "check_result_file", "log_check_result.md")),
        summary_file=str(getattr(args, "summary_file", "log_check_summary.md")),
        agent_name=str(getattr(args, "agent", "codex")),
        verbose=bool(getattr(args, "verbose", False)),
        show_output=bool(getattr(args, "show_output", False)),
        max_concurrency=int(getattr(args, "max_concurrency", 1)),
    )


__all__ = ["handle_log_check", "handle_log_check_batch"]
