from __future__ import annotations

import argparse
from pathlib import Path

from triton_agent.batch_report.collector import write_batch_report_state
from triton_agent.batch_report.render import render_batch_report_file


def handle_batch_report(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    state_path = write_batch_report_state(root)
    print(f"Report-batch state written to: {state_path}", flush=True)

    report_path = render_batch_report_file(state_path)
    print(f"Report-batch written to: {report_path}", flush=True)

    return 0


__all__ = ["handle_batch_report"]
