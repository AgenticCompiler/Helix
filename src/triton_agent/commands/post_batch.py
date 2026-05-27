from __future__ import annotations

import argparse
from pathlib import Path

from triton_agent.post_batch.collector import write_post_batch_state
from triton_agent.post_batch.render import render_post_batch_report_file


def handle_post_batch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    root = Path(args.input).expanduser().resolve()
    if not root.exists():
        parser.error(f"Input path does not exist: {root}")
    if not root.is_dir():
        parser.error(f"Input path is not a directory: {root}")

    state_path = write_post_batch_state(root)
    print(f"Post-batch state written to: {state_path}", flush=True)

    report_path = render_post_batch_report_file(state_path)
    print(f"Post-batch report written to: {report_path}", flush=True)

    return 0


__all__ = ["handle_post_batch"]
