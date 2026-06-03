from __future__ import annotations

import argparse
import sys
from pathlib import Path

from triton_agent.pattern_validation_loop.scaffold_verify import run_pattern_validation_verify


def handle_pattern_validation_verify(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> int:
    batch_root = Path(args.input).expanduser().resolve()
    if not batch_root.exists():
        parser.error(f"Input path does not exist: {batch_root}")
    if not batch_root.is_dir():
        parser.error(f"Input path is not a directory: {batch_root}")
    return run_pattern_validation_verify(
        batch_root,
        json_output=bool(getattr(args, "json", False)),
        stream=sys.stdout,
    )


__all__ = ["handle_pattern_validation_verify"]
