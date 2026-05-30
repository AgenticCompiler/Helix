from __future__ import annotations

import argparse
from pathlib import Path

from triton_agent.trace_analyze import analyze_trace


def handle_trace_analyze(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    trace_path = Path(args.trace).expanduser().resolve()
    if not trace_path.is_file():
        parser.error(f"Trace file does not exist: {trace_path}")

    warnings = analyze_trace(trace_path=trace_path)

    for warning in warnings:
        print(f"trace-analyze: {warning}")

    if not warnings:
        output_dir = trace_path.parent
        print(f"Wrote {output_dir / 'summary.json'}")
    return 0 if not warnings else 1


__all__ = ["handle_trace_analyze"]
