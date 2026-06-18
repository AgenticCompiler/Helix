from __future__ import annotations

import argparse
from pathlib import Path

from triton_agent.otel_trace import trace_summary_path
from triton_agent.trace_analyze import analyze_trace


def handle_trace_analyze(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    trace_path = Path(args.trace).expanduser().resolve()
    if not trace_path.is_file():
        parser.error(f"Trace file does not exist: {trace_path}")

    warnings = analyze_trace(trace_path=trace_path)

    for warning in warnings:
        print(f"trace-analyze: {warning}")

    if not warnings:
        print(f"Wrote {trace_summary_path(trace_path)}")
    return 0 if not warnings else 1
