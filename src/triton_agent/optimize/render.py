from __future__ import annotations

import sys
from typing import TextIO

from triton_agent.optimize.models import BatchOptimizeResult, OptimizeStatusWorkspace


def format_optimize_status_float(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6f}"


def format_optimize_status_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100:+.1f}%"


def render_batch_optimize_results(
    results: list[BatchOptimizeResult],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    ordered_results = sorted(results, key=lambda item: item.workspace.name)
    succeeded = sum(1 for item in ordered_results if item.succeeded)
    failed = len(ordered_results) - succeeded
    for item in ordered_results:
        status = "OK" if item.succeeded else "FAIL"
        print(f"[{status}] {item.workspace.name}: {item.message}", file=stream)
    print(f"Summary: {succeeded} succeeded, {failed} failed", file=stream)
    return 0 if failed == 0 and ordered_results else 1


def render_optimize_status_results(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    ordered_results = sorted(results, key=lambda item: item.workspace.name)
    ok_count = sum(1 for item in ordered_results if item.state == "ok")
    warning_count = sum(1 for item in ordered_results if item.state == "warning")
    no_session_count = sum(1 for item in ordered_results if item.state == "no-session")

    for item in ordered_results:
        status = {
            "ok": "OK",
            "warning": "WARN",
            "no-session": "NO-SESSION",
        }[item.state]
        print(f"[{status}] {item.workspace.name}", file=stream)
        if item.state == "no-session":
            continue
        print(f"  Baseline mean: {format_optimize_status_float(item.baseline_mean)}", file=stream)
        print(f"  Best mean: {format_optimize_status_float(item.best_mean)}", file=stream)
        print(f"  Avg improvement: {format_optimize_status_percent(item.avg_improvement)}", file=stream)
        print(f"  Best round: {item.best_round or 'unknown'}", file=stream)
        if item.logged_best is not None:
            print(f"  Logged best: {item.logged_best}", file=stream)
        for warning in item.warnings:
            print(f"  Warning: {warning}", file=stream)

    print(
        "Summary: "
        f"{ok_count} ok, {warning_count} warning, {no_session_count} no-session",
        file=stream,
    )
    return 0 if ordered_results else 1
