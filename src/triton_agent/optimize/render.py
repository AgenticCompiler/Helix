from __future__ import annotations

import sys
from typing import TextIO

from triton_agent.optimize.models import BatchOptimizeResult, OptimizeStatusWorkspace

_RESET = "\033[0m"
_TITLE_COLOR = "\033[36m"
_BODY_COLOR = "\033[37m"
_WARNING_COLOR = "\033[90m"
_SUMMARY_COLOR = "\033[37m"
_STATE_PRIORITY = {
    "no-session": 0,
    "warning": 1,
    "ok": 2,
}


def format_optimize_status_float(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6f}"


def format_optimize_status_percent(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100:+.1f}%"


def format_optimize_status_speedup(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}x"


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
    ordered_results = sorted(results, key=_optimize_status_sort_key)
    ok_count = sum(1 for item in ordered_results if item.state == "ok")
    warning_count = sum(1 for item in ordered_results if item.state == "warning")
    no_session_count = sum(1 for item in ordered_results if item.state == "no-session")

    for item in ordered_results:
        status = {
            "ok": "OK",
            "warning": "WARN",
            "no-session": "NO-SESSION",
        }[item.state]
        print(_style(stream, f"[{status}] {item.workspace.name}", _TITLE_COLOR), file=stream)
        if item.state == "no-session":
            continue
        print(
            _style(stream, f"  Baseline mean: {format_optimize_status_float(item.baseline_mean)}", _BODY_COLOR),
            file=stream,
        )
        print(
            _style(stream, f"  Best mean: {format_optimize_status_float(item.best_mean)}", _BODY_COLOR),
            file=stream,
        )
        print(
            _style(
                stream,
                f"  Avg improvement: {format_optimize_status_percent(item.avg_improvement)}",
                _BODY_COLOR,
            ),
            file=stream,
        )
        print(
            _style(
                stream,
                f"  Geomean speedup: {format_optimize_status_speedup(item.geomean_speedup)}",
                _BODY_COLOR,
            ),
            file=stream,
        )
        print(
            _style(
                stream,
                f"  Total speedup: {format_optimize_status_speedup(item.total_speedup)}",
                _BODY_COLOR,
            ),
            file=stream,
        )
        print(_style(stream, f"  Best round: {item.best_round or 'unknown'}", _BODY_COLOR), file=stream)
        if item.logged_best is not None:
            print(_style(stream, f"  Logged best: {item.logged_best}", _BODY_COLOR), file=stream)
        for warning in item.warnings:
            print(_style(stream, f"  Warning: {warning}", _WARNING_COLOR), file=stream)

    print(
        _style(
            stream,
            "Summary: "
            f"{ok_count} ok, {warning_count} warning, {no_session_count} no-session",
            _SUMMARY_COLOR,
        ),
        file=stream,
    )
    return 0 if ordered_results else 1


def _optimize_status_sort_key(item: OptimizeStatusWorkspace) -> tuple[int, str]:
    return (_STATE_PRIORITY.get(item.state, len(_STATE_PRIORITY)), item.workspace.name)


def _style(stream: TextIO, text: str, color: str) -> str:
    isatty = getattr(stream, "isatty", None)
    if callable(isatty) and isatty():
        return f"{color}{text}{_RESET}"
    return text
