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
    succeeded = sum(1 for item in ordered_results if item.status == "ok")
    failed = sum(1 for item in ordered_results if item.status == "failed")
    skipped = sum(1 for item in ordered_results if item.status == "skipped")
    for item in ordered_results:
        status = {
            "ok": "OK",
            "failed": "FAIL",
            "skipped": "SKIP",
        }[item.status]
        print(f"[{status}] {item.workspace.name}: {item.message}", file=stream)
    print(f"Summary: {succeeded} succeeded, {failed} failed, {skipped} skipped", file=stream)
    return 0 if failed == 0 and ordered_results else 1


def render_optimize_status_results(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
    output_format: str = "text",
) -> int:
    if output_format == "markdown":
        return render_optimize_status_markdown_table(results, stdout=stdout)
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


def render_optimize_status_markdown_table(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    rows = [
        item
        for item in sorted(results, key=_optimize_status_sort_key)
        if item.state != "no-session"
    ]
    print("| 名称 | Geomean speedup | Total speedup |", file=stream)
    print("| --- | --- | --- |", file=stream)
    for item in rows:
        print(
            "| "
            f"{item.workspace.name} | "
            f"{format_optimize_status_speedup_cell(item.geomean_speedup)} | "
            f"{format_optimize_status_speedup_cell(item.total_speedup)} |",
            file=stream,
        )
    return 0 if results else 1


def _optimize_status_sort_key(item: OptimizeStatusWorkspace) -> tuple[int, str]:
    return (_STATE_PRIORITY.get(item.state, len(_STATE_PRIORITY)), item.workspace.name)


def format_optimize_status_speedup_cell(value: float | None) -> str:
    if value is None:
        return "-"
    return format_optimize_status_speedup(value)


def _style(stream: TextIO, text: str, color: str) -> str:
    isatty = getattr(stream, "isatty", None)
    if callable(isatty) and isatty():
        return f"{color}{text}{_RESET}"
    return text
