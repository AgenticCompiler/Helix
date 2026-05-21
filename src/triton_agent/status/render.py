from __future__ import annotations

import sys
from typing import TextIO

from triton_agent.optimize.models import OptimizeStatusWorkspace

_RESET = "\033[0m"
_TITLE_COLOR = "\033[36m"
_BODY_COLOR = "\033[37m"
_WARNING_COLOR = "\033[90m"
_SUMMARY_COLOR = "\033[37m"


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


def render_optimize_status_results(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
    output_format: str = "text",
) -> int:
    if output_format == "markdown":
        return render_optimize_status_markdown_table(results, stdout=stdout)
    stream = stdout or sys.stdout
    ordered_results = sorted(results, key=_optimize_status_text_sort_key)
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
        print(_style(stream, f"  Best round: {item.best_round or 'unknown'}", _BODY_COLOR), file=stream)
        if item.logged_best is not None:
            print(_style(stream, f"  Logged best: {item.logged_best}", _BODY_COLOR), file=stream)
        if item.latest_verify_state is not None:
            print(_style(stream, f"  Latest verify: {item.latest_verify_state}", _BODY_COLOR), file=stream)
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
        for item in sorted(results, key=_optimize_status_markdown_sort_key)
        if item.state != "no-session"
    ]
    print(
        "| 名称 | Geomean speedup | Verified | "
        "Verified Geomean speedup | Notes |",
        file=stream,
    )
    print("| --- | --- | --- | --- | --- |", file=stream)
    for item in rows:
        print(
            "| "
            f"{item.workspace.name} | "
            f"{format_optimize_status_speedup_cell(item.geomean_speedup)} | "
            f"{format_optimize_status_verified_cell(item)} | "
            f"{format_optimize_status_verified_speedup_cell(item.verified_geomean_speedup)} | "
            f"{format_optimize_status_notes_cell(item)} |",
            file=stream,
        )
    return 0 if results else 1


def _optimize_status_text_sort_key(item: OptimizeStatusWorkspace) -> tuple[int, str]:
    return (0 if item.state == "no-session" else 1, item.workspace.name)


def _optimize_status_markdown_sort_key(item: OptimizeStatusWorkspace) -> str:
    return item.workspace.name


def format_optimize_status_speedup_cell(value: float | None) -> str:
    if value is None:
        return "-"
    return format_optimize_status_speedup(value)


def format_optimize_status_verified_speedup_cell(value: float | None) -> str:
    if value is None:
        return ""
    return format_optimize_status_speedup(value)


def format_optimize_status_verified_cell(item: OptimizeStatusWorkspace) -> str:
    return "Verified" if item.verified else "-"


def format_optimize_status_notes_cell(item: OptimizeStatusWorkspace) -> str:
    notes: list[str] = []
    if item.best_round is not None and item.logged_best is not None and item.best_round != item.logged_best:
        notes.append("best≠log")
    if any(not _is_best_round_mismatch_warning(warning) for warning in item.warnings):
        notes.append("warn")
    if not notes:
        return "-"
    return ",".join(notes)


def _is_best_round_mismatch_warning(warning: str) -> bool:
    return warning.startswith("numeric best round != logged best")


def _style(stream: TextIO, text: str, color: str) -> str:
    isatty = getattr(stream, "isatty", None)
    if callable(isatty) and isatty():
        return f"{color}{text}{_RESET}"
    return text
