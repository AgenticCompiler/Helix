from __future__ import annotations

import json
import sys
from typing import TextIO

from triton_agent.status.models import OptimizeStatusWorkspace

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
    view: str = "best",
) -> int:
    if view == "trend":
        return render_optimize_status_trend_results(results, stdout=stdout, output_format=output_format)
    if view != "best":
        raise ValueError(f"unsupported status view: {view}")
    if output_format == "json":
        return render_optimize_status_json(results, stdout=stdout)
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


def render_optimize_status_json(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    operators = [
        {
            "name": item.workspace.name,
            "state": item.state,
            "avg_improvement": item.avg_improvement,
            "geomean_speedup": item.geomean_speedup,
            "best_round": item.best_round,
            "logged_best": item.logged_best,
            "verified": item.verified,
            "verified_geomean_speedup": item.verified_geomean_speedup,
            "warnings": list(item.warnings),
        }
        for item in sorted(results, key=_optimize_status_json_sort_key)
    ]
    json.dump({"operators": operators}, stream, ensure_ascii=False, indent=2)
    print(file=stream)
    return 0 if results else 1


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


def render_optimize_status_trend_results(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
    output_format: str = "text",
) -> int:
    if output_format == "json":
        return render_optimize_status_trend_json(results, stdout=stdout)
    if output_format == "markdown":
        return render_optimize_status_trend_markdown_table(results, stdout=stdout)
    return render_optimize_status_trend_text_table(results, stdout=stdout)


def render_optimize_status_trend_text_table(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    rows = _trend_rows(results)
    round_names = _trend_round_names(rows)
    table_rows = [
        [item.workspace.name, *[_trend_speedup_cell(item, round_name) for round_name in round_names]]
        for item in rows
    ]
    headers = ["Name", *round_names]
    widths = [
        max(len(row[index]) for row in [headers, *table_rows])
        for index in range(len(headers))
    ]
    print(_format_text_table_row(headers, widths), file=stream)
    for row in table_rows:
        print(_format_text_table_row(row, widths), file=stream)
    return 0 if results else 1


def render_optimize_status_trend_markdown_table(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    rows = _trend_rows(results)
    round_names = _trend_round_names(rows)
    headers = ["Name", *round_names]
    print("| " + " | ".join(headers) + " |", file=stream)
    print("| " + " | ".join("---" for _ in headers) + " |", file=stream)
    for item in rows:
        cells = [item.workspace.name, *[_trend_speedup_cell(item, round_name) for round_name in round_names]]
        print("| " + " | ".join(cells) + " |", file=stream)
    return 0 if results else 1


def render_optimize_status_trend_json(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    rows = _trend_rows(results)
    round_names = _trend_round_names(rows)
    operators: list[dict[str, object]] = []
    for item in rows:
        speedups_by_round = {round.round_name: round.geomean_speedup for round in item.rounds}
        operators.append(
            {
                "name": item.workspace.name,
                "round_speedups": {
                    round_name: speedups_by_round.get(round_name)
                    for round_name in round_names
                },
            }
        )
    json.dump({"operators": operators}, stream, ensure_ascii=False, indent=2)
    print(file=stream)
    return 0 if results else 1


def _optimize_status_text_sort_key(item: OptimizeStatusWorkspace) -> tuple[int, str]:
    return (0 if item.state == "no-session" else 1, item.workspace.name)


def _optimize_status_markdown_sort_key(item: OptimizeStatusWorkspace) -> str:
    return item.workspace.name


def _optimize_status_json_sort_key(item: OptimizeStatusWorkspace) -> tuple[int, str]:
    return _optimize_status_text_sort_key(item)


def _trend_rows(results: list[OptimizeStatusWorkspace]) -> list[OptimizeStatusWorkspace]:
    return sorted(
        (item for item in results if item.state != "no-session"),
        key=lambda item: item.workspace.name,
    )


def _trend_round_names(rows: list[OptimizeStatusWorkspace]) -> list[str]:
    return sorted(
        {round.round_name for item in rows for round in item.rounds},
        key=_round_sort_key,
    )


def _round_sort_key(round_name: str) -> tuple[int, str]:
    prefix = "round-"
    if round_name.startswith(prefix):
        suffix = round_name.removeprefix(prefix)
        if suffix.isdecimal():
            return (int(suffix), round_name)
    return (10**9, round_name)


def _trend_speedup_cell(item: OptimizeStatusWorkspace, round_name: str) -> str:
    for round in item.rounds:
        if round.round_name == round_name:
            return format_optimize_status_speedup(round.geomean_speedup)
    return "-"


def _format_text_table_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))


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
