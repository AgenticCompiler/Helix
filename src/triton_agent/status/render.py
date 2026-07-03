from __future__ import annotations

import html
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
    return render_optimize_status_best_results(results, stdout=stdout, output_format=output_format)


def render_optimize_status_best_results(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
    output_format: str = "text",
) -> int:
    if output_format == "html":
        raise ValueError("HTML format only supports --view trend")
    if output_format == "json":
        return render_optimize_status_json(results, stdout=stdout)
    if output_format == "markdown":
        return render_optimize_status_markdown_table(results, stdout=stdout)
    return render_optimize_status_text(results, stdout=stdout)


def render_optimize_status_text(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
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
    if output_format == "html":
        return render_optimize_status_trend_html(results, stdout=stdout)
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
        operators.append(
            {
                "name": item.workspace.name,
                "round_speedups": _trend_round_speedups(item, round_names),
            }
        )
    json.dump({"operators": operators}, stream, ensure_ascii=False, indent=2)
    print(file=stream)
    return 0 if results else 1


def render_optimize_status_trend_html(
    results: list[OptimizeStatusWorkspace],
    stdout: TextIO | None = None,
) -> int:
    stream = stdout or sys.stdout
    print(_build_optimize_status_trend_html(results), file=stream)
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


def _build_optimize_status_trend_html(
    results: list[OptimizeStatusWorkspace],
) -> str:
    rows = _trend_rows(results)
    round_names = _trend_round_names(rows)
    comparable_rows = [
        item
        for item in rows
        if any(speedup is not None for speedup in _trend_round_speedups(item, round_names).values())
    ]

    cards: list[str] = []
    for item in comparable_rows:
        last_speedup, best_speedup, first_speedup = _trend_html_stats(item, round_names)
        del last_speedup
        name = html.escape(item.workspace.name)
        max_color = "#d62728" if best_speedup is not None and best_speedup < 1.0 else "#2077b4"
        summary = (
            f'<span style="color:{max_color}">max {best_speedup:.2f}x</span> · start {first_speedup:.2f}x'
            if best_speedup is not None and first_speedup is not None
            else "no data"
        )
        cards.append(
            f'<div class="card"><div class="title">{name}</div>'
            f'<div class="sub">{summary}</div>{_trend_chart_svg(item, round_names)}</div>'
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Operator Speedup Trends</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; color: #222; background: #fafafa; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color: #666; font-size: 13px; margin-bottom: 18px; }}
  .grid {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }}
  .card {{ background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 10px 12px; }}
  .title {{ font-weight: 600; font-size: 13px; }}
  .sub {{ font-size: 11px; color: #666; margin-bottom: 4px; }}
  @media print {{
    @page {{ size: A4 landscape; margin: 10mm; }}
    body {{ margin: 0; background: #fff; }}
    .grid {{ gap: 8px; }}
    .card {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<h1>Operator Speedup Trends</h1>
<div class="meta">{len(comparable_rows)} operators · dashed line = 1.00x baseline · hover a point for its value</div>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>
"""


def _trend_chart_svg(
    item: OptimizeStatusWorkspace,
    round_names: list[str],
) -> str:
    width = 440
    height = 230
    margin_left = 46
    margin_right = 12
    margin_top = 26
    margin_bottom = 30
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    items = sorted(_trend_round_speedups(item, round_names).items(), key=lambda entry: _round_sort_key(entry[0]))
    rounds = [_round_sort_key(key)[0] for key, _value in items]
    values = [value for _key, value in items]

    present_values = [value for value in values if value is not None]
    y_low = min([*present_values, 1.0])
    y_high = max([*present_values, 1.0])
    if y_high == y_low:
        y_high += 1.0
    padding = (y_high - y_low) * 0.08
    y_low -= padding
    y_high += padding

    x_low = min(rounds)
    x_high = max(rounds)
    if x_high == x_low:
        x_high += 1

    def x_pixel(round_number: int) -> float:
        return margin_left + (round_number - x_low) / (x_high - x_low) * plot_width

    def y_pixel(value: float) -> float:
        return margin_top + (y_high - value) / (y_high - y_low) * plot_height

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" preserveAspectRatio="xMidYMid meet">'
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = y_low + fraction * (y_high - y_low)
        y_axis = y_pixel(value)
        parts.append(
            f'<line x1="{margin_left}" y1="{y_axis:.1f}" x2="{width - margin_right}" y2="{y_axis:.1f}" stroke="#eee" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 5}" y="{y_axis + 3:.1f}" font-size="9" text-anchor="end" fill="#888">{value:.2f}</text>'
        )

    baseline_y = y_pixel(1.0)
    parts.append(
        f'<line x1="{margin_left}" y1="{baseline_y:.1f}" x2="{width - margin_right}" y2="{baseline_y:.1f}" '
        f'stroke="#f0a" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>'
    )

    if present_values:
        best_speedup = max(present_values)
        best_y = y_pixel(best_speedup)
        parts.append(
            f'<line x1="{margin_left}" y1="{best_y:.1f}" x2="{width - margin_right}" y2="{best_y:.1f}" '
            f'stroke="#2077b4" stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>'
        )
        parts.append(
            f'<text x="{width - margin_right}" y="{best_y - 3:.1f}" font-size="9" text-anchor="end" fill="#2077b4">max {best_speedup:.2f}x</text>'
        )

    x_step = max(1, (x_high - x_low) // 6)
    for round_number in range(x_low, x_high + 1, x_step):
        parts.append(
            f'<text x="{x_pixel(round_number):.1f}" y="{height - 10}" font-size="9" text-anchor="middle" fill="#888">{round_number}</text>'
        )

    segments: list[list[str]] = []
    current_segment: list[str] = []
    for round_number, value in zip(rounds, values):
        if value is None:
            if len(current_segment) > 1:
                segments.append(current_segment)
            current_segment = []
            continue
        current_segment.append(f"{x_pixel(round_number):.1f},{y_pixel(value):.1f}")
    if len(current_segment) > 1:
        segments.append(current_segment)

    for segment in segments:
        parts.append(
            f'<polyline points="{" ".join(segment)}" fill="none" stroke="#1b9e4b" stroke-width="1.8"/>'
        )

    for round_number, value in zip(rounds, values):
        if value is None:
            continue
        parts.append(
            f'<circle cx="{x_pixel(round_number):.1f}" cy="{y_pixel(value):.1f}" r="2" fill="#1b9e4b"><title>round-{round_number}: {value:.3f}x</title></circle>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _trend_html_stats(
    item: OptimizeStatusWorkspace,
    round_names: list[str],
) -> tuple[float | None, float | None, float | None]:
    items = sorted(_trend_round_speedups(item, round_names).items(), key=lambda entry: _round_sort_key(entry[0]))
    present_values = [value for _key, value in items if value is not None]
    if not present_values:
        return None, None, None
    return present_values[-1], max(present_values), present_values[0]


def _trend_round_speedups(
    item: OptimizeStatusWorkspace,
    round_names: list[str],
) -> dict[str, float | None]:
    speedups_by_round = {round.round_name: round.geomean_speedup for round in item.rounds}
    speedups: dict[str, float | None] = {}
    for round_name in round_names:
        speedups[round_name] = speedups_by_round.get(round_name)
    return speedups


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
