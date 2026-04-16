#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OP_NAME_COLUMNS = ("OP Type", "Op Type", "Op Name", "Kernel Name")
DURATION_COLUMNS = ("Task Duration(us)", "Task Duration", "Duration(us)")
STATISTIC_REQUIRED_COLUMNS = (
    "OP Type",
    "Core Type",
    "Count",
    "Total Time(us)",
    "Min Time(us)",
    "Avg Time(us)",
    "Max Time(us)",
    "Ratio(%)",
)


def _format_number(value: float) -> str:
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        padded = row + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _parse_float(value: str) -> float:
    return float(value.strip())


def resolve_profile_dir(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        raise FileNotFoundError(f"Expected a directory, got file: {candidate}")
    if candidate.name.startswith("PROF_") and (candidate / "mindstudio_profiler_output").is_dir():
        return candidate
    if candidate.name == "mindstudio_profiler_output" and candidate.is_dir():
        return candidate.parent
    if (candidate / "mindstudio_profiler_output").is_dir():
        return candidate

    matches = [
        match
        for match in candidate.rglob("PROF_*")
        if match.is_dir() and (match / "mindstudio_profiler_output").is_dir()
    ]
    if not matches:
        raise FileNotFoundError(f"No PROF_* directory found under {candidate}")
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _find_newest_csv(output_dir: Path, prefix: str) -> Path | None:
    matches = sorted(output_dir.glob(f"{prefix}_*.csv"))
    if not matches:
        return None
    return max(matches, key=lambda item: item.stat().st_mtime_ns)


def _load_statistic_rows(csv_path: Path) -> list[dict[str, float | str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in STATISTIC_REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path}: {', '.join(missing)}"
            )

        rows: list[dict[str, float | str]] = []
        for row in reader:
            rows.append(
                {
                    "op_type": row["OP Type"].strip(),
                    "core_type": row["Core Type"].strip(),
                    "count": _parse_float(row["Count"]),
                    "total_time_us": _parse_float(row["Total Time(us)"]),
                    "min_time_us": _parse_float(row["Min Time(us)"]),
                    "avg_time_us": _parse_float(row["Avg Time(us)"]),
                    "max_time_us": _parse_float(row["Max Time(us)"]),
                    "ratio_percent": _parse_float(row["Ratio(%)"]),
                }
            )

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def _select_target_row(
    rows: list[dict[str, float | str]], target_op: str | None
) -> tuple[dict[str, float | str], bool]:
    if target_op is not None:
        for row in rows:
            if str(row["op_type"]) == target_op:
                return row, False
        raise ValueError(f"Target operator not found in op_statistic: {target_op}")
    return max(rows, key=lambda item: float(item["total_time_us"])), True


def _find_summary_columns(fieldnames: list[str]) -> tuple[str | None, str | None]:
    op_name_column = next((name for name in OP_NAME_COLUMNS if name in fieldnames), None)
    duration_column = next((name for name in DURATION_COLUMNS if name in fieldnames), None)
    return op_name_column, duration_column


def _summarize_op_summary(csv_path: Path | None, target_op: str) -> dict[str, float | int | str | None]:
    if csv_path is None:
        return {
            "path": None,
            "matched_rows": 0,
            "total_duration_us": None,
            "avg_duration_us": None,
            "min_duration_us": None,
            "max_duration_us": None,
            "note": "No op_summary CSV found.",
        }

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        op_name_column, duration_column = _find_summary_columns(fieldnames)
        if op_name_column is None:
            return {
                "path": str(csv_path),
                "matched_rows": 0,
                "total_duration_us": None,
                "avg_duration_us": None,
                "min_duration_us": None,
                "max_duration_us": None,
                "note": "No operator-identifying column was recognized in op_summary.",
            }

        matched_rows = 0
        duration_rows = 0
        total_duration = 0.0
        min_duration: float | None = None
        max_duration: float | None = None

        for row in reader:
            if row.get(op_name_column, "").strip() != target_op:
                continue
            matched_rows += 1
            if duration_column is None:
                continue
            duration_value = row.get(duration_column, "").strip()
            if not duration_value:
                continue
            duration = _parse_float(duration_value)
            duration_rows += 1
            total_duration += duration
            min_duration = duration if min_duration is None else min(min_duration, duration)
            max_duration = duration if max_duration is None else max(max_duration, duration)

    average_duration = None
    if duration_rows > 0 and duration_column is not None:
        average_duration = total_duration / duration_rows

    note = None
    if duration_column is None:
        note = "No duration column was recognized in op_summary."

    return {
        "path": str(csv_path),
        "matched_rows": matched_rows,
        "total_duration_us": total_duration if average_duration is not None else None,
        "avg_duration_us": average_duration,
        "min_duration_us": min_duration,
        "max_duration_us": max_duration,
        "note": note,
    }


def build_profile_report(
    profile_path: str | Path,
    *,
    target_op: str | None = None,
    top_count: int = 5,
) -> str:
    profile_dir = resolve_profile_dir(profile_path)
    output_dir = profile_dir / "mindstudio_profiler_output"
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Missing mindstudio_profiler_output directory in {profile_dir}")

    op_statistic_csv = _find_newest_csv(output_dir, "op_statistic")
    if op_statistic_csv is None:
        raise FileNotFoundError(f"No op_statistic_*.csv found in {output_dir}")
    op_summary_csv = _find_newest_csv(output_dir, "op_summary")

    statistic_rows = _load_statistic_rows(op_statistic_csv)
    target_row, inferred = _select_target_row(statistic_rows, target_op)
    summary_stats = _summarize_op_summary(op_summary_csv, str(target_row["op_type"]))

    top_rows = sorted(
        statistic_rows,
        key=lambda item: float(item["total_time_us"]),
        reverse=True,
    )[:top_count]
    table_rows = [
        [
            str(row["op_type"]),
            str(row["core_type"]),
            _format_number(float(row["count"])),
            _format_number(float(row["total_time_us"])),
            _format_number(float(row["avg_time_us"])),
            _format_number(float(row["ratio_percent"])),
        ]
        for row in top_rows
    ]

    lines = [
        "# Ascend NPU Operator Profile Summary",
        "",
        f"- Profile directory: `{profile_dir}`",
        f"- `op_statistic` file: `{op_statistic_csv.name}`",
        f"- `op_summary` file: `{op_summary_csv.name if op_summary_csv is not None else 'not found'}`",
        f"- Target operator: `{target_row['op_type']}`",
        (
            "- Selection: inferred from the hottest `op_statistic` row by `Total Time(us)`"
            if inferred
            else "- Selection: matched the explicit `--target-op` value"
        ),
        "",
        "## Operator timing",
        "",
        f"- Core type: `{target_row['core_type']}`",
        f"- Invocation count: `{_format_number(float(target_row['count']))}`",
        f"- Total time: `{_format_number(float(target_row['total_time_us']))} us`",
        f"- Average time: `{_format_number(float(target_row['avg_time_us']))} us`",
        f"- Min time: `{_format_number(float(target_row['min_time_us']))} us`",
        f"- Max time: `{_format_number(float(target_row['max_time_us']))} us`",
        f"- Runtime ratio: `{_format_number(float(target_row['ratio_percent']))}%`",
        "",
        "## op_summary cross-check",
        "",
        f"- Matched op_summary rows: `{summary_stats['matched_rows']}`",
    ]

    if summary_stats["avg_duration_us"] is not None:
        lines.extend(
            [
                f"- Summed task duration: `{_format_number(float(summary_stats['total_duration_us']))} us`",
                f"- Average task duration: `{_format_number(float(summary_stats['avg_duration_us']))} us`",
                f"- Min task duration: `{_format_number(float(summary_stats['min_duration_us']))} us`",
                f"- Max task duration: `{_format_number(float(summary_stats['max_duration_us']))} us`",
            ]
        )
    if summary_stats["note"]:
        lines.append(f"- Note: {summary_stats['note']}")

    lines.extend(
        [
            "",
            "## Top operators by total time",
            "",
            _markdown_table(
                ["OP Type", "Core Type", "Count", "Total Time(us)", "Avg Time(us)", "Ratio(%)"],
                table_rows,
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Ascend msprof CSV output.")
    parser.add_argument("profile_path", help="A PROF_* directory or a parent directory containing one.")
    parser.add_argument("--target-op", help="Operator name to summarize from op_statistic/op_summary.")
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top operators to include in the hotspot table.",
    )
    args = parser.parse_args()

    print(build_profile_report(args.profile_path, target_op=args.target_op, top_count=args.top), end="")


if __name__ == "__main__":
    main()
