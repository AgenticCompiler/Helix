#!/usr/bin/env python3

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path
from typing import NamedTuple, TypedDict


KEYWORDS: tuple[str, ...] = (
    "alloc",
    "copy",
    "matmul",
    "for",
    "if",
    "vector",
    "load",
    "store",
    "dma",
    "wait",
    "set_flag",
    "barrier",
)


class StageInfo(NamedTuple):
    path: Path
    relative_path: str
    stem: str
    size_bytes: int
    line_count: int


_StageChange = TypedDict(
    "_StageChange",
    {
        "from": StageInfo,
        "to": StageInfo,
        "line_delta": int,
        "size_delta": int,
        "keyword_deltas": dict[str, int],
        "score": int,
    },
)


class _StageSignalSummary(TypedDict):
    stage: str
    path: str
    line_count: int
    size_bytes: int
    vector_ops: int
    transfer_ops: int
    sync_ops: int
    alloc_ops: int
    interesting_score: int


class _SuspiciousTransition(TypedDict):
    from_stage: str
    to_stage: str
    signal_score: int
    vector_delta: int
    transfer_delta: int
    sync_delta: int
    line_delta: int
    size_delta: int


def resolve_stages_dir(ir_dir: str | Path) -> Path:
    ir_path = Path(ir_dir).expanduser().resolve()
    if not ir_path.exists():
        raise FileNotFoundError(f"IR directory does not exist: {ir_path}")
    if not ir_path.is_dir():
        raise FileNotFoundError(f"IR path is not a directory: {ir_path}")
    stages_dir = ir_path / "bishengir_stages"
    if not stages_dir.is_dir():
        raise FileNotFoundError(f"IR directory is missing bishengir_stages/: {ir_path}")
    return stages_dir


def discover_stages(stages_dir: str | Path) -> list[StageInfo]:
    root = Path(stages_dir).expanduser().resolve()
    stage_files = sorted(root.rglob("*.mlir"), key=lambda path: _stage_sort_key(root, path))
    if not stage_files:
        raise FileNotFoundError(f"No .mlir stages found under {root}")
    return [_build_stage_info(root, path) for path in stage_files]


def resolve_stage_selector(stages_dir: str | Path, selector: str) -> Path:
    stage_infos = discover_stages(stages_dir)
    normalized = selector.strip().removesuffix(".mlir")
    exact_rel_matches = [
        info.path
        for info in stage_infos
        if info.relative_path.removesuffix(".mlir") == normalized
    ]
    if len(exact_rel_matches) == 1:
        return exact_rel_matches[0]

    exact_stem_matches = [info.path for info in stage_infos if info.stem == normalized]
    if len(exact_stem_matches) == 1:
        return exact_stem_matches[0]

    substring_matches = [
        info.path
        for info in stage_infos
        if normalized in info.stem or normalized in info.relative_path.removesuffix(".mlir")
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if not substring_matches:
        raise FileNotFoundError(f"No stage matches selector: {selector}")
    raise ValueError(
        "Stage selector is ambiguous: "
        f"{selector} -> {', '.join(path.name for path in substring_matches[:5])}"
    )


def list_stages_text(
    ir_dir: str | Path,
    *,
    grep: str | None = None,
    limit: int | None = None,
    sort_by: str = "order",
) -> str:
    stage_infos = discover_stages(resolve_stages_dir(ir_dir))
    if grep:
        pattern = re.compile(grep)
        stage_infos = [
            info
            for info in stage_infos
            if pattern.search(info.stem) or pattern.search(info.relative_path)
        ]
    stage_infos = _sort_stage_infos(stage_infos, sort_by=sort_by)
    if limit is not None:
        stage_infos = stage_infos[:limit]
    if not stage_infos:
        return "No stages matched.\n"

    lines = ["Stages:"]
    for info in stage_infos:
        prefix = f"{info.stem:<36} {_format_size(info.size_bytes):>6}"
        if sort_by == "interesting":
            score = _interesting_score(_keyword_counts(info.path.read_text(encoding="utf-8")))
            prefix = f"{prefix}  score={score}"
        lines.append(f"{prefix}  {info.relative_path}")
    return "\n".join(lines) + "\n"


def stage_summary_text(ir_dir: str | Path, selector: str) -> str:
    stages_dir = resolve_stages_dir(ir_dir)
    stage_path = resolve_stage_selector(stages_dir, selector)
    info = _build_stage_info(stages_dir, stage_path)
    text = stage_path.read_text(encoding="utf-8")
    counts = _keyword_counts(text)
    highlights = _stage_highlights(text)

    lines = [
        f"Stage: {info.stem}",
        f"Path: {info.relative_path}",
        f"Size: {_format_size(info.size_bytes)}",
        f"Lines: {info.line_count}",
        "",
        "Keyword counts:",
    ]
    for keyword in KEYWORDS:
        lines.append(f"- {keyword}: {counts[keyword]}")
    lines.extend(
        [
            "",
            "Highlights:",
        ]
    )
    if highlights:
        lines.extend(f"- {line}" for line in highlights)
    else:
        lines.append("- No highlight lines matched the default heuristics.")
    return "\n".join(lines) + "\n"


def diff_stages_text(
    ir_dir: str | Path,
    *,
    from_selector: str,
    to_selector: str,
    context: int = 2,
) -> str:
    stages_dir = resolve_stages_dir(ir_dir)
    from_path = resolve_stage_selector(stages_dir, from_selector)
    to_path = resolve_stage_selector(stages_dir, to_selector)
    from_info = _build_stage_info(stages_dir, from_path)
    to_info = _build_stage_info(stages_dir, to_path)
    from_text = from_path.read_text(encoding="utf-8")
    to_text = to_path.read_text(encoding="utf-8")
    from_counts = _keyword_counts(from_text)
    to_counts = _keyword_counts(to_text)

    diff_lines = list(
        difflib.unified_diff(
            from_text.splitlines(),
            to_text.splitlines(),
            fromfile=from_info.relative_path,
            tofile=to_info.relative_path,
            lineterm="",
            n=context,
        )
    )

    lines = [
        f"From: {from_info.stem}",
        f"To: {to_info.stem}",
        f"Line delta: {to_info.line_count - from_info.line_count:+d}",
        f"Size delta: {to_info.size_bytes - from_info.size_bytes:+d} bytes",
        "",
        "Keyword deltas:",
    ]
    for keyword in KEYWORDS:
        delta = to_counts[keyword] - from_counts[keyword]
        lines.append(f"- {keyword}: {delta:+d} ({from_counts[keyword]} -> {to_counts[keyword]})")
    lines.extend(["", "Unified diff:"])
    if diff_lines:
        lines.extend(diff_lines)
    else:
        lines.append("(no textual differences)")
    return "\n".join(lines) + "\n"


def find_changes_text(
    ir_dir: str | Path,
    *,
    limit: int | None = None,
    sort_by: str = "score",
) -> str:
    stage_infos = discover_stages(resolve_stages_dir(ir_dir))
    changes = _adjacent_stage_changes(stage_infos)
    changes = _sort_stage_changes(changes, sort_by=sort_by)
    if limit is not None:
        changes = changes[:limit]
    if not changes:
        return "No adjacent stage changes found.\n"

    lines = ["Adjacent stage changes:"]
    for change in changes:
        lines.append(
            f"{change['from'].stem} -> {change['to'].stem}  "
            f"score={change['score']}  lines={change['line_delta']:+d}  size={change['size_delta']:+d}B"
        )
        lines.append(f"  keyword deltas: {_format_keyword_delta_summary(change['keyword_deltas'])}")
    return "\n".join(lines) + "\n"


def performance_signals_text(
    ir_dir: str | Path,
    *,
    limit: int | None = None,
    output_format: str = "text",
) -> str:
    payload = build_performance_signals_payload(ir_dir, limit=limit)
    if output_format == "json":
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_format != "text":
        raise ValueError(f"Unsupported output format: {output_format}")

    lines = [
        "Performance signals:",
        "",
        "Vector-heavy stages:",
    ]
    lines.extend(_render_signal_stage_lines(payload["vector_heavy_stages"]))
    lines.extend(
        [
            "",
            "Transfer-heavy stages:",
        ]
    )
    lines.extend(_render_signal_stage_lines(payload["transfer_heavy_stages"]))
    lines.extend(
        [
            "",
            "Sync-heavy stages:",
        ]
    )
    lines.extend(_render_signal_stage_lines(payload["sync_heavy_stages"]))
    lines.extend(
        [
            "",
            "Suspicious transitions:",
        ]
    )
    transitions = payload["suspicious_transitions"]
    if transitions:
        for change in transitions:
            lines.append(
                f"- {change['from_stage']} -> {change['to_stage']} "
                f"(score={change['signal_score']}, vector_delta={change['vector_delta']:+d}, "
                f"transfer_delta={change['transfer_delta']:+d}, sync_delta={change['sync_delta']:+d})"
            )
    else:
        lines.append("- No suspicious transitions matched the default heuristics.")
    return "\n".join(lines) + "\n"


class _PerformanceSignalsPayload(TypedDict):
    stage_summaries: list[_StageSignalSummary]
    vector_heavy_stages: list[_StageSignalSummary]
    transfer_heavy_stages: list[_StageSignalSummary]
    sync_heavy_stages: list[_StageSignalSummary]
    suspicious_transitions: list[_SuspiciousTransition]


def build_performance_signals_payload(
    ir_dir: str | Path,
    *,
    limit: int | None = None,
) -> _PerformanceSignalsPayload:
    stage_infos = discover_stages(resolve_stages_dir(ir_dir))
    stage_summaries = [_stage_signal_summary(info) for info in stage_infos]
    suspicious_transitions = _suspicious_transitions(stage_infos)
    if limit is not None:
        suspicious_transitions = suspicious_transitions[:limit]

    sorted_vector = sorted(
        [summary for summary in stage_summaries if summary["vector_ops"] > 0],
        key=lambda item: (-item["vector_ops"], str(item["stage"])),
    )
    sorted_transfer = sorted(
        [summary for summary in stage_summaries if summary["transfer_ops"] > 0],
        key=lambda item: (-item["transfer_ops"], str(item["stage"])),
    )
    sorted_sync = sorted(
        [summary for summary in stage_summaries if summary["sync_ops"] > 0],
        key=lambda item: (-item["sync_ops"], str(item["stage"])),
    )

    if limit is not None:
        stage_summaries = stage_summaries[:]
        sorted_vector = sorted_vector[:limit]
        sorted_transfer = sorted_transfer[:limit]
        sorted_sync = sorted_sync[:limit]

    return {
        "stage_summaries": stage_summaries,
        "vector_heavy_stages": sorted_vector,
        "transfer_heavy_stages": sorted_transfer,
        "sync_heavy_stages": sorted_sync,
        "suspicious_transitions": suspicious_transitions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect archived Triton Ascend IR stages.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-stages")
    list_parser.add_argument("--ir-dir", required=True)
    list_parser.add_argument("--grep")
    list_parser.add_argument("--limit", type=int)
    list_parser.add_argument(
        "--sort-by",
        choices=["order", "size", "lines", "interesting"],
        default="order",
    )

    summary_parser = subparsers.add_parser("stage-summary")
    summary_parser.add_argument("--ir-dir", required=True)
    summary_parser.add_argument("--stage", required=True)

    diff_parser = subparsers.add_parser("diff-stages")
    diff_parser.add_argument("--ir-dir", required=True)
    diff_parser.add_argument("--from", dest="from_selector", required=True)
    diff_parser.add_argument("--to", dest="to_selector", required=True)
    diff_parser.add_argument("--context", type=int, default=2)

    changes_parser = subparsers.add_parser("find-changes")
    changes_parser.add_argument("--ir-dir", required=True)
    changes_parser.add_argument("--limit", type=int)
    changes_parser.add_argument(
        "--sort-by",
        choices=["score", "lines", "size"],
        default="score",
    )

    signals_parser = subparsers.add_parser("performance-signals")
    signals_parser.add_argument("--ir-dir", required=True)
    signals_parser.add_argument("--limit", type=int)
    signals_parser.add_argument(
        "--format",
        dest="output_format",
        choices=("text", "json"),
        default="text",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list-stages":
            print(
                list_stages_text(
                    args.ir_dir,
                    grep=args.grep,
                    limit=args.limit,
                    sort_by=args.sort_by,
                ),
                end="",
            )
            return 0
        if args.command == "stage-summary":
            print(stage_summary_text(args.ir_dir, args.stage), end="")
            return 0
        if args.command == "find-changes":
            print(
                find_changes_text(
                    args.ir_dir,
                    limit=args.limit,
                    sort_by=args.sort_by,
                ),
                end="",
            )
            return 0
        if args.command == "performance-signals":
            print(
                performance_signals_text(
                    args.ir_dir,
                    limit=args.limit,
                    output_format=args.output_format,
                ),
                end="",
            )
            return 0
        print(
            diff_stages_text(
                args.ir_dir,
                from_selector=args.from_selector,
                to_selector=args.to_selector,
                context=args.context,
            ),
            end="",
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc))
        return 1


def _build_stage_info(stages_dir: Path, path: Path) -> StageInfo:
    relative_path = path.relative_to(stages_dir).as_posix()
    text = path.read_text(encoding="utf-8")
    return StageInfo(
        path=path,
        relative_path=relative_path,
        stem=path.stem,
        size_bytes=path.stat().st_size,
        line_count=len(text.splitlines()),
    )


def _stage_sort_key(stages_dir: Path, path: Path) -> tuple[int, str, str]:
    stem = path.stem
    match = re.match(r"(\d+)_", stem)
    order = int(match.group(1)) if match else 10**9
    relative = path.relative_to(stages_dir).as_posix()
    return (order, stem, relative)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes}B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f}K"
    return f"{size_bytes / (1024 * 1024):.1f}M"


def _keyword_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {keyword: lowered.count(keyword.lower()) for keyword in KEYWORDS}


def _interesting_score(counts: dict[str, int]) -> int:
    weights = {
        "alloc": 3,
        "copy": 4,
        "matmul": 5,
        "for": 1,
        "if": 1,
        "vector": 3,
        "load": 2,
        "store": 2,
        "dma": 4,
        "wait": 6,
        "set_flag": 6,
        "barrier": 6,
    }
    return sum(counts[keyword] * weights[keyword] for keyword in KEYWORDS)


def _sort_stage_infos(stage_infos: list[StageInfo], *, sort_by: str) -> list[StageInfo]:
    if sort_by == "order":
        return list(stage_infos)
    if sort_by == "size":
        return sorted(stage_infos, key=lambda info: (-info.size_bytes, info.relative_path))
    if sort_by == "lines":
        return sorted(stage_infos, key=lambda info: (-info.line_count, info.relative_path))
    return sorted(
        stage_infos,
        key=lambda info: (
            -_interesting_score(_keyword_counts(info.path.read_text(encoding="utf-8"))),
            info.relative_path,
        ),
    )


def _adjacent_stage_changes(stage_infos: list[StageInfo]) -> list[_StageChange]:
    changes: list[_StageChange] = []
    for index in range(len(stage_infos) - 1):
        previous = stage_infos[index]
        current = stage_infos[index + 1]
        previous_counts = _keyword_counts(previous.path.read_text(encoding="utf-8"))
        current_counts = _keyword_counts(current.path.read_text(encoding="utf-8"))
        keyword_deltas = {
            keyword: current_counts[keyword] - previous_counts[keyword]
            for keyword in KEYWORDS
        }
        score = (
            abs(current.line_count - previous.line_count)
            + abs(current.size_bytes - previous.size_bytes) // 64
            + sum(abs(delta) for delta in keyword_deltas.values())
        )
        changes.append(
            {
                "from": previous,
                "to": current,
                "line_delta": current.line_count - previous.line_count,
                "size_delta": current.size_bytes - previous.size_bytes,
                "keyword_deltas": keyword_deltas,
                "score": score,
            }
        )
    return changes


def _sort_stage_changes(changes: list[_StageChange], *, sort_by: str) -> list[_StageChange]:
    if sort_by == "lines":
        return sorted(changes, key=lambda item: (-abs(item["line_delta"]), str(item["to"])))
    if sort_by == "size":
        return sorted(changes, key=lambda item: (-abs(item["size_delta"]), str(item["to"])))
    return sorted(changes, key=lambda item: (-item["score"], str(item["to"])))


def _format_keyword_delta_summary(keyword_deltas: dict[str, int], limit: int = 5) -> str:
    interesting = [(key, delta) for key, delta in keyword_deltas.items() if delta != 0]
    if not interesting:
        return "no keyword count changes"
    interesting.sort(key=lambda item: (-abs(item[1]), item[0]))
    return ", ".join(f"{key}={delta:+d}" for key, delta in interesting[:limit])


def _stage_highlights(text: str, limit: int = 8) -> list[str]:
    highlights: list[str] = []
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if any(keyword in lowered for keyword in KEYWORDS):
            highlights.append(stripped)
        if len(highlights) >= limit:
            break
    return highlights


def _stage_signal_summary(info: StageInfo) -> _StageSignalSummary:
    counts = _keyword_counts(info.path.read_text(encoding="utf-8"))
    transfer_ops = counts["copy"] + counts["dma"] + counts["load"] + counts["store"]
    sync_ops = counts["wait"] + counts["set_flag"] + counts["barrier"]
    return {
        "stage": info.stem,
        "path": info.relative_path,
        "line_count": info.line_count,
        "size_bytes": info.size_bytes,
        "vector_ops": counts["vector"],
        "transfer_ops": transfer_ops,
        "sync_ops": sync_ops,
        "alloc_ops": counts["alloc"],
        "interesting_score": _interesting_score(counts),
    }


def _suspicious_transitions(stage_infos: list[StageInfo]) -> list[_SuspiciousTransition]:
    suspicious: list[_SuspiciousTransition] = []
    for change in _adjacent_stage_changes(stage_infos):
        keyword_deltas = change["keyword_deltas"]
        vector_delta = keyword_deltas["vector"]
        transfer_delta = (
            keyword_deltas["copy"]
            + keyword_deltas["dma"]
            + keyword_deltas["load"]
            + keyword_deltas["store"]
        )
        sync_delta = (
            keyword_deltas["wait"]
            + keyword_deltas["set_flag"]
            + keyword_deltas["barrier"]
        )
        signal_score = abs(vector_delta) + abs(transfer_delta) + abs(sync_delta)
        if signal_score == 0:
            continue
        previous = change["from"]
        current = change["to"]
        suspicious.append(
            {
                "from_stage": previous.stem,
                "to_stage": current.stem,
                "signal_score": signal_score,
                "vector_delta": vector_delta,
                "transfer_delta": transfer_delta,
                "sync_delta": sync_delta,
                "line_delta": change["line_delta"],
                "size_delta": change["size_delta"],
            }
        )
    return sorted(suspicious, key=lambda item: (-item["signal_score"], str(item["to_stage"])))


def _render_signal_stage_lines(entries: list[_StageSignalSummary]) -> list[str]:
    if not entries:
        return ["- No stages matched the default heuristics."]
    lines: list[str] = []
    for entry in entries:
        lines.append(
            f"- {entry['stage']} "
            f"(vector={entry['vector_ops']}, transfer={entry['transfer_ops']}, "
            f"sync={entry['sync_ops']}, lines={entry['line_count']})"
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
