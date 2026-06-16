from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TextIO

from triton_agent.diff_skills_update.models import DiscoveryResult, OperatorPair, SkipRecord


def discover_operator_pairs(
    root: Path,
    *,
    stream: TextIO | None = None,
    exclude_dirs: set[Path] | None = None,
) -> DiscoveryResult:
    if not root.exists():
        raise ValueError(f"Input path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Input path is not a directory: {root}")

    pairs: list[OperatorPair] = []
    skips: list[SkipRecord] = []
    excluded = {path.resolve() for path in exclude_dirs or set()}
    if (root / "learned_lessons.md").is_file():
        pair, skip = _discover_optimize_process_pair(root, stream=stream)
        if pair is not None:
            pairs.append(pair)
        if skip is not None:
            skips.append(skip)
        return DiscoveryResult(pairs=tuple(pairs), skips=tuple(skips))
    for operator_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if operator_dir.resolve() in excluded:
            continue
        if (operator_dir / "learned_lessons.md").is_file():
            pair, skip = _discover_optimize_process_pair(operator_dir, stream=stream)
            if pair is not None:
                pairs.append(pair)
            if skip is not None:
                skips.append(skip)
            continue
        opt_files = sorted(operator_dir.glob("opt_*.py"))
        if not opt_files:
            skips.append(_record_skip(operator_dir, "no opt_*.py file found", stream=stream))
            continue
        for opt_path in opt_files:
            baseline_name = opt_path.name.removeprefix("opt_")
            baseline_path = operator_dir / baseline_name
            if not baseline_path.exists():
                skips.append(
                    _record_skip(
                        operator_dir,
                        f"missing baseline file {baseline_name} for {opt_path.name}",
                        opt_path=opt_path,
                        stream=stream,
                    )
                )
                continue
            if not baseline_path.is_file():
                skips.append(
                    _record_skip(
                        operator_dir,
                        f"baseline path is not a file: {baseline_path.name}",
                        opt_path=opt_path,
                        stream=stream,
                    )
                )
                continue
            pairs.append(
                OperatorPair(
                    operator_dir=operator_dir,
                    baseline_path=baseline_path,
                    expected_path=opt_path,
                )
            )
    return DiscoveryResult(pairs=tuple(pairs), skips=tuple(skips))


def _discover_optimize_process_pair(
    operator_dir: Path,
    *,
    stream: TextIO | None = None,
) -> tuple[OperatorPair | None, SkipRecord | None]:
    baseline_path = _resolve_baseline_operator(operator_dir)
    if baseline_path is None:
        skip = _record_skip(
            operator_dir,
            "learned_lessons.md found but baseline operator was not found",
            stream=stream,
        )
        return None, skip
    expected_path = _resolve_final_round_operator(operator_dir, baseline_path.name)
    if expected_path is None:
        skip = _record_skip(
            operator_dir,
            "learned_lessons.md found but final optimized operator was not found",
            stream=stream,
        )
        return None, skip
    opt_note_path = operator_dir / "opt-note.md"
    context_paths = _optimize_process_context_paths(operator_dir, opt_note_path)
    return (
        OperatorPair(
            operator_dir=operator_dir,
            baseline_path=baseline_path,
            expected_path=expected_path,
            learned_lessons_path=operator_dir / "learned_lessons.md",
            opt_note_path=opt_note_path if opt_note_path.is_file() else None,
            context_paths=context_paths,
            source_kind="optimize-process",
        ),
        None,
    )


def _resolve_baseline_operator(operator_dir: Path) -> Path | None:
    baseline_dir = operator_dir / "baseline"
    state_path = baseline_dir / "state.json"
    if state_path.is_file():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        if isinstance(data, dict):
            for key in ("baseline_operator", "source_operator"):
                candidate = data.get(key)
                if isinstance(candidate, str) and candidate:
                    for base_dir in (baseline_dir, operator_dir):
                        path = (base_dir / candidate).resolve()
                        if path.is_file():
                            return path
    candidates = _operator_py_candidates(baseline_dir)
    return candidates[0] if candidates else None


def _resolve_final_round_operator(operator_dir: Path, baseline_name: str) -> Path | None:
    round_dir = _resolve_final_round_dir(operator_dir)
    if round_dir is None:
        return None
    preferred = round_dir / f"opt_{baseline_name}"
    if preferred.is_file():
        return preferred
    candidates = sorted(round_dir.glob("opt_*.py"))
    if candidates:
        return candidates[0]
    py_candidates = _operator_py_candidates(round_dir)
    return py_candidates[0] if py_candidates else None


def _resolve_final_round_dir(operator_dir: Path) -> Path | None:
    opt_note_path = operator_dir / "opt-note.md"
    if opt_note_path.is_file():
        text = opt_note_path.read_text(encoding="utf-8")
        match = re.search(r"Final best round:\s*(?:opt-)?round-(\d+)", text)
        if match:
            round_dir = operator_dir / f"opt-round-{match.group(1)}"
            if round_dir.is_dir():
                return round_dir
    round_dirs = sorted(
        (path for path in operator_dir.glob("opt-round-*") if path.is_dir()),
        key=_round_sort_key,
    )
    return round_dirs[-1] if round_dirs else None


def _round_sort_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"opt-round-(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return -1, path.name


def _operator_py_candidates(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in sorted(directory.glob("*.py"))
        if not path.name.startswith(("test_", "bench_", "differential_test_"))
    ]


def _optimize_process_context_paths(operator_dir: Path, opt_note_path: Path) -> tuple[Path, ...]:
    paths: list[Path] = [operator_dir / "learned_lessons.md"]
    if opt_note_path.is_file():
        paths.append(opt_note_path)
    for round_dir in sorted(
        (path for path in operator_dir.glob("opt-round-*") if path.is_dir()),
        key=_round_sort_key,
    ):
        for name in ("summary.md", "attempts.md", "perf-analysis.md"):
            path = round_dir / name
            if path.is_file():
                paths.append(path)
    return tuple(paths)


def _record_skip(
    operator_dir: Path,
    reason: str,
    *,
    opt_path: Path | None = None,
    stream: TextIO | None = None,
) -> SkipRecord:
    record = SkipRecord(operator_dir=operator_dir, reason=reason, opt_path=opt_path)
    if stream is not None:
        print(f"skip {operator_dir}: {reason}", file=stream)
    return record
