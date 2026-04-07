from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from triton_agent.bench_runner import parse_perf_file
from triton_agent.optimize.models import OptimizeStatusRound, OptimizeStatusWorkspace


def inspect_optimize_status_workspace(
    workspace: Path,
    *,
    verbose: bool = False,
) -> OptimizeStatusWorkspace:
    del verbose
    opt_note = workspace / "opt-note.md"
    round_dirs = sorted(
        (path for path in workspace.iterdir() if path.is_dir() and round_number(path.name) is not None),
        key=lambda path: (round_number(path.name) or 0),
    )
    top_level_perf_files = sorted(workspace.glob("*_perf.txt"))

    has_artifacts = bool(opt_note.exists() or round_dirs or top_level_perf_files)
    if not has_artifacts:
        return OptimizeStatusWorkspace(
            workspace=workspace,
            state="no-session",
            baseline_mean=None,
            best_mean=None,
            avg_improvement=None,
            best_round=None,
            logged_best=None,
            warnings=(),
        )

    warnings: list[str] = []
    baseline_path = select_baseline_perf_file(top_level_perf_files, warnings)
    baseline_values: dict[str, float] | None = None
    baseline_mean: float | None = None
    if baseline_path is not None:
        try:
            parsed_baseline_values = parse_perf_file(baseline_path)
            baseline_values = parsed_baseline_values
            baseline_mean = mean_value(parsed_baseline_values.values())
        except ValueError as exc:
            warnings.append(str(exc))

    logged_best = parse_logged_best_round(opt_note) if opt_note.exists() else None
    comparable_rounds: list[OptimizeStatusRound] = []

    for round_dir in round_dirs:
        if baseline_values is None:
            continue
        perf_path = find_round_perf_file(round_dir)
        if perf_path is None:
            warnings.append(f"missing perf artifact for {round_dir.name}")
            continue
        try:
            round_values = parse_perf_file(perf_path)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        if set(baseline_values) != set(round_values):
            warnings.append("latency ids do not match for comparable perf data")
            continue

        score_values: list[float] = []
        for latency_id in sorted(baseline_values):
            baseline_value = baseline_values[latency_id]
            if baseline_value <= 0:
                warnings.append(f"baseline latency must be > 0 for {latency_id}")
                continue
            score_values.append((baseline_value - round_values[latency_id]) / baseline_value)
        if not score_values:
            continue
        comparable_rounds.append(
            OptimizeStatusRound(
                round_name=f"round-{round_number(round_dir.name)}",
                score=mean_value(score_values),
                mean_latency=mean_value(round_values.values()),
            )
        )

    if comparable_rounds:
        best_round = max(comparable_rounds, key=lambda item: (item.score, -item.mean_latency))
        if logged_best is not None and logged_best != best_round.round_name:
            warnings.append("numeric best round differs from logged best round")
        return OptimizeStatusWorkspace(
            workspace=workspace,
            state="ok",
            baseline_mean=baseline_mean,
            best_mean=best_round.mean_latency,
            avg_improvement=best_round.score,
            best_round=best_round.round_name,
            logged_best=logged_best,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    if baseline_path is None:
        warnings.append("missing baseline perf data")
    elif baseline_values is not None and round_dirs:
        warnings.append("missing comparable round perf data")

    return OptimizeStatusWorkspace(
        workspace=workspace,
        state="warning",
        baseline_mean=baseline_mean,
        best_mean=None,
        avg_improvement=None,
        best_round=None,
        logged_best=logged_best,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def scan_optimize_status_workspaces(root: Path, *, verbose: bool = False) -> list[OptimizeStatusWorkspace]:
    return [
        inspect_optimize_status_workspace(workspace, verbose=verbose)
        for workspace in sorted(path for path in root.iterdir() if path.is_dir())
    ]


def select_baseline_perf_file(paths: list[Path], warnings: list[str]) -> Path | None:
    if not paths:
        return None
    if len(paths) > 1:
        warnings.append("found multiple baseline perf files")
        return None
    return paths[0]


def find_round_perf_file(round_dir: Path) -> Path | None:
    perf_txt = round_dir / "perf.txt"
    if perf_txt.is_file():
        return perf_txt
    perf_files = sorted(round_dir.glob("*_perf.txt"))
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def parse_logged_best_round(path: Path) -> str | None:
    current_round: str | None = None
    logged_best: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.match(r"##\s+Round\s+(\d+)", line)
        if match:
            current_round = f"round-{match.group(1)}"
            continue
        if line.lower().startswith("best status:") and "current best" in line.lower():
            logged_best = current_round
    return logged_best


def round_number(name: str) -> int | None:
    match = re.fullmatch(r"opt-round-(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def mean_value(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)
