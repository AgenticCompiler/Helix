from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

from triton_agent.optimize.baseline import baseline_dir
from triton_agent.optimize.batch import resolve_batch_optimize_operator_file
from triton_agent.optimize.naming import resolve_round_perf_file
from triton_agent.optimize.models import OptimizeStatusRound, OptimizeStatusWorkspace
from triton_agent.optimize.round_contract import load_round_state
from triton_agent.skill_loader import load_operator_eval_script_module


class BenchPerfParserModule(Protocol):
    def parse_perf_file(self, path: Path) -> dict[str, float]: ...

    def parse_required_perf_file(
        self,
        path: Path,
        required_latency_ids: Iterable[str],
    ) -> dict[str, float]: ...

    def parse_perf_file_for_metric_source(
        self,
        path: Path,
        *,
        metric_source: str = "auto",
    ) -> dict[str, float]: ...

    def parse_required_perf_file_for_metric_source(
        self,
        path: Path,
        required_latency_ids: Iterable[str],
        *,
        metric_source: str = "auto",
    ) -> dict[str, float]: ...


def _load_bench_perf_parser() -> BenchPerfParserModule:
    return cast(BenchPerfParserModule, load_operator_eval_script_module("perf_artifacts"))

def inspect_optimize_status_workspace(
    workspace: Path,
    *,
    verbose: bool = False,
) -> OptimizeStatusWorkspace:
    del verbose
    opt_note, round_dirs, top_level_perf_files = collect_optimize_status_artifacts(workspace)
    latest_verify_state, verified, verified_geomean_speedup = inspect_latest_verify_result(
        workspace
    )

    has_artifacts = bool(opt_note.exists() or round_dirs or top_level_perf_files)
    if not has_artifacts:
        return OptimizeStatusWorkspace(
            workspace=workspace,
            state="no-session",
            avg_improvement=None,
            geomean_speedup=None,
            best_round=None,
            logged_best=None,
            warnings=(),
            latest_verify_state=latest_verify_state,
            verified=verified,
            verified_geomean_speedup=verified_geomean_speedup,
        )

    warnings: list[str] = []
    baseline_path, baseline_selection_failed = select_baseline_perf_file(workspace, top_level_perf_files, warnings)
    baseline_values_by_source: dict[str, dict[str, float]] = {}
    has_any_baseline_values = False
    if baseline_path is not None:
        try:
            _load_bench_perf_parser().parse_perf_file(baseline_path)
            has_any_baseline_values = True
        except ValueError as exc:
            warnings.append(str(exc))

    logged_best: str | None = None
    summary_logged_best: str | None = None
    legacy_logged_best: str | None = None
    logged_geomean_speedup: str | None = None
    if opt_note.exists():
        (
            summary_logged_best,
            legacy_logged_best,
            logged_geomean_speedup,
        ) = parse_logged_best_round_details(opt_note)
        logged_best = summary_logged_best or legacy_logged_best
        if (
            summary_logged_best is not None
            and legacy_logged_best is not None
            and summary_logged_best != legacy_logged_best
        ):
            warnings.append("overall summary best round differs from legacy current best marker")
    comparable_rounds: list[OptimizeStatusRound] = []

    for round_dir in round_dirs:
        perf_path = find_round_perf_file(round_dir)
        if perf_path is None:
            warnings.append(f"missing perf artifact for {round_dir.name}")
            continue
        metric_source = _metric_source_for_round_status(round_dir)
        try:
            baseline_values = baseline_values_by_source.get(metric_source)
            if baseline_values is None:
                baseline_values = _parse_baseline_values_for_metric_source(
                    baseline_path,
                    metric_source=metric_source,
                )
                baseline_values_by_source[metric_source] = baseline_values
            round_values = _parse_round_values_for_metric_source(
                perf_path,
                baseline_values,
                metric_source=metric_source,
            )
        except (ValueError, OSError) as exc:
            warnings.append(str(exc))
            continue
        if set(baseline_values) != set(round_values):
            warnings.append("latency ids do not match for comparable perf data")
            continue

        improvement_values: list[float] = []
        speedup_values: list[float] = []
        valid_baseline_values: list[float] = []
        valid_round_values: list[float] = []
        for latency_id in sorted(baseline_values):
            baseline_value = baseline_values[latency_id]
            if baseline_value <= 0:
                warnings.append(f"baseline latency must be > 0 for {latency_id}")
                continue
            round_value = round_values[latency_id]
            if round_value <= 0:
                warnings.append(f"round latency must be > 0 for {round_dir.name}:{latency_id}")
                continue
            improvement_values.append((baseline_value - round_value) / baseline_value)
            speedup_values.append(baseline_value / round_value)
            valid_baseline_values.append(baseline_value)
            valid_round_values.append(round_value)
        if not improvement_values:
            continue
        comparable_rounds.append(
            OptimizeStatusRound(
                round_name=f"round-{round_number(round_dir.name)}",
                effective_metric_source=metric_source,
                avg_improvement=mean_value(improvement_values),
                geomean_speedup=geomean_value(speedup_values),
                mean_latency=mean_value(valid_round_values),
            )
        )

    if comparable_rounds:
        best_round = max(
            comparable_rounds,
            key=lambda item: (item.geomean_speedup, -item.mean_latency),
        )
        if logged_best is not None and logged_best != best_round.round_name:
            warnings.append(
                "numeric best round != logged best. "
                "computed speedup: "
                f"{format_speedup_value(best_round.geomean_speedup)}; "
                "logged speedup: "
                f"{format_speedup_source(logged_geomean_speedup)}"
            )
        return OptimizeStatusWorkspace(
            workspace=workspace,
            state="ok",
            avg_improvement=best_round.avg_improvement,
            geomean_speedup=best_round.geomean_speedup,
            best_round=best_round.round_name,
            logged_best=logged_best,
            warnings=tuple(dict.fromkeys(warnings)),
            latest_verify_state=latest_verify_state,
            verified=verified,
            verified_geomean_speedup=verified_geomean_speedup,
        )

    if baseline_path is None and not baseline_selection_failed:
        warnings.append("missing baseline perf data")
    elif (has_any_baseline_values or baseline_values_by_source) and round_dirs:
        warnings.append("missing comparable round perf data")

    return OptimizeStatusWorkspace(
        workspace=workspace,
        state="warning",
        avg_improvement=None,
        geomean_speedup=None,
        best_round=None,
        logged_best=logged_best,
        warnings=tuple(dict.fromkeys(warnings)),
        latest_verify_state=latest_verify_state,
        verified=verified,
        verified_geomean_speedup=verified_geomean_speedup,
    )


def scan_optimize_status_workspaces(root: Path, *, verbose: bool = False) -> list[OptimizeStatusWorkspace]:
    return [
        inspect_optimize_status_workspace(workspace, verbose=verbose)
        for workspace in sorted(path for path in root.iterdir() if path.is_dir())
    ]


def workspace_has_optimize_artifacts(workspace: Path) -> bool:
    opt_note, round_dirs, top_level_perf_files = collect_optimize_status_artifacts(workspace)
    baseline_dir_path = baseline_dir(workspace)
    if baseline_dir_path.is_dir() and (
        (baseline_dir_path / "perf.txt").is_file() or any(baseline_dir_path.glob("*_perf.txt"))
    ):
        return True
    return bool(opt_note.exists() or round_dirs or top_level_perf_files)


def collect_optimize_status_artifacts(
    workspace: Path,
) -> tuple[Path, list[Path], list[Path]]:
    opt_note = workspace / "opt-note.md"
    round_dirs = sorted(
        (path for path in workspace.iterdir() if path.is_dir() and round_number(path.name) is not None),
        key=lambda path: (round_number(path.name) or 0),
    )
    top_level_perf_files = sorted(workspace.glob("*_perf.txt"))
    return opt_note, round_dirs, top_level_perf_files


def inspect_latest_verify_result(workspace: Path) -> tuple[Path | None, bool, float | None]:
    state_path = find_latest_verify_state(workspace)
    if state_path is None:
        return None, False, None
    verified, geomean_speedup = inspect_verify_state_summary(state_path)
    return state_path, verified, geomean_speedup


def find_latest_verify_state(workspace: Path) -> Path | None:
    verify_root = workspace / "opt-verify"
    if not verify_root.is_dir():
        return None
    candidates = sorted(
        path / "verify-state.json"
        for path in verify_root.iterdir()
        if path.is_dir() and path.name.startswith("verify-") and (path / "verify-state.json").is_file()
    )
    if not candidates:
        return None
    return candidates[-1]


def verify_state_is_verified(state_path: Path) -> bool:
    verified, _ = inspect_verify_state_summary(state_path)
    return verified


def inspect_verify_state_summary(state_path: Path) -> tuple[bool, float | None]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    verify_result = payload.get("verify-result")
    if not isinstance(verify_result, dict):
        return False, None
    verify_result_dict = cast(dict[str, object], verify_result)
    required_entries: tuple[object | None, ...] = (
        verify_result_dict.get("test"),
        verify_result_dict.get("rerun_baseline_bench"),
        verify_result_dict.get("rerun_best_bench"),
        verify_result_dict.get("compare_perf"),
    )
    for entry in required_entries:
        if not isinstance(entry, dict):
            return False, None
        entry_dict = cast(dict[str, object], entry)
        status = entry_dict.get("status")
        if status != "passed":
            return False, None
    speedup = verify_result_dict.get("speedup")
    if not isinstance(speedup, dict):
        return True, None
    speedup_dict = cast(dict[str, object], speedup)
    return (
        True,
        _optional_float(speedup_dict.get("geomean_speedup")),
    )


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def select_baseline_perf_file(
    workspace: Path,
    paths: list[Path],
    warnings: list[str],
) -> tuple[Path | None, bool]:
    baseline_perf = baseline_dir(workspace) / "perf.txt"
    if baseline_perf.is_file():
        return baseline_perf, False

    baseline_dir_path = baseline_dir(workspace)
    if baseline_dir_path.is_dir():
        operator_perf_files = sorted(baseline_dir_path.glob("*_perf.txt"))
        if len(operator_perf_files) == 1:
            return operator_perf_files[0], False
        if len(operator_perf_files) > 1:
            warnings.append("found multiple baseline perf files")
            return None, True

    if not paths:
        return None, False

    operator_perf = resolve_workspace_operator_perf_file(workspace, paths)
    if operator_perf is not None:
        return operator_perf, False

    baseline_named = next((path for path in paths if path.name == "baseline_perf.txt"), None)
    if baseline_named is not None:
        return baseline_named, False

    if len(paths) == 1:
        return paths[0], False
    non_opt_paths = [path for path in paths if not path.stem.startswith("opt_")]
    if len(non_opt_paths) == 1:
        return non_opt_paths[0], False
    if len(non_opt_paths) > 1:
        warnings.append("found multiple baseline perf files")
        return None, True
    warnings.append("found multiple baseline perf files")
    return None, True


def resolve_workspace_operator_perf_file(workspace: Path, paths: list[Path]) -> Path | None:
    try:
        operator_file = resolve_batch_optimize_operator_file(workspace)
    except ValueError:
        return None
    expected_name = f"{operator_file.stem}_perf.txt"
    return next((path for path in paths if path.name == expected_name), None)


def find_round_perf_file(round_dir: Path) -> Path | None:
    return resolve_round_perf_file(round_dir)


def _metric_source_for_round_status(round_dir: Path) -> str:
    try:
        state = load_round_state(round_dir)
    except ValueError:
        return "auto"
    effective_metric_source = state.effective_metric_source
    if effective_metric_source == "kernel":
        return "kernel"
    if effective_metric_source == "total-op":
        return "total-op"
    return "auto"


def _parse_baseline_values_for_metric_source(
    baseline_path: Path | None,
    *,
    metric_source: str,
) -> dict[str, float]:
    if baseline_path is None:
        raise ValueError("missing baseline perf data")
    parser = _load_bench_perf_parser()
    if metric_source == "auto":
        return parser.parse_perf_file(baseline_path)
    return parser.parse_perf_file_for_metric_source(
        baseline_path,
        metric_source=metric_source,
    )


def _parse_round_values_for_metric_source(
    perf_path: Path,
    baseline_values: dict[str, float],
    *,
    metric_source: str,
) -> dict[str, float]:
    parser = _load_bench_perf_parser()
    if metric_source == "auto":
        return parser.parse_required_perf_file(perf_path, baseline_values)
    return parser.parse_required_perf_file_for_metric_source(
        perf_path,
        baseline_values,
        metric_source=metric_source,
    )


def parse_logged_best_round(path: Path) -> str | None:
    summary_best, legacy_best = parse_logged_best_rounds(path)
    return summary_best or legacy_best


def parse_logged_best_rounds(path: Path) -> tuple[str | None, str | None]:
    summary_best, legacy_best, _ = parse_logged_best_round_details(path)
    return summary_best, legacy_best


def parse_logged_best_round_details(path: Path) -> tuple[str | None, str | None, str | None]:
    current_round: str | None = None
    legacy_best: str | None = None
    summary_best: str | None = None
    summary_geomean_speedup: str | None = None
    in_overall_summary = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = re.match(r"##\s+Round\s+(\d+)", line)
        if match:
            current_round = f"round-{match.group(1)}"
            in_overall_summary = False
            continue
        if re.match(r"##\s+Overall\s+Summary\b", line, flags=re.IGNORECASE):
            current_round = None
            in_overall_summary = True
            continue
        if line.startswith("##"):
            current_round = None
            in_overall_summary = False
            continue
        if in_overall_summary:
            summary_match = re.match(r"Final\s+best\s+round:\s*(.+)", line, flags=re.IGNORECASE)
            if summary_match:
                normalized = normalize_round_name(summary_match.group(1))
                if normalized is not None:
                    summary_best = normalized
            geomean_match = re.match(r"Geomean\s+speedup:\s*(.+)", line, flags=re.IGNORECASE)
            if geomean_match:
                summary_geomean_speedup = normalize_summary_value(geomean_match.group(1))
            continue
        if line.lower().startswith("best status:") and "current best" in line.lower():
            legacy_best = current_round
    return summary_best, legacy_best, summary_geomean_speedup


def normalize_round_name(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None

    for pattern in (
        r"round-(\d+)",
        r"round\s+(\d+)",
        r"opt-round-(\d+)",
    ):
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return f"round-{match.group(1)}"
    return None


def round_number(name: str) -> int | None:
    match = re.fullmatch(r"opt-round-(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def mean_value(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def geomean_value(values: Iterable[float]) -> float:
    items = list(values)
    return math.exp(sum(math.log(item) for item in items) / len(items))


def format_speedup_value(value: float) -> str:
    return f"{value:.2f}x"


def format_speedup_source(value: str | None) -> str:
    if value is None:
        return "missing"
    return value


def normalize_summary_value(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    return text
