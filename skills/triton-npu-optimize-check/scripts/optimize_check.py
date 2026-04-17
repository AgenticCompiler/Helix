from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "references" / "contract.json"
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
BASELINE_STATE_REQUIRED_FIELDS = tuple(CONTRACT_DATA["baseline_state_required_fields"])
ROUND_STATE_REQUIRED_FIELDS = tuple(CONTRACT_DATA["round_state_required_fields"])

_BASELINE_METADATA_FILENAMES = {
    "state.json",
    "perf.txt",
}

_ROUND_METADATA_FILENAMES = {
    "attempts.md",
    "summary.md",
    "perf.txt",
    "perf-analysis.md",
    "round-state.json",
}


@dataclass(frozen=True)
class OptimizeCheckResult:
    ok: bool
    kind: Literal["baseline", "round"]
    decision: Literal["pass", "revise-required", "hard-fail"]
    issues: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class BaselineState:
    baseline_kind: str
    source_operator: str
    baseline_operator: str
    test_file: str
    test_mode: str
    bench_file: str
    bench_mode: str
    perf_artifact: str
    correctness_status: str
    benchmark_status: str
    baseline_established: bool


@dataclass(frozen=True)
class RoundState:
    round_name: str
    parent_round: str
    hypothesis: str
    evidence_sources: tuple[str, ...]
    correctness_status: str
    benchmark_status: str
    perf_artifact: str
    canonical_baseline: str
    comparison_target: str
    perf_summary_source: str
    summary_path: str
    opt_note_updated: bool
    next_recommendation: str
    perf_analysis_path: str | None = None
    analysis_comparison_sources: tuple[str, ...] = ()


def check_baseline(baseline_dir: Path) -> OptimizeCheckResult:
    issues = _baseline_gate_issues(baseline_dir)
    if issues:
        return _build_result(
            kind="baseline",
            decision="revise-required",
            issues=issues,
        )
    return _build_result(kind="baseline", decision="pass", issues=())


def check_round(round_dir: Path) -> OptimizeCheckResult:
    artifact_issues = _inspect_round_artifacts(round_dir)
    if artifact_issues:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=artifact_issues,
        )

    try:
        round_state = _load_round_state(round_dir)
    except ValueError as exc:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=(str(exc),),
        )

    if round_state.correctness_status != "passed":
        return _build_result(
            kind="round",
            decision="hard-fail",
            issues=(f"correctness_status={round_state.correctness_status}",),
        )
    if round_state.benchmark_status != "passed":
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=(f"benchmark_status={round_state.benchmark_status}",),
        )

    baseline_issues = _baseline_gate_issues(round_dir.parent / "baseline")
    if baseline_issues:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=baseline_issues,
        )

    semantic_issues: list[str] = []
    if round_state.canonical_baseline != "baseline":
        semantic_issues.append(f"canonical_baseline={round_state.canonical_baseline}")
    if round_state.comparison_target != "baseline/perf.txt":
        semantic_issues.append(f"comparison_target={round_state.comparison_target}")
    if round_state.perf_summary_source != "compare-perf":
        semantic_issues.append(f"perf_summary_source={round_state.perf_summary_source}")
    if not round_state.evidence_sources:
        semantic_issues.append("missing supporting evidence sources")

    if semantic_issues:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=tuple(semantic_issues),
        )

    return _build_result(kind="round", decision="pass", issues=())


def _baseline_gate_issues(baseline_dir: Path) -> tuple[str, ...]:
    artifact_issues = _inspect_baseline_artifacts(baseline_dir)
    if artifact_issues:
        return artifact_issues

    try:
        state = _load_baseline_state(baseline_dir)
    except ValueError as exc:
        return (str(exc),)

    issues: list[str] = []
    if not state.baseline_established:
        issues.append("baseline/state.json marks baseline as not established")
    if state.correctness_status != "passed":
        issues.append(f"baseline correctness_status={state.correctness_status}")
    if state.benchmark_status != "passed":
        issues.append(f"baseline benchmark_status={state.benchmark_status}")
    return tuple(issues)


def _inspect_baseline_artifacts(baseline_dir: Path) -> tuple[str, ...]:
    state_path = _existing_file(baseline_dir / "state.json")
    state: BaselineState | None = None
    if state_path is not None:
        try:
            state = _load_baseline_state(baseline_dir)
        except ValueError:
            state = None

    workspace = baseline_dir.parent
    declared_perf = state.perf_artifact if state is not None else None
    declared_operator = state.baseline_operator if state is not None else None

    perf_path = _declared_workspace_file(workspace, declared_perf) if state is not None else None
    operator_path = (
        _declared_workspace_file(workspace, declared_operator) if state is not None else None
    )

    if state is None and perf_path is None:
        perf_path = _existing_file(baseline_dir / "perf.txt")
    if state is None and operator_path is None:
        operator_path = _find_baseline_operator_snapshot(baseline_dir)

    issues: list[str] = []
    if state_path is None:
        issues.append("missing baseline/state.json")
    if perf_path is None:
        issues.append(_missing_issue(declared_perf, default_path="baseline/perf.txt"))
    if operator_path is None:
        if declared_operator is None:
            issues.append("missing baseline operator snapshot")
        else:
            issues.append(_missing_issue(declared_operator, default_path="baseline operator snapshot"))
    return tuple(issues)


def _load_baseline_state(baseline_dir: Path) -> BaselineState:
    data = _load_json_object(baseline_dir / "state.json", display_name="baseline/state.json")
    missing_fields = [
        field_name for field_name in BASELINE_STATE_REQUIRED_FIELDS if field_name not in data
    ]
    if missing_fields:
        raise ValueError(
            "missing required baseline-state fields: " + ", ".join(missing_fields)
        )
    return BaselineState(
        baseline_kind=str(data["baseline_kind"]),
        source_operator=str(data["source_operator"]),
        baseline_operator=str(data["baseline_operator"]),
        test_file=str(data["test_file"]),
        test_mode=str(data["test_mode"]),
        bench_file=str(data["bench_file"]),
        bench_mode=str(data["bench_mode"]),
        perf_artifact=str(data["perf_artifact"]),
        correctness_status=str(data["correctness_status"]),
        benchmark_status=str(data["benchmark_status"]),
        baseline_established=bool(data["baseline_established"]),
    )


def _inspect_round_artifacts(round_dir: Path) -> tuple[str, ...]:
    attempts_path = _existing_file(round_dir / "attempts.md")
    round_state_path = _existing_file(round_dir / "round-state.json")
    state: RoundState | None = None
    if round_state_path is not None:
        try:
            state = _load_round_state(round_dir)
        except ValueError:
            state = None

    declared_summary = state.summary_path if state is not None else None
    declared_perf = state.perf_artifact if state is not None else None
    declared_analysis = state.perf_analysis_path if state is not None else None

    summary_path = _declared_round_file(round_dir, declared_summary) if state is not None else None
    perf_path = _declared_round_file(round_dir, declared_perf) if state is not None else None
    perf_analysis_path = _declared_round_file(round_dir, declared_analysis) if state is not None else None

    if state is None and summary_path is None:
        summary_path = _existing_file(round_dir / "summary.md")
    if state is None and perf_path is None:
        perf_path = _find_perf_artifact(round_dir)
    operator_path = _find_round_operator(round_dir)

    issues: list[str] = []
    if attempts_path is None:
        issues.append("missing attempts.md")
    if summary_path is None:
        issues.append(_missing_issue(declared_summary, default_path="summary.md"))
    if round_state_path is None:
        issues.append("missing round-state.json")
    if perf_path is None:
        issues.append(_missing_issue(declared_perf, default_path="perf artifact"))
    if declared_analysis is not None and perf_analysis_path is None:
        issues.append(_missing_issue(declared_analysis, default_path="perf-analysis.md"))
    if operator_path is None:
        issues.append("missing round-local operator output")
    return tuple(issues)


def _load_round_state(round_dir: Path) -> RoundState:
    data = _load_json_object(round_dir / "round-state.json", display_name="round-state.json")
    missing_fields = [field_name for field_name in ROUND_STATE_REQUIRED_FIELDS if field_name not in data]
    if missing_fields:
        raise ValueError("missing required round-state fields: " + ", ".join(missing_fields))

    evidence_sources_value = data["evidence_sources"]
    if not isinstance(evidence_sources_value, list):
        raise ValueError("round-state evidence_sources must be a list of strings")

    evidence_sources: list[str] = []
    for item in evidence_sources_value:
        if not isinstance(item, str):
            raise ValueError("round-state evidence_sources must be a list of strings")
        evidence_sources.append(item)
    comparison_sources = _optional_str_tuple(data.get("analysis_comparison_sources"))

    return RoundState(
        round_name=str(data["round"]),
        parent_round=str(data["parent_round"]),
        hypothesis=str(data["hypothesis"]),
        evidence_sources=tuple(evidence_sources),
        correctness_status=str(data["correctness_status"]),
        benchmark_status=str(data["benchmark_status"]),
        perf_artifact=str(data["perf_artifact"]),
        canonical_baseline=str(data["canonical_baseline"]),
        comparison_target=str(data["comparison_target"]),
        perf_summary_source=str(data["perf_summary_source"]),
        summary_path=str(data["summary_path"]),
        opt_note_updated=bool(data["opt_note_updated"]),
        next_recommendation=str(data["next_recommendation"]),
        perf_analysis_path=_optional_str(data.get("perf_analysis_path")),
        analysis_comparison_sources=comparison_sources,
    )


def _load_json_object(path: Path, *, display_name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing {display_name} in {path.parent}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {display_name} in {path.parent}: {exc.msg}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{display_name} must contain an object in {path.parent}")
    return cast(dict[str, Any], payload)


def _existing_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("round-state analysis_comparison_sources must be a list of strings")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("round-state analysis_comparison_sources must be a list of strings")
        items.append(item)
    return tuple(items)


def _declared_workspace_file(workspace: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    return _existing_file(workspace / Path(relative_path))


def _declared_round_file(round_dir: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    return _existing_file(round_dir / Path(relative_path))


def _find_baseline_operator_snapshot(baseline_dir: Path) -> Path | None:
    if not baseline_dir.is_dir():
        return None
    candidates = [
        path
        for path in sorted(baseline_dir.iterdir())
        if path.is_file() and path.name not in _BASELINE_METADATA_FILENAMES
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        preferred_python = [path for path in candidates if path.suffix == ".py"]
        if len(preferred_python) == 1:
            return preferred_python[0]
        return candidates[0]
    return None


def _find_perf_artifact(round_dir: Path) -> Path | None:
    perf_txt = round_dir / "perf.txt"
    if perf_txt.is_file():
        return perf_txt
    perf_files = sorted(round_dir.glob("*_perf.txt"))
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def _find_round_operator(round_dir: Path) -> Path | None:
    candidates = [
        path
        for path in sorted(round_dir.iterdir())
        if path.is_file()
        and path.name not in _ROUND_METADATA_FILENAMES
        and not path.name.endswith("_perf.txt")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        preferred_python = [path for path in candidates if path.suffix == ".py"]
        if len(preferred_python) == 1:
            return preferred_python[0]
        return candidates[0]
    return None


def _missing_issue(relative_path: str | None, *, default_path: str) -> str:
    if relative_path is None:
        return f"missing {default_path}"
    return f"missing {relative_path}"


def _build_result(
    *,
    kind: Literal["baseline", "round"],
    decision: Literal["pass", "revise-required", "hard-fail"],
    issues: tuple[str, ...],
) -> OptimizeCheckResult:
    ok = decision == "pass"
    summary = f"{kind} check passed" if ok else f"{kind} check requires fixes: {'; '.join(issues)}"
    return OptimizeCheckResult(
        ok=ok,
        kind=kind,
        decision=decision,
        issues=issues,
        summary=summary,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=Path(__file__).name)
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("check-baseline")
    baseline.add_argument("--baseline-dir", required=True)

    round_parser = subparsers.add_parser("check-round")
    round_parser.add_argument("--round-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "check-baseline":
        result = check_baseline(Path(args.baseline_dir).expanduser().resolve())
    else:
        result = check_round(Path(args.round_dir).expanduser().resolve())

    print(json.dumps(asdict(result), ensure_ascii=True))
    print(result.summary, file=sys.stderr)
    if result.decision == "pass":
        return 0
    if result.decision == "hard-fail":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
