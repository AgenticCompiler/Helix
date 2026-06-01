from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from kernel_continuity_check import analyze_triton_kernel_continuity
from local_optimum_check import collect_local_optimum_warnings

_BATCH_OPTIMIZE_EXCLUDED_PREFIXES = ("test_", "differential_test_", "bench_", "opt_")
_BATCH_OPTIMIZE_EXCLUDED_NAMES = {"__init__.py"}
_OPTIMIZE_DELETE_PT_FILES_ENV = "TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def ordinary_optimize_pt_cleanup_enabled() -> bool:
    raw_value = os.environ.get(_OPTIMIZE_DELETE_PT_FILES_ENV)
    if raw_value is None:
        return False
    return raw_value.strip().lower() in _TRUTHY_ENV_VALUES


def cleanup_dir_pt_files(directory: Path) -> list[str]:
    cleaned: list[str] = []
    for pt_file in sorted(directory.iterdir()):
        if not pt_file.is_file():
            continue
        name_lower = pt_file.name.lower()
        if not (name_lower == "test_result.pt" or name_lower.endswith("_result.pt")):
            continue
        try:
            pt_file.unlink()
            cleaned.append(pt_file.name)
        except OSError:
            pass
    return cleaned

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "references" / "contract.json"
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
BASELINE_STATE_FIELDS = {
    str(field_name): str(description)
    for field_name, description in CONTRACT_DATA["baseline_state_fields"].items()
}
BASELINE_STATE_REQUIRED_FIELDS = tuple(BASELINE_STATE_FIELDS)
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
    next_option: str | None = None


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
    preparation_notes: str | None = None
    baseline_repairs_summary: str | None = None


@dataclass(frozen=True)
class BaselineArtifactsInspection:
    baseline_dir: Path
    state_path: Path | None
    perf_path: Path | None
    operator_path: Path | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RoundState:
    round_name: str
    parent_round: str
    hypothesis: str
    evidence_sources: tuple[str, ...]
    correctness_status: str
    benchmark_status: str
    perf_artifact: str
    comparison_target: str
    effective_metric_source: str
    summary_path: str
    opt_note_updated: bool
    round_disposition: str
    analysis_skipped_reason: str | None = None
    profile_dir: str | None = None
    ir_dir: str | None = None
    perf_analysis_path: str | None = None
    analysis_comparison_sources: tuple[str, ...] = ()
    validated_candidate: bool | None = None


@dataclass(frozen=True)
class RoundArtifactsInspection:
    round_dir: Path
    operator_path: Path | None
    attempts_path: Path | None
    summary_path: Path | None
    perf_path: Path | None
    perf_analysis_path: Path | None
    round_state_path: Path | None
    issues: tuple[str, ...]


def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"


def load_baseline_state(workspace: Path) -> BaselineState:
    state_path = baseline_dir(workspace) / "state.json"
    data = _load_json_object(state_path, display_name="baseline/state.json")
    missing_fields = [
        field_name for field_name in BASELINE_STATE_REQUIRED_FIELDS if field_name not in data
    ]
    if missing_fields:
        raise ValueError("missing required baseline-state fields: " + ", ".join(missing_fields))
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
        preparation_notes=_optional_str(data.get("preparation_notes")),
        baseline_repairs_summary=_optional_str(data.get("baseline_repairs_summary")),
    )


def inspect_baseline_artifacts(workspace: Path) -> BaselineArtifactsInspection:
    root = baseline_dir(workspace)
    state_path = _existing_file(root / "state.json")
    state: BaselineState | None = None
    if state_path is not None:
        try:
            state = load_baseline_state(workspace)
        except ValueError:
            state = None

    declared_perf = state.perf_artifact if state is not None else None
    declared_operator = state.baseline_operator if state is not None else None

    if state is None:
        perf_path = None
        operator_path = None
    else:
        perf_path = _declared_workspace_file(workspace, declared_perf)
        operator_path = _declared_workspace_file(workspace, declared_operator)

    if state is None and perf_path is None:
        perf_path = _existing_file(root / "perf.txt")
    if state is None and operator_path is None:
        operator_path = _find_baseline_operator_snapshot(root)

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

    return BaselineArtifactsInspection(
        baseline_dir=root,
        state_path=state_path,
        perf_path=perf_path,
        operator_path=operator_path,
        issues=tuple(issues),
    )


def baseline_gate_issues(workspace: Path) -> tuple[str, ...]:
    inspection = inspect_baseline_artifacts(workspace)
    if inspection.issues:
        return inspection.issues

    try:
        state = load_baseline_state(workspace)
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


def load_round_state(round_dir: Path) -> RoundState:
    round_state_path = round_dir / "round-state.json"
    data = _load_json_object(round_state_path, display_name="round-state.json")
    missing_fields = [
        field_name for field_name in ROUND_STATE_REQUIRED_FIELDS if field_name not in data
    ]
    if missing_fields:
        raise ValueError("missing required round-state fields: " + ", ".join(missing_fields))

    evidence_sources_value = data["evidence_sources"]
    if not isinstance(evidence_sources_value, list):
        raise ValueError("round-state evidence_sources must be a list of strings")
    evidence_sources_raw = cast(list[Any], evidence_sources_value)
    evidence_sources: list[str] = []
    for item in evidence_sources_raw:
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
        comparison_target=str(data["comparison_target"]),
        effective_metric_source=str(data["effective_metric_source"]),
        summary_path=str(data["summary_path"]),
        opt_note_updated=bool(data["opt_note_updated"]),
        round_disposition=str(data["round_disposition"]),
        analysis_skipped_reason=_optional_str(data.get("analysis_skipped_reason")),
        profile_dir=_optional_str(data.get("profile_dir")),
        ir_dir=_optional_str(data.get("ir_dir")),
        perf_analysis_path=_optional_str(data.get("perf_analysis_path")),
        analysis_comparison_sources=comparison_sources,
        validated_candidate=_optional_bool(data.get("validated_candidate")),
    )


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    workspace = round_dir.parent
    attempts_path = _existing_file(round_dir / "attempts.md")
    round_state_path = _existing_file(round_dir / "round-state.json")
    state: RoundState | None = None
    if round_state_path is not None:
        try:
            state = load_round_state(round_dir)
        except ValueError:
            state = None

    declared_summary = state.summary_path if state is not None else None
    declared_perf = state.perf_artifact if state is not None else None
    declared_analysis = state.perf_analysis_path if state is not None else None

    if state is None:
        summary_path = None
        perf_path = None
        perf_analysis_path = None
    else:
        summary_path = _declared_round_file(round_dir, declared_summary)
        perf_path = _declared_round_file(round_dir, declared_perf)
        perf_analysis_path = _declared_round_file(round_dir, declared_analysis)

    if state is None and summary_path is None:
        summary_path = _existing_file(round_dir / "summary.md")
    expected_operator_name_value, expected_perf_name_value = _expected_round_artifact_names(workspace)
    if perf_path is None:
        perf_path = resolve_round_perf_file(round_dir)
    operator_path = resolve_round_operator_file(round_dir)

    issues: list[str] = []
    if attempts_path is None:
        issues.append("missing attempts.md")
    if summary_path is None:
        issues.append(_missing_issue(declared_summary, default_path="summary.md"))
    if round_state_path is None:
        issues.append("missing round-state.json")
    if perf_path is None:
        issues.append(_missing_issue(declared_perf, default_path=expected_perf_name_value))
    elif state is not None and declared_perf != perf_path.name:
        issues.append(f"perf_artifact must be {expected_perf_name_value}")
    if declared_analysis is not None and perf_analysis_path is None:
        issues.append(_missing_issue(declared_analysis, default_path="perf-analysis.md"))
    if operator_path is None:
        issues.append(f"missing {expected_operator_name_value}")

    return RoundArtifactsInspection(
        round_dir=round_dir,
        operator_path=operator_path,
        attempts_path=attempts_path,
        summary_path=summary_path,
        perf_path=perf_path,
        perf_analysis_path=perf_analysis_path,
        round_state_path=round_state_path,
        issues=tuple(issues),
    )


def check_baseline(baseline_dir_path: Path) -> OptimizeCheckResult:
    issues = baseline_gate_issues(baseline_dir_path.parent)
    if issues:
        return _build_result(
            kind="baseline",
            decision="revise-required",
            issues=issues,
        )
    return _build_result(kind="baseline", decision="pass", issues=())


def _count_round_directories(workspace: Path) -> int:
    return sum(1 for path in workspace.glob("opt-round-*") if path.is_dir())


def _next_round_name_for_round(round_dir: Path, *, completed: int) -> str:
    name = round_dir.name
    prefix = "opt-round-"
    if name.startswith(prefix):
        suffix = name[len(prefix):]
        if suffix.isdigit():
            return f"{prefix}{int(suffix) + 1}"
    return f"{prefix}{completed + 1}"


def check_round(
    round_dir: Path,
    *,
    min_rounds: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
) -> OptimizeCheckResult:
    artifact_inspection = inspect_round_artifacts(round_dir)
    artifact_issues = artifact_inspection.issues
    if artifact_issues:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=artifact_issues,
        )

    try:
        round_state = load_round_state(round_dir)
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

    baseline_issues = baseline_gate_issues(round_dir.parent)
    if baseline_issues:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=baseline_issues,
        )

    semantic_issues: list[str] = []
    baseline_perf_path: Path | None = None
    try:
        baseline = load_baseline_state(round_dir.parent)
        if round_state.comparison_target != baseline.perf_artifact:
            semantic_issues.append(
                f"comparison_target={round_state.comparison_target} "
                f"(expected {baseline.perf_artifact})"
            )
        baseline_perf_path = _declared_workspace_file(round_dir.parent, baseline.perf_artifact)
    except ValueError:
        semantic_issues.append("cannot validate comparison_target: baseline state is invalid")
    if round_state.effective_metric_source not in {"kernel", "total-op", "mixed"}:
        semantic_issues.append(
            f"effective_metric_source={round_state.effective_metric_source}"
        )
    if not round_state.evidence_sources:
        semantic_issues.append("missing supporting evidence sources")

    if semantic_issues:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=tuple(semantic_issues),
        )

    operator_path = artifact_inspection.operator_path
    if operator_path is None:
        expected_operator_name_value, _expected_perf_name_value = _expected_round_artifact_names(round_dir.parent)
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=(f"missing {expected_operator_name_value}",),
        )

    continuity = analyze_triton_kernel_continuity(operator_path)
    if not continuity.ok:
        return _build_result(
            kind="round",
            decision="revise-required",
            issues=((continuity.reason or "round operator failed Triton continuity check"),),
        )

    runtime_warnings: list[str] = []
    if optimize_target == "kernel" and round_state.effective_metric_source in {"total-op", "mixed"}:
        runtime_warnings.append(
            "kernel optimize target fell back to "
            f"effective_metric_source={round_state.effective_metric_source}; "
            "the round may still participate in best-round selection, but review the comparison basis."
        )

    cleaned: list[str] = []
    if ordinary_optimize_pt_cleanup_enabled():
        cleaned = cleanup_dir_pt_files(round_dir)
    local_optimum_warnings: tuple[str, ...] = ()
    if baseline_perf_path is not None:
        local_optimum_warnings = collect_local_optimum_warnings(
            round_dir,
            baseline_perf_path=baseline_perf_path,
        )
    if cleaned:
        result = _build_result(
            kind="round",
            decision="pass",
            issues=(
                *tuple(runtime_warnings),
                *local_optimum_warnings,
                f"cleaned up {len(cleaned)} unused pt file(s) in {round_dir.name}: {', '.join(cleaned)}",
            ),
        )
    else:
        result = _build_result(
            kind="round",
            decision="pass",
            issues=(*tuple(runtime_warnings), *local_optimum_warnings),
        )

    if min_rounds is not None:
        completed = _count_round_directories(round_dir.parent)
        if completed >= min_rounds:
            result = _build_result(
                kind="round",
                decision="pass",
                issues=result.issues,
                summary=_append_pass_issues_to_summary(
                    f"round check passed. "
                    f"Minimum round requirement satisfied ({completed}/{min_rounds}) — "
                    f"the optimize session may stop after this round.",
                    result.issues,
                ),
                next_option=None,
            )
        else:
            next_round_name = _next_round_name_for_round(round_dir, completed=completed)
            result = _build_result(
                kind="round",
                decision="pass",
                issues=result.issues,
                summary=_append_pass_issues_to_summary(
                    f"round check passed. "
                    f"Round {completed}/{min_rounds} complete — "
                    f"at least {min_rounds - completed} more round(s) required before stopping. "
                    f"Next round: {next_round_name}. "
                    "Do not rush into the next code change. "
                    "First decide which operator, kernel path, or wrapper bottleneck should anchor the next round. "
                    "Decide whether existing evidence is already sufficient or whether profiling, IR, or compiler-source analysis is needed first. "
                    "Do not use agents or subagents to optimize multiple rounds in parallel. "
                    "Do not treat the next round as a parameter-only tuning sweep.",
                    result.issues,
                ),
                next_option=next_round_name,
            )

    return result


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


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("round-state analysis_comparison_sources must be a list of strings")
    raw_items = cast(list[Any], value)
    items: list[str] = []
    for item in raw_items:
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


def _find_baseline_operator_snapshot(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    candidates = [
        path
        for path in sorted(root.iterdir())
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


def _is_batch_optimize_operator_candidate(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix != ".py":
        return False
    if path.name in _BATCH_OPTIMIZE_EXCLUDED_NAMES:
        return False
    return not path.name.startswith(_BATCH_OPTIMIZE_EXCLUDED_PREFIXES)


def _resolve_batch_optimize_operator_file(workspace: Path) -> Path:
    candidates = [
        path for path in sorted(workspace.iterdir()) if _is_batch_optimize_operator_candidate(path)
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"no candidate operator file found in workspace: {workspace}")
    raise ValueError(f"multiple candidate operator files found in workspace: {workspace}")


def expected_round_operator_name(workspace: Path) -> str:
    operator_file = _resolve_batch_optimize_operator_file(workspace)
    return f"opt_{operator_file.name}"


def expected_round_perf_name(workspace: Path) -> str:
    operator_file = _resolve_batch_optimize_operator_file(workspace)
    return f"opt_{operator_file.stem}_perf.txt"


def resolve_round_perf_file(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    try:
        perf_name = expected_round_perf_name(workspace)
    except ValueError:
        perf_name = None
    if perf_name is not None:
        perf_path = round_dir / perf_name
        if perf_path.is_file():
            return perf_path

    perf_txt = round_dir / "perf.txt"
    if perf_txt.is_file():
        return perf_txt

    perf_files = sorted(path for path in round_dir.glob("*_perf.txt") if path.is_file())
    if len(perf_files) == 1:
        return perf_files[0]
    return None


def resolve_round_operator_file(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    try:
        operator_name = expected_round_operator_name(workspace)
    except ValueError:
        operator_name = None
    if operator_name is not None:
        operator_path = round_dir / operator_name
        if operator_path.is_file():
            return operator_path

    try:
        legacy_operator_name = _resolve_batch_optimize_operator_file(workspace).name
    except ValueError:
        legacy_operator_name = None
    if legacy_operator_name is not None:
        legacy_operator_path = round_dir / legacy_operator_name
        if legacy_operator_path.is_file():
            return legacy_operator_path

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


def _expected_round_artifact_names(workspace: Path) -> tuple[str, str]:
    try:
        return expected_round_operator_name(workspace), expected_round_perf_name(workspace)
    except ValueError:
        return "opt_<operator>.py", "opt_<operator>_perf.txt"


def _missing_issue(relative_path: str | None, *, default_path: str) -> str:
    if relative_path is None:
        return f"missing {default_path}"
    return f"missing {relative_path}"


def _build_result(
    *,
    kind: Literal["baseline", "round"],
    decision: Literal["pass", "revise-required", "hard-fail"],
    issues: tuple[str, ...],
    summary: str | None = None,
    next_option: str | None = None,
) -> OptimizeCheckResult:
    ok = decision == "pass"
    if summary is None:
        summary = (
            _append_pass_issues_to_summary(f"{kind} check passed", issues)
            if ok
            else f"{kind} check requires fixes: {'; '.join(issues)}"
        )
    return OptimizeCheckResult(
        ok=ok,
        kind=kind,
        decision=decision,
        issues=issues,
        summary=summary,
        next_option=next_option,
    )


def _append_pass_issues_to_summary(summary: str, issues: tuple[str, ...]) -> str:
    if not issues:
        return summary
    return f"{summary} Notes: {'; '.join(issues)}"
