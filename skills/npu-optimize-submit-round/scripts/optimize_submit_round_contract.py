from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from kernel_continuity_check import analyze_kernel_continuity
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

_SKILLS_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_CONTRACT_PATH = (
    _SKILLS_ROOT / "npu-optimize-submit-baseline" / "references" / "contract.json"
)
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "references" / "contract.json"
_BASELINE_CONTRACT_DATA = json.loads(_BASELINE_CONTRACT_PATH.read_text(encoding="utf-8"))
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
BASELINE_STATE_REQUIRED_FIELDS = tuple(_BASELINE_CONTRACT_DATA["baseline_state_fields"])
ROUND_STATE_REQUIRED_FIELDS = tuple(CONTRACT_DATA["round_state_required_fields"])
ROUND_STATE_OPTIONAL_FIELDS = {
    str(field_name): str(description)
    for field_name, description in CONTRACT_DATA["round_state_optional_fields"].items()
}

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
    kind: Literal["baseline", "round"]
    status: Literal["pass", "fail"]
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
    analysis_skipped_reason: str | None = None
    profile_dir: str | None = None
    ir_dir: str | None = None
    perf_analysis_path: str | None = None


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
        perf_path = _declared_state_file(root, workspace, declared_perf)
        operator_path = _declared_state_file(root, workspace, declared_operator)

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
        analysis_skipped_reason=_optional_str(data.get("analysis_skipped_reason")),
        profile_dir=_optional_str(data.get("profile_dir")),
        ir_dir=_optional_str(data.get("ir_dir")),
        perf_analysis_path=_optional_str(data.get("perf_analysis_path")),
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
        summary_path = _declared_state_file(round_dir, workspace, declared_summary)
        perf_path = _declared_state_file(round_dir, workspace, declared_perf)
        perf_analysis_path = _declared_state_file(round_dir, workspace, declared_analysis)

    if summary_path is None:
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
    elif state is not None and declared_summary is not None and Path(declared_summary).name != summary_path.name:
        issues.append("summary_path must be summary.md")
    if round_state_path is None:
        issues.append("missing round-state.json")
    if perf_path is None:
        issues.append(_missing_issue(declared_perf, default_path=expected_perf_name_value))
    elif state is not None and declared_perf is not None and Path(declared_perf).name != perf_path.name:
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
            status="fail",
            issues=issues,
        )
    return _build_result(kind="baseline", status="pass", issues=())


def _inspect_round_minimum_artifact_package(
    round_dir: Path,
) -> tuple[RoundArtifactsInspection, RoundState | None, str | None]:
    artifact_inspection = inspect_round_artifacts(round_dir)
    if artifact_inspection.issues:
        return artifact_inspection, None, None
    try:
        return artifact_inspection, load_round_state(round_dir), None
    except ValueError as exc:
        return artifact_inspection, None, str(exc)


def is_completed_round_directory(round_dir: Path) -> bool:
    if not round_dir.is_dir():
        return False
    name = round_dir.name
    if not name.startswith("opt-round-"):
        return False
    suffix = name[len("opt-round-"):]
    if not suffix.isdigit():
        return False

    inspection, round_state, _state_error = _inspect_round_minimum_artifact_package(round_dir)
    if inspection.issues or round_state is None:
        return False

    return (
        round_state.correctness_status == "passed"
        and round_state.benchmark_status == "passed"
    )


def iter_completed_round_directories(workspace: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(workspace.glob("opt-round-*"))
        if is_completed_round_directory(path)
    )


def check_round(
    round_dir: Path,
    *,
    current_round: int | None = None,
    final_round: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
) -> OptimizeCheckResult:
    artifact_inspection, round_state, state_error = _inspect_round_minimum_artifact_package(round_dir)
    artifact_issues = artifact_inspection.issues
    if artifact_issues:
        return _build_result(
            kind="round",
            status="fail",
            issues=artifact_issues,
        )

    if state_error is not None:
        return _build_result(
            kind="round",
            status="fail",
            issues=(state_error,),
        )
    assert round_state is not None

    if round_state.correctness_status != "passed":
        return _build_result(
            kind="round",
            status="fail",
            issues=(f"correctness_status={round_state.correctness_status}",),
        )
    if round_state.benchmark_status != "passed":
        return _build_result(
            kind="round",
            status="fail",
            issues=(f"benchmark_status={round_state.benchmark_status}",),
        )

    baseline_issues = baseline_gate_issues(round_dir.parent)
    if baseline_issues:
        return _build_result(
            kind="round",
            status="fail",
            issues=baseline_issues,
        )

    semantic_issues: list[str] = []
    baseline_perf_path: Path | None = None
    try:
        baseline = load_baseline_state(round_dir.parent)
        baseline_perf_path = _declared_state_file(
            baseline_dir(round_dir.parent),
            round_dir.parent,
            baseline.perf_artifact,
        )
        comparison_target_path = _declared_state_file(
            round_dir,
            round_dir.parent,
            round_state.comparison_target,
        )
        expected_comparison_target = None
        if baseline_perf_path is not None:
            expected_comparison_target = os.path.relpath(baseline_perf_path.resolve(), round_dir)
        if comparison_target_path is None:
            semantic_issues.append(
                _missing_issue(
                    round_state.comparison_target,
                    default_path=expected_comparison_target or round_state.comparison_target,
                )
            )
        elif (
            baseline_perf_path is not None
            and comparison_target_path.resolve() != baseline_perf_path.resolve()
        ):
            semantic_issues.append(
                f"comparison_target={round_state.comparison_target} "
                f"(expected {expected_comparison_target or baseline.perf_artifact})"
            )
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
            status="fail",
            issues=tuple(semantic_issues),
        )

    operator_path = artifact_inspection.operator_path
    if operator_path is None:
        expected_operator_name_value, _expected_perf_name_value = _expected_round_artifact_names(round_dir.parent)
        return _build_result(
            kind="round",
            status="fail",
            issues=(f"missing {expected_operator_name_value}",),
        )

    continuity = analyze_kernel_continuity(operator_path)
    if not continuity.ok:
        return _build_result(
            kind="round",
            status="fail",
            issues=((continuity.reason or "round operator failed Triton continuity check"),),
        )

    runtime_warnings: list[str] = []
    if optimize_target == "kernel" and round_state.effective_metric_source in {"total-op", "mixed"}:
        runtime_warnings.append(
            "kernel optimize target fell back to "
            f"effective_metric_source={round_state.effective_metric_source}; "
            "the round may still participate in best-round selection, but review the comparison basis."
        )

    if ordinary_optimize_pt_cleanup_enabled():
        cleanup_dir_pt_files(round_dir)
    local_optimum_warnings: tuple[str, ...] = ()
    if baseline_perf_path is not None:
        local_optimum_warnings = collect_local_optimum_warnings(
            round_dir,
            baseline_perf_path=baseline_perf_path,
        )
    result = _build_result(
        kind="round",
        status="pass",
        issues=(*tuple(runtime_warnings), *local_optimum_warnings),
    )

    if current_round is not None and final_round is not None:
        if current_round < final_round:
            next_round_name = f"opt-round-{current_round + 1}"
            result = _build_result(
                kind="round",
                status="pass",
                issues=result.issues,
                summary=_append_pass_issues_to_summary(
                    f"round check passed. "
                    f"Round {current_round}/{final_round} in the current worker batch is complete. "
                    f"Next round: {next_round_name}. "
                    f"Use the staged `triton-npu-optimize-start-round` skill to open {next_round_name} "
                    "before beginning the next round.",
                    result.issues,
                ),
                next_option=next_round_name,
            )
        else:
            result = _build_result(
                kind="round",
                status="pass",
                issues=result.issues,
                summary=_append_pass_issues_to_summary(
                    "round check passed. This round satisfied the current worker batch target.",
                    result.issues,
                ),
                next_option=None,
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


def _declared_state_file(state_dir: Path, workspace: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    declared_path = Path(relative_path)
    state_relative = _existing_file(state_dir / declared_path)
    if state_relative is not None:
        return state_relative
    return _existing_file(workspace / declared_path)


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
    status: Literal["pass", "fail"],
    issues: tuple[str, ...],
    summary: str | None = None,
    next_option: str | None = None,
) -> OptimizeCheckResult:
    if summary is None:
        summary = (
            _append_pass_issues_to_summary(f"{kind} check passed", issues)
            if status == "pass"
            else f"{kind} check requires fixes: {'; '.join(issues)}"
        )
    return OptimizeCheckResult(
        kind=kind,
        status=status,
        issues=issues,
        summary=summary,
        next_option=next_option,
    )


def _append_pass_issues_to_summary(summary: str, issues: tuple[str, ...]) -> str:
    if not issues:
        return summary
    return f"{summary} Notes: {'; '.join(issues)}"
