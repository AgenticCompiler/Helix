from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from baseline.check import (
    baseline_gate_issues,
    load_baseline_state,
)
from round.contract import ROUND_STATE_REQUIRED_FIELDS
from round.kernel_continuity import analyze_triton_kernel_continuity
from round.local_optimum import collect_local_optimum_warnings
from shared.json_io import load_json_object, optional_str
from shared.models import OptimizeCheckResult, RoundArtifactsInspection, RoundState
from shared.paths import baseline_dir, declared_state_file, existing_file, missing_issue
from shared.results import append_pass_issues_to_summary, build_check_result
from shared.round_naming import expected_round_operator_name, expected_round_perf_name, resolve_workspace_operator_file


_OPTIMIZE_DELETE_PT_FILES_ENV = "TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES"
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_ROUND_METADATA_FILENAMES = {
    "attempts.md",
    "summary.md",
    "perf.txt",
    "perf-analysis.md",
    "round-state.json",
}


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


def load_round_state(round_dir: Path) -> RoundState:
    round_state_path = round_dir / "round-state.json"
    data = load_json_object(round_state_path, display_name="round-state.json")
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
        analysis_skipped_reason=optional_str(data.get("analysis_skipped_reason")),
        profile_dir=optional_str(data.get("profile_dir")),
        ir_dir=optional_str(data.get("ir_dir")),
        perf_analysis_path=optional_str(data.get("perf_analysis_path")),
    )


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    workspace = round_dir.parent
    attempts_path = existing_file(round_dir / "attempts.md")
    round_state_path = existing_file(round_dir / "round-state.json")
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
        summary_path = declared_state_file(round_dir, workspace, declared_summary)
        perf_path = declared_state_file(round_dir, workspace, declared_perf)
        perf_analysis_path = declared_state_file(round_dir, workspace, declared_analysis)

    if summary_path is None:
        summary_path = existing_file(round_dir / "summary.md")
    expected_operator_name_value, expected_perf_name_value = _expected_round_artifact_names(workspace)
    if perf_path is None:
        perf_path = resolve_round_perf_file(round_dir)
    operator_path = resolve_round_operator_file(round_dir)

    issues: list[str] = []
    if attempts_path is None:
        issues.append("missing attempts.md")
    if summary_path is None:
        issues.append(missing_issue(declared_summary, default_path="summary.md"))
    elif state is not None and declared_summary is not None and Path(declared_summary).name != summary_path.name:
        issues.append("summary_path must be summary.md")
    if round_state_path is None:
        issues.append("missing round-state.json")
    if perf_path is None:
        issues.append(missing_issue(declared_perf, default_path=expected_perf_name_value))
    elif state is not None and declared_perf is not None and Path(declared_perf).name != perf_path.name:
        issues.append(f"perf_artifact must be {expected_perf_name_value}")
    if declared_analysis is not None and perf_analysis_path is None:
        issues.append(missing_issue(declared_analysis, default_path="perf-analysis.md"))
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
        return build_check_result(
            kind="round",
            status="fail",
            issues=artifact_issues,
        )

    if state_error is not None:
        return build_check_result(
            kind="round",
            status="fail",
            issues=(state_error,),
        )
    assert round_state is not None

    if round_state.correctness_status != "passed":
        return build_check_result(
            kind="round",
            status="fail",
            issues=(f"correctness_status={round_state.correctness_status}",),
        )
    if round_state.benchmark_status != "passed":
        return build_check_result(
            kind="round",
            status="fail",
            issues=(f"benchmark_status={round_state.benchmark_status}",),
        )

    baseline_issues = baseline_gate_issues(round_dir.parent)
    if baseline_issues:
        return build_check_result(
            kind="round",
            status="fail",
            issues=baseline_issues,
        )

    semantic_issues: list[str] = []
    baseline_perf_path: Path | None = None
    try:
        baseline = load_baseline_state(round_dir.parent)
        baseline_perf_path = declared_state_file(
            baseline_dir(round_dir.parent),
            round_dir.parent,
            baseline.perf_artifact,
        )
        comparison_target_path = declared_state_file(
            round_dir,
            round_dir.parent,
            round_state.comparison_target,
        )
        expected_comparison_target = None
        if baseline_perf_path is not None:
            expected_comparison_target = os.path.relpath(baseline_perf_path.resolve(), round_dir)
        if comparison_target_path is None:
            semantic_issues.append(
                missing_issue(
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
        return build_check_result(
            kind="round",
            status="fail",
            issues=tuple(semantic_issues),
        )

    operator_path = artifact_inspection.operator_path
    if operator_path is None:
        expected_operator_name_value, _expected_perf_name_value = _expected_round_artifact_names(round_dir.parent)
        return build_check_result(
            kind="round",
            status="fail",
            issues=(f"missing {expected_operator_name_value}",),
        )

    continuity = analyze_triton_kernel_continuity(operator_path)
    if not continuity.ok:
        return build_check_result(
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
    result = build_check_result(
        kind="round",
        status="pass",
        issues=(*tuple(runtime_warnings), *local_optimum_warnings),
    )

    if current_round is not None and final_round is not None:
        if current_round < final_round:
            next_round_name = f"opt-round-{current_round + 1}"
            result = build_check_result(
                kind="round",
                status="pass",
                issues=result.issues,
                summary=append_pass_issues_to_summary(
                    f"round check passed. "
                    f"Round {current_round}/{final_round} in the current worker batch is complete. "
                    f"Next round: {next_round_name}. "
                    f"Use the staged `ascend-npu-optimize-state` skill's `start-round` subcommand "
                    f"to open {next_round_name} before beginning the next round.",
                    result.issues,
                ),
                next_option=next_round_name,
            )
        else:
            result = build_check_result(
                kind="round",
                status="pass",
                issues=result.issues,
                summary=append_pass_issues_to_summary(
                    "round check passed. This round satisfied the current worker batch target.",
                    result.issues,
                ),
                next_option=None,
            )

    return result


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
        legacy_operator_name = resolve_workspace_operator_file(workspace).name
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


def _expected_round_artifact_names(workspace: Path) -> tuple[str, str]:
    try:
        return expected_round_operator_name(workspace), expected_round_perf_name(workspace)
    except ValueError:
        return "opt_<operator>.py", "opt_<operator>_perf.txt"
