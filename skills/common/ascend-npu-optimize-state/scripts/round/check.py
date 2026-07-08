from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any, Literal, cast

from baseline.check import (
    baseline_gate_issues,
    load_baseline_state,
)
from round.contract import ROUND_STATE_REQUIRED_FIELDS
from round.kernel_continuity import analyze_triton_kernel_continuity
from round.local_optimum import (
    collect_local_optimum_warnings,
    compute_round_geomean_speedup,
)
from shared.json_io import load_json_object, optional_str
from shared.models import OptimizeCheckResult, RoundArtifactsInspection, RoundState
from shared.paths import (
    baseline_dir,
    declared_state_file,
    existing_file,
    invalid_dependency_issue,
    missing_path_issue,
    noncanonical_path_issue,
    unexpected_path_name_issue,
)
from shared.results import append_pass_issues_to_summary, build_check_result
from shared.round_naming import expected_round_operator_name, expected_round_perf_name, resolve_workspace_operator_file


_OPTIMIZE_DELETE_PT_FILES_ENV = "TRITON_AGENT_OPTIMIZE_DELETE_PT_FILES"
OptimizePtCleanupMode = Literal["never", "round", "run-test"]
_PT_CLEANUP_MODES = frozenset({"never", "round", "run-test"})
_LEGACY_ROUND_CLEANUP_VALUES = frozenset({"1", "true", "yes", "on"})
_LEGACY_NEVER_CLEANUP_VALUES = frozenset({"0", "false", "no", "off"})
_ROUND_METADATA_FILENAMES = {
    "attempts.md",
    "summary.md",
    "perf.txt",
    "perf-analysis.md",
    "round-state.json",
}
_PROFILE_ARTIFACT_PREFIXES = ("PROF_", "OPPROF_")


def ordinary_optimize_pt_cleanup_mode() -> OptimizePtCleanupMode:
    raw_value = os.environ.get(_OPTIMIZE_DELETE_PT_FILES_ENV)
    if raw_value is None:
        return "round"
    value = raw_value.strip().lower()
    if value in _PT_CLEANUP_MODES:
        return cast(OptimizePtCleanupMode, value)
    if value in _LEGACY_ROUND_CLEANUP_VALUES:
        return "round"
    if value in _LEGACY_NEVER_CLEANUP_VALUES:
        return "never"
    return "round"


def ordinary_optimize_pt_cleanup_enabled() -> bool:
    return ordinary_optimize_pt_cleanup_mode() == "round"


def is_ordinary_pt_result_file(path: Path) -> bool:
    name_lower = path.name.lower()
    return name_lower == "test_result.pt" or name_lower.endswith("_result.pt")


def cleanup_pt_file(pt_file: Path) -> str | None:
    if not pt_file.is_file() or not is_ordinary_pt_result_file(pt_file):
        return None
    try:
        pt_file.unlink()
        return pt_file.name
    except OSError:
        return None


def cleanup_dir_pt_files(directory: Path) -> list[str]:
    cleaned: list[str] = []
    try:
        candidates = sorted(directory.iterdir())
    except OSError:
        return cleaned
    for pt_file in candidates:
        cleaned_name = cleanup_pt_file(pt_file)
        if cleaned_name is not None:
            cleaned.append(cleaned_name)
    return cleaned


def _is_profile_artifact_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _PROFILE_ARTIFACT_PREFIXES)


def _remove_artifact_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def cleanup_dir_prof_artifacts(
    directory: Path,
    *,
    preserve_paths: tuple[Path, ...] = (),
) -> list[str]:
    cleaned: list[str] = []
    preserved = {path.resolve(strict=False) for path in preserve_paths}
    try:
        candidates = sorted(directory.iterdir())
    except OSError:
        return cleaned
    for artifact in candidates:
        if not _is_profile_artifact_name(artifact.name):
            continue
        if artifact.resolve(strict=False) in preserved:
            continue
        try:
            _remove_artifact_path(artifact)
            cleaned.append(artifact.name)
        except OSError:
            pass
    return cleaned


def _state_relative_path(state_dir: Path, workspace: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    declared_path = Path(relative_path)
    state_relative = state_dir / declared_path
    if state_relative.exists():
        return state_relative
    workspace_relative = workspace / declared_path
    if workspace_relative.exists():
        return workspace_relative
    return None


def resolve_round_profile_dir(round_dir: Path) -> Path | None:
    workspace = round_dir.parent
    round_state_path = round_dir / "round-state.json"
    if round_state_path.is_file():
        try:
            state = load_round_state(round_dir)
        except ValueError:
            state = None
        if state is not None:
            declared_profile_dir = _state_relative_path(
                round_dir,
                workspace,
                state.profile_dir,
            )
            if declared_profile_dir is not None and declared_profile_dir.is_dir():
                return declared_profile_dir

    conventional_profile_dir = round_dir / "profile"
    if conventional_profile_dir.is_dir():
        return conventional_profile_dir
    return None


def _cleanup_profile_dir_csv_only(current_dir: Path, *, root_dir: Path) -> list[str]:
    cleaned: list[str] = []
    try:
        candidates = sorted(current_dir.iterdir())
    except OSError:
        return cleaned

    for candidate in candidates:
        try:
            if candidate.is_symlink():
                _remove_artifact_path(candidate)
                cleaned.append(str(candidate.relative_to(root_dir)))
                continue
            if candidate.is_dir():
                cleaned.extend(_cleanup_profile_dir_csv_only(candidate, root_dir=root_dir))
                try:
                    if candidate != root_dir and not any(candidate.iterdir()):
                        candidate.rmdir()
                except OSError:
                    pass
                continue
            if candidate.suffix.lower() == ".csv":
                continue
            _remove_artifact_path(candidate)
            cleaned.append(str(candidate.relative_to(root_dir)))
        except OSError:
            pass

    return cleaned


def cleanup_profile_dir_csv_only(profile_dir: Path) -> list[str]:
    if not profile_dir.is_dir():
        return []
    return _cleanup_profile_dir_csv_only(profile_dir, root_dir=profile_dir)


def cleanup_round_profile_artifacts(round_dir: Path) -> list[str]:
    cleaned: list[str] = []
    profile_dir = resolve_round_profile_dir(round_dir)
    if profile_dir is not None:
        profile_dir_label = os.path.relpath(profile_dir.resolve(), round_dir.resolve())
        for name in cleanup_profile_dir_csv_only(profile_dir):
            cleaned.append(f"{profile_dir_label}/{name}")

    preserve_paths: tuple[Path, ...] = ()
    if profile_dir is not None:
        try:
            if profile_dir.parent.resolve() == round_dir.resolve():
                preserve_paths = (profile_dir,)
        except OSError:
            preserve_paths = ()

    for name in cleanup_dir_prof_artifacts(round_dir, preserve_paths=preserve_paths):
        cleaned.append(name)
    return cleaned


def cleanup_workspace_profile_artifacts(workspace: Path) -> list[str]:
    cleaned: list[str] = []
    for round_dir in sorted(workspace.glob("opt-round-*")):
        if not round_dir.is_dir():
            continue
        for name in cleanup_round_profile_artifacts(round_dir):
            cleaned.append(f"{round_dir.name}/{name}")
    for name in cleanup_dir_prof_artifacts(workspace):
        cleaned.append(name)
    return cleaned


def _comparison_target_dependency_issue(round_dir: Path) -> str | None:
    state_path = baseline_dir(round_dir.parent) / "state.json"
    if not state_path.is_file():
        # baseline_gate_issues() already reports a missing baseline/state.json; avoid duplicating it here.
        return None
    try:
        load_baseline_state(round_dir.parent)
    except ValueError as exc:
        return invalid_dependency_issue(
            "comparison_target_path",
            "baseline/state.json",
            str(exc),
        )
    return None


def load_round_state(round_dir: Path) -> RoundState:
    round_state_path = round_dir / "round-state.json"
    data = load_json_object(round_state_path, display_name="round-state.json")
    comparison_target_path_value = data.get("comparison_target_path")
    legacy_comparison_target_value = data.get("comparison_target")
    if comparison_target_path_value is None and legacy_comparison_target_value is not None:
        comparison_target_path_value = legacy_comparison_target_value
    elif (
        comparison_target_path_value is not None
        and legacy_comparison_target_value is not None
        and str(comparison_target_path_value) != str(legacy_comparison_target_value)
    ):
        raise ValueError(
            "comparison_target_path and comparison_target disagree: "
            f"{comparison_target_path_value!r} != {legacy_comparison_target_value!r}"
        )
    missing_fields = [
        field_name
        for field_name in ROUND_STATE_REQUIRED_FIELDS
        if field_name not in data
        and not (
            field_name == "comparison_target_path"
            and comparison_target_path_value is not None
        )
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
        comparison_target_path=str(comparison_target_path_value),
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
        issues.append(
            missing_path_issue(
                "summary_path",
                declared_summary,
                expected_path="summary.md",
            )
        )
    elif state is not None and declared_summary is not None and Path(declared_summary).name != summary_path.name:
        issues.append(
            unexpected_path_name_issue(
                "summary_path",
                declared_summary,
                expected_name="summary.md",
            )
        )
    if round_state_path is None:
        issues.append("missing round-state.json")
    if perf_path is None:
        issues.append(
            missing_path_issue(
                "perf_artifact",
                declared_perf,
                expected_path=expected_perf_name_value,
            )
        )
    elif state is not None and declared_perf is not None and Path(declared_perf).name != perf_path.name:
        issues.append(
            unexpected_path_name_issue(
                "perf_artifact",
                declared_perf,
                expected_name=expected_perf_name_value,
            )
        )
    if declared_analysis is not None and perf_analysis_path is None:
        issues.append(
            missing_path_issue(
                "perf_analysis_path",
                declared_analysis,
                expected_path="perf-analysis.md",
            )
        )
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


def best_completed_round_geomean_speedup(workspace: Path) -> float | None:
    try:
        baseline = load_baseline_state(workspace)
    except ValueError:
        return None
    baseline_perf_path = declared_state_file(
        baseline_dir(workspace),
        workspace,
        baseline.perf_artifact,
    )
    if baseline_perf_path is None:
        return None

    best_speedup: float | None = None
    for round_dir in iter_completed_round_directories(workspace):
        geomean_speedup = compute_round_geomean_speedup(
            round_dir,
            baseline_perf_path=baseline_perf_path,
        )
        if geomean_speedup is None:
            continue
        if best_speedup is None or geomean_speedup > best_speedup:
            best_speedup = geomean_speedup
    return best_speedup


def _min_speedup_pending_prefix(
    *,
    best_speedup: float | None,
    min_speedup: float | None,
) -> str:
    if min_speedup is None or best_speedup is None:
        return "round check passed. "
    return (
        "round check passed. "
        f"Minimum speedup target not yet satisfied: best completed-round geomean speedup "
        f"is {best_speedup:.2f}x (target {min_speedup:.2f}x). "
    )


def check_round(
    round_dir: Path,
    *,
    current_round: int | None = None,
    final_round: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
    min_speedup: float | None = None,
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
        issues = list(baseline_issues)
        comparison_dependency_issue = _comparison_target_dependency_issue(round_dir)
        if comparison_dependency_issue is not None:
            issues.append(comparison_dependency_issue)
        return build_check_result(
            kind="round",
            status="fail",
            issues=tuple(issues),
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
            round_state.comparison_target_path,
        )
        expected_comparison_target = None
        if baseline_perf_path is not None:
            expected_comparison_target = os.path.relpath(
                baseline_perf_path.resolve(),
                round_dir.resolve(),
            )
        if comparison_target_path is None:
            semantic_issues.append(
                missing_path_issue(
                    "comparison_target_path",
                    round_state.comparison_target_path,
                    expected_path=expected_comparison_target,
                )
            )
        elif (
            baseline_perf_path is not None
            and comparison_target_path.resolve() != baseline_perf_path.resolve()
        ):
            semantic_issues.append(
                noncanonical_path_issue(
                    "comparison_target_path",
                    round_state.comparison_target_path,
                    expected_path=expected_comparison_target or baseline.perf_artifact,
                )
            )
    except ValueError as exc:
        semantic_issues.append(
            invalid_dependency_issue(
                "comparison_target_path",
                "baseline/state.json",
                str(exc),
            )
        )
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
    cleanup_round_profile_artifacts(round_dir)
    cleanup_dir_prof_artifacts(round_dir.parent)
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

    best_speedup = None
    if min_speedup is not None:
        best_speedup = best_completed_round_geomean_speedup(round_dir.parent)
    target_speedup = min_speedup

    if current_round is not None and final_round is not None:
        if best_speedup is not None and target_speedup is not None and best_speedup >= target_speedup:
            result = build_check_result(
                kind="round",
                status="pass",
                issues=result.issues,
                summary=append_pass_issues_to_summary(
                    "round check passed. "
                    f"Minimum speedup target satisfied: best completed-round geomean speedup "
                    f"is {best_speedup:.2f}x (target {target_speedup:.2f}x). "
                    "Stop the optimize session immediately instead of opening another round.",
                    result.issues,
                ),
                next_option=None,
            )
        elif current_round < final_round:
            next_round_name = f"opt-round-{current_round + 1}"
            result = build_check_result(
                kind="round",
                status="pass",
                issues=result.issues,
                summary=append_pass_issues_to_summary(
                    _min_speedup_pending_prefix(
                        best_speedup=best_speedup,
                        min_speedup=target_speedup,
                    )
                    +
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
                    _min_speedup_pending_prefix(
                        best_speedup=best_speedup,
                        min_speedup=target_speedup,
                    )
                    +
                    "This round satisfied the current worker batch target.",
                    result.issues,
                ),
                next_option=None,
            )
    elif target_speedup is not None and best_speedup is not None and best_speedup >= target_speedup:
        result = build_check_result(
            kind="round",
            status="pass",
            issues=result.issues,
            summary=append_pass_issues_to_summary(
                "round check passed. "
                f"Minimum speedup target satisfied: best completed-round geomean speedup "
                f"is {best_speedup:.2f}x (target {target_speedup:.2f}x). "
                "Stop the optimize session immediately instead of opening another round.",
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
