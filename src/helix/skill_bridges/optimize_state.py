"""Typed bridge for the ascend-npu-optimize-state skill facade."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal, Protocol, cast

from helix.optimize.models import (
    BaselineArtifactsInspection,
    BaselineState,
    OptimizeCheckResult,
    RoundArtifactsInspection,
    RoundState,
)
from helix.skills.loader import load_skill_script_module


class OptimizeStateApi(Protocol):
    def check_baseline(self, baseline_dir: Path) -> object: ...
    def check_round(self, round_dir: Path, *, current_round: int | None, final_round: int | None, optimize_target: str | None, min_speedup: float | None) -> object: ...
    def load_baseline_state(self, workspace: Path) -> object: ...
    def inspect_baseline_artifacts(self, workspace: Path) -> object: ...
    def baseline_gate_issues(self, workspace: Path) -> object: ...
    def load_round_state(self, round_dir: Path) -> object: ...
    def inspect_round_artifacts(self, round_dir: Path) -> object: ...
    def iter_terminal_round_directories(self, workspace: Path) -> object: ...
    def count_terminal_round_directories(self, workspace: Path) -> object: ...
    def count_completed_round_directories(self, workspace: Path) -> object: ...
    def best_completed_round_geomean_speedup(self, workspace: Path) -> object: ...
    def resolve_round_operator_file(self, round_dir: Path) -> object: ...
    def resolve_round_perf_file(self, round_dir: Path) -> object: ...
    def ordinary_optimize_pt_cleanup_mode(self) -> object: ...
    def cleanup_pt_file(self, path: Path) -> object: ...
    def cleanup_dir_pt_files(self, directory: Path) -> object: ...
    def cleanup_workspace_profile_artifacts(self, workspace: Path) -> object: ...
    def load_state(self, state_path: Path) -> object: ...
    def bootstrap_state(self, state_path: Path, *, run_id: str, baseline_reused: bool) -> None: ...
    def mark_baseline_passed(self, state_path: Path) -> None: ...
    def render_phase_summary(self, state_path: Path) -> object: ...


@lru_cache(maxsize=1)
def _api() -> OptimizeStateApi:
    return cast(
        OptimizeStateApi,
        load_skill_script_module("ascend-npu-optimize-state", "optimize_state_api"),
    )


def check_baseline(baseline_dir: Path) -> OptimizeCheckResult:
    return _normalize_check_result(_api().check_baseline(baseline_dir))


def check_round(
    round_dir: Path,
    *,
    current_round: int | None = None,
    final_round: int | None = None,
    optimize_target: Literal["kernel", "operator"] | None = None,
    min_speedup: float | None = None,
) -> OptimizeCheckResult:
    return _normalize_check_result(
        _api().check_round(
            round_dir,
            current_round=current_round,
            final_round=final_round,
            optimize_target=optimize_target,
            min_speedup=min_speedup,
        )
    )


def load_baseline_state(workspace: Path) -> BaselineState:
    return _normalize_baseline_state(_api().load_baseline_state(workspace))


def inspect_baseline_artifacts(workspace: Path) -> BaselineArtifactsInspection:
    return _normalize_baseline_inspection(_api().inspect_baseline_artifacts(workspace))


def baseline_gate_issues(workspace: Path) -> tuple[str, ...]:
    return _normalize_issues(_api().baseline_gate_issues(workspace))


def load_round_state(round_dir: Path) -> RoundState:
    return _normalize_round_state(_api().load_round_state(round_dir))


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    return _normalize_round_inspection(_api().inspect_round_artifacts(round_dir))


def iter_terminal_round_directories(workspace: Path) -> tuple[Path, ...]:
    return _normalize_paths(_api().iter_terminal_round_directories(workspace))


def count_terminal_round_directories(workspace: Path) -> int:
    return _normalize_int(_api().count_terminal_round_directories(workspace))


def count_completed_round_directories(workspace: Path) -> int:
    return _normalize_int(_api().count_completed_round_directories(workspace))


def best_completed_round_geomean_speedup(workspace: Path) -> float | None:
    value = _api().best_completed_round_geomean_speedup(workspace)
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        raise TypeError("Optimize-state API speedup must be float-compatible")
    return float(value)


def resolve_round_operator_file(round_dir: Path) -> Path | None:
    return _normalize_optional_path(_api().resolve_round_operator_file(round_dir))


def resolve_round_perf_file(round_dir: Path) -> Path | None:
    return _normalize_optional_path(_api().resolve_round_perf_file(round_dir))


def ordinary_optimize_pt_cleanup_mode() -> str:
    return str(_api().ordinary_optimize_pt_cleanup_mode())


def cleanup_pt_file(path: Path) -> str | None:
    value = _api().cleanup_pt_file(path)
    return None if value is None else str(value)


def cleanup_dir_pt_files(directory: Path) -> list[str]:
    return [str(item) for item in _normalize_issues(_api().cleanup_dir_pt_files(directory))]


def cleanup_workspace_profile_artifacts(workspace: Path) -> list[str]:
    return [str(item) for item in _normalize_issues(_api().cleanup_workspace_profile_artifacts(workspace))]


def load_state(state_path: Path) -> dict[str, object]:
    return cast(dict[str, object], _api().load_state(state_path))


def bootstrap_state(state_path: Path, *, run_id: str, baseline_reused: bool) -> None:
    _api().bootstrap_state(state_path, run_id=run_id, baseline_reused=baseline_reused)


def mark_baseline_passed(state_path: Path) -> None:
    _api().mark_baseline_passed(state_path)


def render_phase_summary(state_path: Path) -> str:
    return str(_api().render_phase_summary(state_path))


def _normalize_check_result(value: object) -> OptimizeCheckResult:
    data = _object_data(value, ("kind", "status", "issues", "summary"))
    if not isinstance(value, Mapping):
        next_option = getattr(value, "next_option", None)
    else:
        next_option = data.get("next_option")
    kind = str(data["kind"])
    status = str(data["status"])
    if kind not in {"baseline", "round"} or status not in {"pass", "fail"}:
        raise ValueError("Optimize-state API returned an invalid check result")
    return OptimizeCheckResult(
        kind=cast(Literal["baseline", "round"], kind),
        status=cast(Literal["pass", "fail"], status),
        issues=_normalize_issues(data["issues"]),
        summary=str(data["summary"]),
        next_option=None if next_option is None else str(next_option),
    )


def _normalize_baseline_state(value: object) -> BaselineState:
    data = _object_data(value, tuple(BaselineState.__dataclass_fields__))
    return BaselineState(**cast(dict[str, Any], data))


def _normalize_baseline_inspection(value: object) -> BaselineArtifactsInspection:
    data = _object_data(value, tuple(BaselineArtifactsInspection.__dataclass_fields__))
    data["issues"] = _normalize_issues(data["issues"])
    return BaselineArtifactsInspection(**cast(dict[str, Any], data))


def _normalize_round_state(value: object) -> RoundState:
    data = _object_data(value, tuple(RoundState.__dataclass_fields__))
    data["evidence_sources"] = _normalize_issues(data["evidence_sources"])
    return RoundState(**cast(dict[str, Any], data))


def _normalize_round_inspection(value: object) -> RoundArtifactsInspection:
    data = _object_data(value, tuple(RoundArtifactsInspection.__dataclass_fields__))
    data["issues"] = _normalize_issues(data["issues"])
    return RoundArtifactsInspection(**cast(dict[str, Any], data))


def _object_data(value: object, fields: tuple[str, ...]) -> dict[str, object]:
    if isinstance(value, Mapping):
        typed_value = cast(Mapping[object, object], value)
        data: dict[str, object] = {
            str(key): item for key, item in typed_value.items()
        }
    else:
        data = {field: getattr(value, field) for field in fields}
    missing = [field for field in fields if field not in data]
    if missing:
        raise TypeError("Optimize-state API result is missing fields: " + ", ".join(missing))
    return data


def _normalize_issues(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Optimize-state API issues must be a list or tuple")
    typed_value = cast(tuple[object, ...] | list[object], value)
    return tuple(str(item) for item in typed_value)


def _normalize_paths(value: object) -> tuple[Path, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Optimize-state API paths must be a list or tuple")
    paths: list[Path] = []
    typed_value = cast(tuple[object, ...] | list[object], value)
    for item in typed_value:
        if not isinstance(item, (str, Path)):
            raise TypeError("Optimize-state API path entries must be strings or Paths")
        paths.append(Path(item))
    return tuple(paths)


def _normalize_optional_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError("Optimize-state API path must be a string or Path")
    return Path(value)


def _normalize_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("Optimize-state API counter must be an integer")
    return int(cast(int | str, value))
