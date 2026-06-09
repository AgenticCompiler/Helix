from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "references" / "contract.json"
CONTRACT_DATA = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
BASELINE_STATE_FIELDS = {
    str(field_name): str(description)
    for field_name, description in CONTRACT_DATA["baseline_state_fields"].items()
}
BASELINE_STATE_REQUIRED_FIELDS = tuple(BASELINE_STATE_FIELDS)

_BASELINE_METADATA_FILENAMES = {
    "state.json",
    "perf.txt",
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


def check_baseline(baseline_dir_path: Path) -> OptimizeCheckResult:
    issues = baseline_gate_issues(baseline_dir_path.parent)
    if issues:
        return _build_result(
            status="fail",
            issues=issues,
        )
    return _build_result(status="pass", issues=())


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


def _declared_workspace_file(workspace: Path, relative_path: str | None) -> Path | None:
    if relative_path is None:
        return None
    return _existing_file(workspace / Path(relative_path))


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


def _missing_issue(relative_path: str | None, *, default_path: str) -> str:
    if relative_path is None:
        return f"missing {default_path}"
    return f"missing {relative_path}"


def _build_result(
    *,
    status: Literal["pass", "fail"],
    issues: tuple[str, ...],
    summary: str | None = None,
) -> OptimizeCheckResult:
    if summary is None:
        summary = (
            "baseline check passed"
            if status == "pass"
            else f"baseline check requires fixes: {'; '.join(issues)}"
        )
    return OptimizeCheckResult(
        kind="baseline",
        status=status,
        issues=issues,
        summary=summary,
        next_option=None,
    )
