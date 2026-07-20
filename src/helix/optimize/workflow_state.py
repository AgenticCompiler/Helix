from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from helix.optimize.baseline import baseline_dir
from helix.optimize.resume import classify_optimize_workspace
from helix.skill_bridges import optimize_state


@dataclass(frozen=True)
class WorkflowStateBootstrapResult:
    mode: str
    state_path: Path


def prepare_or_restore_optimize_workflow_state(
    source_operator: Path | None,
    workdir: Path,
    *,
    state_path: Path,
    run_id: str,
) -> WorkflowStateBootstrapResult:
    if state_path.exists():
        # Validate before declaring this persisted session resumable.
        optimize_state.load_state(state_path)
        return WorkflowStateBootstrapResult("reused-existing-state", state_path)
    inspection_operator, resolution_issue = _resolve_source_operator_hint(
        source_operator, workdir
    )
    inspection = (
        classify_optimize_workspace(inspection_operator, workdir)
        if inspection_operator is not None
        else _no_session_inspection()
    )
    if resolution_issue is not None and _has_optimize_session_residue(workdir):
        raise ValueError(resolution_issue)
    if inspection.state == "resumable-session":
        optimize_state.bootstrap_state(state_path, run_id=run_id, baseline_reused=True)
        return WorkflowStateBootstrapResult(
            "rebuilt-from-durable-artifacts", state_path
        )
    if inspection.state == "partial-session":
        raise ValueError(
            f"cannot rebuild optimize workflow state from partial session: {inspection.detail}"
        )
    optimize_state.bootstrap_state(state_path, run_id=run_id, baseline_reused=False)
    return WorkflowStateBootstrapResult("bootstrapped-fresh-baseline", state_path)


def mark_baseline_passed_in_workflow_state(state_path: Path | None) -> None:
    if state_path is not None and state_path.exists():
        optimize_state.mark_baseline_passed(state_path)


def render_optimize_phase_summary(state_path: Path | None) -> str | None:
    if state_path is None or not state_path.exists():
        return None
    return optimize_state.render_phase_summary(state_path)


def archive_round_timings_from_state(
    state_path: Path | None, archive_path: Path
) -> bool:
    if state_path is None or not state_path.exists():
        return False
    timing_dir = state_path.parent / "round-timings"
    if not timing_dir.is_dir():
        return False
    shutil.copytree(timing_dir, archive_path, dirs_exist_ok=True)
    return True


def _resolve_source_operator_hint(
    source_operator: Path | None, workdir: Path
) -> tuple[Path | None, str | None]:
    if source_operator is not None:
        return source_operator, None
    try:
        state = optimize_state.load_baseline_state(workdir)
    except FileNotFoundError:
        return None, None
    except ValueError as exc:
        return None, f"cannot determine source operator from baseline/state.json: {exc}"
    declared = Path(state.source_operator)
    if declared.is_absolute():
        return declared, None
    return (baseline_dir(workdir) / declared).resolve(), None


def _has_optimize_session_residue(workdir: Path) -> bool:
    return (
        baseline_dir(workdir).exists()
        or (workdir / "opt-note.md").exists()
        or any(path.is_dir() for path in workdir.glob("opt-round-*"))
    )


def _no_session_inspection():
    @dataclass(frozen=True)
    class _Inspection:
        state: str
        detail: str | None
        test_mode: str | None
        bench_mode: str | None

    return _Inspection("no-session", None, None, None)
