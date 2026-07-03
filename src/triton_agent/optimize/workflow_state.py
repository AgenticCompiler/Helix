from __future__ import annotations

from pathlib import Path

from hook_runtime.optimize import workflow_state as _runtime_workflow_state

WorkflowStateBootstrapResult = _runtime_workflow_state.WorkflowStateBootstrapResult


def bootstrap_optimize_workflow_state(
    state_path: Path,
    *,
    run_id: str,
    baseline_reused: bool,
) -> None:
    _runtime_workflow_state.bootstrap_optimize_workflow_state(
        state_path,
        run_id=run_id,
        baseline_reused=baseline_reused,
    )


def prepare_or_restore_optimize_workflow_state(
    source_operator: Path | None,
    workdir: Path,
    *,
    state_path: Path,
    run_id: str,
) -> WorkflowStateBootstrapResult:
    return _runtime_workflow_state.prepare_or_restore_optimize_workflow_state(
        source_operator,
        workdir,
        state_path=state_path,
        run_id=run_id,
    )


def mark_baseline_passed_in_workflow_state(state_path: Path | None) -> None:
    _runtime_workflow_state.mark_baseline_passed_in_workflow_state(state_path)


def render_optimize_phase_summary(state_path: Path | None) -> str | None:
    return _runtime_workflow_state.render_optimize_phase_summary(state_path)


def archive_round_timings_from_state(state_path: Path | None, archive_path: Path) -> bool:
    return _runtime_workflow_state.archive_round_timings_from_state(state_path, archive_path)
