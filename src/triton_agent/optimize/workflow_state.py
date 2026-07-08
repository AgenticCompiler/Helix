from __future__ import annotations

import logging
from pathlib import Path

from hook_runtime.optimize import workflow_state as _runtime_workflow_state
from triton_agent.optimize.stages import Stage

WorkflowStateBootstrapResult = _runtime_workflow_state.WorkflowStateBootstrapResult

logger = logging.getLogger(__name__)


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


def record_stage_addressed_in_workflow_state(
    state_path: Path | None, stage: Stage
) -> None:
    """Record that a round targeted ``stage`` (appends to stages_addressed)."""
    _runtime_workflow_state.record_stage_addressed(state_path, stage.value)


def get_stages_addressed_from_state(state_path: Path | None) -> list[Stage]:
    """Return the stages addressed in prior rounds (invalid ids dropped)."""
    raw_stages = _runtime_workflow_state.get_stages_addressed(state_path)
    addressed: list[Stage] = []
    for raw in raw_stages:
        try:
            addressed.append(Stage(raw))
        except ValueError:
            logger.warning("ignoring unknown stage id in workflow state: %r", raw)
    return addressed
