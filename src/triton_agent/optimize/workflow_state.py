from __future__ import annotations

from pathlib import Path

from triton_agent.skill_loader import load_skill_script_module


def _state_machine_module():
    return load_skill_script_module("ascend-npu-optimize-state", "state_manage/state_machine")


def bootstrap_optimize_workflow_state(
    state_path: Path,
    *,
    run_id: str,
    source_operator: Path,
    baseline_reused: bool,
) -> None:
    module = _state_machine_module()
    module.bootstrap_state(
        state_path,
        run_id=run_id,
        source_operator=_workspace_relative_source_operator(state_path, source_operator),
        baseline_reused=baseline_reused,
    )


def mark_baseline_passed_in_workflow_state(state_path: Path | None) -> None:
    if state_path is None or not state_path.exists():
        return
    _state_machine_module().mark_baseline_passed(state_path)


def render_optimize_phase_summary(state_path: Path | None) -> str | None:
    if state_path is None or not state_path.exists():
        return None
    return str(_state_machine_module().render_phase_summary(state_path))


def archive_round_timings_from_state(state_path: Path | None, archive_path: Path) -> bool:
    if state_path is None or not state_path.exists():
        return False
    return bool(_state_machine_module().write_round_timings_archive(state_path, archive_path))


def _workspace_relative_source_operator(state_path: Path, source_operator: Path) -> str:
    workspace_root = state_path.parent.parent
    try:
        return source_operator.resolve().relative_to(workspace_root.resolve()).as_posix()
    except ValueError:
        return source_operator.name
