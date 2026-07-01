from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.skill_contract import optimize_state_round_module

_OPTIMIZE_ROUND_MODULE = optimize_state_round_module()

cleanup_dir_pt_files = _OPTIMIZE_ROUND_MODULE.cleanup_dir_pt_files  # type: ignore[reportUnknownVariableType]
ordinary_optimize_pt_cleanup_enabled = _OPTIMIZE_ROUND_MODULE.ordinary_optimize_pt_cleanup_enabled  # type: ignore[reportUnknownVariableType]


def cleanup_workspace_pt_files(workdir: Path) -> list[str]:
    if not ordinary_optimize_pt_cleanup_enabled():
        return []
    cleaned: list[str] = []
    cleaned.extend(cleanup_dir_pt_files(workdir))
    baseline_dir = workdir / "baseline"
    if baseline_dir.is_dir():
        for name in cleanup_dir_pt_files(baseline_dir):
            cleaned.append(f"{baseline_dir.name}/{name}")
    for round_dir in sorted(workdir.glob("opt-round-*")):
        if not round_dir.is_dir():
            continue
        for name in cleanup_dir_pt_files(round_dir):
            cleaned.append(f"{round_dir.name}/{name}")
    return cleaned
