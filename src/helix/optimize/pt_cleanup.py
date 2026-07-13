from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from helix.optimize.skill_contract import optimize_state_round_module

_OPTIMIZE_ROUND_MODULE = optimize_state_round_module()

cleanup_dir_pt_files = _OPTIMIZE_ROUND_MODULE.cleanup_dir_pt_files  # type: ignore[reportUnknownVariableType]
cleanup_pt_file = _OPTIMIZE_ROUND_MODULE.cleanup_pt_file  # type: ignore[reportUnknownVariableType]
ordinary_optimize_pt_cleanup_enabled = _OPTIMIZE_ROUND_MODULE.ordinary_optimize_pt_cleanup_enabled  # type: ignore[reportUnknownVariableType]
ordinary_optimize_pt_cleanup_mode = _OPTIMIZE_ROUND_MODULE.ordinary_optimize_pt_cleanup_mode  # type: ignore[reportUnknownVariableType]
OptimizePtCleanupTrigger = Literal["round", "run-test"]


def cleanup_workspace_pt_files(workdir: Path, *, trigger: OptimizePtCleanupTrigger = "round") -> list[str]:
    if ordinary_optimize_pt_cleanup_mode() != trigger:
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


def cleanup_run_test_pt_files(paths: Iterable[Path | None]) -> list[str]:
    if ordinary_optimize_pt_cleanup_mode() != "run-test":
        return []
    cleaned: list[str] = []
    seen: set[Path] = set()
    for path in paths:
        if path is None:
            continue
        resolved_path = path.resolve()
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        cleaned_name = cleanup_pt_file(resolved_path)
        if cleaned_name is not None:
            cleaned.append(str(resolved_path))
    return cleaned
