from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from helix.skill_bridges import optimize_state
OptimizePtCleanupTrigger = Literal["round", "run-test"]


def cleanup_workspace_pt_files(workdir: Path, *, trigger: OptimizePtCleanupTrigger = "round") -> list[str]:
    if optimize_state.ordinary_optimize_pt_cleanup_mode() != trigger:
        return []
    cleaned: list[str] = []
    cleaned.extend(optimize_state.cleanup_dir_pt_files(workdir))
    baseline_dir = workdir / "baseline"
    if baseline_dir.is_dir():
        for name in optimize_state.cleanup_dir_pt_files(baseline_dir):
            cleaned.append(f"{baseline_dir.name}/{name}")
    for round_dir in sorted(workdir.glob("opt-round-*")):
        if not round_dir.is_dir():
            continue
        for name in optimize_state.cleanup_dir_pt_files(round_dir):
            cleaned.append(f"{round_dir.name}/{name}")
    return cleaned


def cleanup_run_test_pt_files(paths: Iterable[Path | None]) -> list[str]:
    if optimize_state.ordinary_optimize_pt_cleanup_mode() != "run-test":
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
        cleaned_name = optimize_state.cleanup_pt_file(resolved_path)
        if cleaned_name is not None:
            cleaned.append(str(resolved_path))
    return cleaned
