from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.skill_contract import optimize_check_module

_OPTIMIZE_CHECK_MODULE = optimize_check_module()

cleanup_dir_pt_files = _OPTIMIZE_CHECK_MODULE.cleanup_dir_pt_files  # type: ignore[reportUnknownVariableType]


def cleanup_workspace_pt_files(workdir: Path) -> list[str]:
    cleaned: list[str] = []
    cleaned.extend(cleanup_dir_pt_files(workdir))
    for round_dir in sorted(workdir.glob("opt-round-*")):
        if not round_dir.is_dir():
            continue
        for name in cleanup_dir_pt_files(round_dir):
            cleaned.append(f"{round_dir.name}/{name}")
    return cleaned
