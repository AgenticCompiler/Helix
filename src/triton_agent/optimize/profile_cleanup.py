from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.skill_contract import optimize_state_round_module

_OPTIMIZE_ROUND_MODULE = optimize_state_round_module()

cleanup_round_profile_artifacts = _OPTIMIZE_ROUND_MODULE.cleanup_round_profile_artifacts  # type: ignore[reportUnknownVariableType]
cleanup_workspace_profile_artifacts = _OPTIMIZE_ROUND_MODULE.cleanup_workspace_profile_artifacts  # type: ignore[reportUnknownVariableType]
resolve_round_profile_dir = _OPTIMIZE_ROUND_MODULE.resolve_round_profile_dir  # type: ignore[reportUnknownVariableType]


def cleanup_optimize_workspace_profile_artifacts(workdir: Path) -> list[str]:
    return list(cleanup_workspace_profile_artifacts(workdir))
