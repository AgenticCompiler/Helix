from __future__ import annotations

from pathlib import Path

from helix.skill_bridges import optimize_state


def cleanup_optimize_workspace_profile_artifacts(workdir: Path) -> list[str]:
    return optimize_state.cleanup_workspace_profile_artifacts(workdir)
