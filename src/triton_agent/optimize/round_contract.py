from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.models import RoundArtifactsInspection, RoundState
from triton_agent.optimize.skill_contract import optimize_check_module


_OPTIMIZE_CHECK_MODULE = optimize_check_module()


def load_round_state(round_dir: Path) -> RoundState:
    return _OPTIMIZE_CHECK_MODULE.load_round_state(round_dir)


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    return _OPTIMIZE_CHECK_MODULE.inspect_round_artifacts(round_dir)
