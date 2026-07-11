from __future__ import annotations

from pathlib import Path

from helix.optimize.models import RoundArtifactsInspection, RoundState
from helix.optimize.skill_contract import optimize_state_round_module


_OPTIMIZE_ROUND_MODULE = optimize_state_round_module()


def load_round_state(round_dir: Path) -> RoundState:
    return _OPTIMIZE_ROUND_MODULE.load_round_state(round_dir)


def inspect_round_artifacts(round_dir: Path) -> RoundArtifactsInspection:
    return _OPTIMIZE_ROUND_MODULE.inspect_round_artifacts(round_dir)
