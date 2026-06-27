from __future__ import annotations

from pathlib import Path

from triton_agent.optimize.models import BaselineArtifactsInspection, BaselineState
from triton_agent.optimize.skill_contract import optimize_state_baseline_module


_OPTIMIZE_BASELINE_MODULE = optimize_state_baseline_module()


def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"


def load_baseline_state(workspace: Path) -> BaselineState:
    return _OPTIMIZE_BASELINE_MODULE.load_baseline_state(workspace)


def inspect_baseline_artifacts(workspace: Path) -> BaselineArtifactsInspection:
    return _OPTIMIZE_BASELINE_MODULE.inspect_baseline_artifacts(workspace)


def baseline_gate_issues(workspace: Path) -> tuple[str, ...]:
    return _OPTIMIZE_BASELINE_MODULE.baseline_gate_issues(workspace)
