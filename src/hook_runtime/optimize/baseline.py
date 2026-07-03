from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from hook_runtime.skill_loader import load_skill_script_module


@lru_cache(maxsize=1)
def _optimize_baseline_module():
    return load_skill_script_module(
        "ascend-npu-optimize-state",
        "baseline/check",
    )


def baseline_dir(workspace: Path) -> Path:
    return workspace / "baseline"


def load_baseline_state(workspace: Path) -> Any:
    return _optimize_baseline_module().load_baseline_state(workspace)


def inspect_baseline_artifacts(workspace: Path) -> Any:
    return _optimize_baseline_module().inspect_baseline_artifacts(workspace)


def baseline_gate_issues(workspace: Path) -> tuple[str, ...]:
    return _optimize_baseline_module().baseline_gate_issues(workspace)
