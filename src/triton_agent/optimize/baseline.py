from __future__ import annotations

from pathlib import Path
from typing import Any

from hook_runtime.optimize import baseline as _runtime_baseline


def baseline_dir(workspace: Path) -> Path:
    return _runtime_baseline.baseline_dir(workspace)


def load_baseline_state(workspace: Path) -> Any:
    return _runtime_baseline.load_baseline_state(workspace)


def inspect_baseline_artifacts(workspace: Path) -> Any:
    return _runtime_baseline.inspect_baseline_artifacts(workspace)


def baseline_gate_issues(workspace: Path) -> tuple[str, ...]:
    return _runtime_baseline.baseline_gate_issues(workspace)
