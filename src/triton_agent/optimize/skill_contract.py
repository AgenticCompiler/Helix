from __future__ import annotations

from types import ModuleType

from triton_agent.skill_loader import load_skill_script_module


def optimize_state_baseline_module() -> ModuleType:
    return load_skill_script_module(
        "ascend-npu-optimize-state",
        "baseline/check",
    )


def optimize_state_round_module() -> ModuleType:
    return load_skill_script_module(
        "ascend-npu-optimize-state",
        "round/check",
    )
