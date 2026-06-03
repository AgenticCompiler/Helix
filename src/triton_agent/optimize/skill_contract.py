from __future__ import annotations

from types import ModuleType

from triton_agent.skill_loader import load_skill_script_module


def optimize_submit_baseline_module() -> ModuleType:
    return load_skill_script_module(
        "triton-npu-optimize-submit-baseline",
        "optimize_submit_baseline",
    )


def optimize_submit_round_module() -> ModuleType:
    return load_skill_script_module(
        "triton-npu-optimize-submit-round",
        "optimize_submit_round",
    )
