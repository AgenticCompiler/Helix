from __future__ import annotations

from types import ModuleType

from triton_agent.skills.loader import load_skill_script_module


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


def scan_kernel_issues_module() -> ModuleType:
    """Code-structure scanner skill script (emits raw issue dicts)."""
    return load_skill_script_module("triton-npu-optimize", "scan_kernel_issues")
