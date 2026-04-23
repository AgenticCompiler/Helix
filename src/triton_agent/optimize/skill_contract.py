from __future__ import annotations

from types import ModuleType

from triton_agent.skill_loader import load_skill_script_module


def optimize_check_module() -> ModuleType:
    return load_skill_script_module("triton-npu-optimize-check", "optimize_check")
