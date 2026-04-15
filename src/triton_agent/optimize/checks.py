from __future__ import annotations

from triton_agent.run_skill import load_skill_script_module


def __getattr__(name: str):
    return getattr(load_skill_script_module("optimize-check", "optimize_check"), name)
