from __future__ import annotations

from triton_agent.run_skill import load_run_skill_module


def __getattr__(name: str):
    return getattr(load_run_skill_module("bench_runner"), name)
