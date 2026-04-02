from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.run_skill import load_run_skill_module


def load_test_runner_module():
    return load_run_skill_module("test_runner")


def load_bench_runner_module():
    return load_run_skill_module("bench_runner")


def make_skill_result(
    return_code: int,
    stdout: str,
    stderr: str,
    *,
    stalled: bool = False,
    session_id: str | None = None,
):
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "stalled": stalled,
        "session_id": session_id,
    }
