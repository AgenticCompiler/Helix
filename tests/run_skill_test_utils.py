from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.skills.loader import load_operator_eval_script_module


def load_test_runner_module():
    return load_operator_eval_script_module("test_runner")


def load_bench_runner_module():
    return load_operator_eval_script_module("bench_runner")


def load_perf_artifacts_module():
    return load_operator_eval_script_module("perf_artifacts")


def load_profile_runner_module():
    return load_operator_eval_script_module("profile_runner")


def load_probe_runner_module():
    return load_operator_eval_script_module("probe_runner")


def load_simulator_runner_module():
    return load_operator_eval_script_module("simulator_runner")


def load_compare_result_module():
    return load_operator_eval_script_module("compare_result")


def load_npu_compare_module():
    return load_operator_eval_script_module("npu_compare")


def load_bench_runtime_module():
    return load_operator_eval_script_module("bench_runtime")


def load_profile_csv_parser_module():
    return load_operator_eval_script_module("profile_csv_parser")


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
