from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.skills.loader import load_operator_eval_script_module


def load_local_test_api_module():
    return load_operator_eval_script_module("run_test_local_api")


def load_remote_api_module():
    return load_operator_eval_script_module("run_test_remote_api")


def load_local_test_worker_module():
    return load_operator_eval_script_module("run_test_local_worker")


def load_test_execution_module():
    return load_operator_eval_script_module("run_test_execution")


def load_test_contract_module():
    return load_operator_eval_script_module("test_contract")


def load_bench_modes_module():
    return load_operator_eval_script_module("run_bench_modes")


def load_run_bench_api_module():
    return load_operator_eval_script_module("run_bench_api")


def load_local_bench_api_module():
    return load_operator_eval_script_module("run_bench_local_api")


def load_bench_remote_api_module():
    return load_operator_eval_script_module("run_bench_remote_api")


def load_remote_python_bundle_module():
    return load_operator_eval_script_module("remote_python_bundle")


def load_probe_local_api_module():
    return load_operator_eval_script_module("run_probe_local_api")


def load_probe_remote_api_module():
    return load_operator_eval_script_module("run_probe_remote_api")


def load_perf_artifacts_module():
    return load_operator_eval_script_module("perf_artifacts")


def load_profile_execution_module():
    return load_operator_eval_script_module("run_profile_execution")


def load_local_profile_api_module():
    return load_operator_eval_script_module("run_profile_local_api")


def load_remote_profile_api_module():
    return load_operator_eval_script_module("run_profile_remote_api")


def load_probe_execution_module():
    return load_operator_eval_script_module("run_probe_execution")


def load_simulator_runner_module():
    return load_operator_eval_script_module("simulator_runner")


def load_compare_result_module():
    return load_operator_eval_script_module("compare_result")


def load_npu_compare_module():
    return load_operator_eval_script_module("npu_compare")


def load_bench_execution_module():
    return load_operator_eval_script_module("run_bench_execution")


def load_run_bench_execution_module():
    return load_bench_execution_module()


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
