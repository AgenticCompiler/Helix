import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import (
    load_profile_execution_module,
    load_local_profile_api_module,
    load_remote_profile_api_module,
    make_skill_result,
)


def _local_profile_worker_success(command, *_args, profile_dir: Path, **_kwargs):
    result_file = Path(command[command.index("--result-file") + 1])
    result_file.write_text(
        json.dumps(
            {
                "result": {
                    "return_code": 0,
                    "stdout": "profile stdout\n",
                    "stderr": "",
                    "stalled": False,
                    "session_id": None,
                },
                "profile_dir": str(profile_dir),
            }
        ),
        encoding="utf-8",
    )
    return make_skill_result(0, "", "")


class ProfileRunnerTests(unittest.TestCase):
    # ------------------------------------------------------------------
    # local profile-bench (always torch-npu-profiler)
    # ------------------------------------------------------------------

    def test_resolve_local_profile_dir_finds_nested_valid_profiler_output(self) -> None:
        module = load_profile_execution_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outer = root / "PROF_outer"
            nested = outer / "device-run" / "PROF_nested"
            output = nested / "mindstudio_profiler_output"
            output.mkdir(parents=True)
            (output / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")

            self.assertEqual(module.resolve_local_profile_dir(root), nested)

    def test_run_local_profile_bench_uses_case_id_runtime(self) -> None:
        module = load_local_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            bench_file.write_text("# api-name: kernel\n# api-kind: torch-function\n# kernels: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "run_streaming_process",
                side_effect=lambda *args, **kwargs: _local_profile_worker_success(
                    *args, profile_dir=profile_dir, **kwargs
                ),
            ):
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir.resolve())

    def test_run_local_profile_bench_labels_profile_timeout(self) -> None:
        module = load_local_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# api-name: kernel\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(module, "run_streaming_process", return_value=make_skill_result(1, "", "")) as run:
                module.run_local_profile_bench(bench_file, operator_file, case_id="case-a")

        self.assertEqual(run.call_args.kwargs["timeout_env_name"], "HELIX_PROFILE_TIMEOUT_SECONDS")

    def test_run_local_profile_bench_requires_case_id(self) -> None:
        module = load_local_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# api-name: kernel\n# api-kind: torch-function\n# kernels: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires --case-id"):
                module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                )

    def test_run_local_profile_bench_ignores_kernel_name(self) -> None:
        module = load_local_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            bench_file.write_text("# api-name: kernel\n# api-kind: torch-function\n# kernels: kernel_a, kernel_b\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "run_streaming_process",
                side_effect=lambda *args, **kwargs: _local_profile_worker_success(
                    *args, profile_dir=profile_dir, **kwargs
                ),
            ):
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    case_id="case-b",
                    kernel_name="kernel_b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir.resolve())

    def test_run_local_profile_bench_with_multi_kernel_metadata(self) -> None:
        module = load_local_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            bench_file.write_text(
                "# api-name: kernel\n# api-kind: torch-function\n# kernels: kernel_a, kernel_b\n",
                encoding="utf-8",
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "run_streaming_process",
                side_effect=lambda *args, **kwargs: _local_profile_worker_success(
                    *args, profile_dir=profile_dir, **kwargs
                ),
            ):
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    case_id="case-a",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir.resolve())

    def test_run_local_profile_bench_selects_case_by_id(self) -> None:
        module = load_local_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            bench_file.write_text(
                """# api-name: kernel
# api-kind: torch-function
# kernels: kernel_name

def build_operator_api(operator_module):
    return operator_module.kernel

def build_bench_cases():
    return [{"id": "case-a"}, {"id": "case-b"}]

def build_bench_case_fn(operator_api, case):
    return lambda: operator_api()
""",
                encoding="utf-8",
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "run_streaming_process",
                side_effect=lambda *args, **kwargs: _local_profile_worker_success(
                    *args, profile_dir=profile_dir, **kwargs
                ),
            ):
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir.resolve())

    # ------------------------------------------------------------------
    # remote profile-bench (always torch-npu-profiler)
    # ------------------------------------------------------------------

    def test_run_remote_profile_bench_uses_case_id_runtime_helper(self) -> None:
        module = load_remote_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# api-name: kernel\n# api-kind: torch-function\n# kernels: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            copied_profile_dir = root / "PROF_remote"
            def _fake_copy(_spec, _remote_path, local_path, **_kwargs):
                output_dir = local_path / "mindstudio_profiler_output"
                output_dir.mkdir(parents=True)
                (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-profile"),
            ), patch.object(module, "copy_file_to_remote") as copy_to_remote, patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "profile stdout\n", ""),
            ) as remote_run, patch.object(
                module,
                "run_remote_command_buffered",
                return_value=make_skill_result(
                    0,
                    json.dumps(
                        {
                            "result": {
                                "return_code": 0,
                                "stdout": "profile stdout\n",
                                "stderr": "",
                                "stalled": False,
                                "session_id": None,
                            },
                            "profile_name": "PROF_remote",
                        }
                    ),
                    "",
                ),
            ), patch.object(
                module,
                "copy_directory_from_remote",
                side_effect=_fake_copy,
            ) as copy_back, patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, resolved_profile_dir, remote_workspace = module.run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    "alice@example.com",
                    None,
                    case_id="case-b",
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, copied_profile_dir)
        self.assertEqual(remote_workspace, "/tmp/remote-profile")
        copy_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
        self.assertIn("run_bench_execution.py", copy_targets)
        remote_command = remote_run.call_args.args[2]
        self.assertEqual(remote_command[0:2], ["python3", "run_profile_remote_worker.py"])
        self.assertIn("--case-id", remote_command)
        self.assertIn("case-b", remote_command)
        self.assertEqual(remote_run.call_args.kwargs["timeout_env_name"], "HELIX_PROFILE_TIMEOUT_SECONDS")
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-profile/PROF_remote",
            copied_profile_dir,
            verbose=False,
            stderr=None,
        )
        cleanup.assert_not_called()

    def test_remote_profile_payload_ignores_remote_shell_preamble(self) -> None:
        module = load_remote_profile_api_module()
        payload = {
            "result": {"return_code": 0, "stdout": "", "stderr": "", "stalled": False, "session_id": None},
            "profile_name": "PROF_remote",
        }

        result, profile_name = module._read_payload(
            make_skill_result(0, "The environment has been set\n" + json.dumps(payload) + "\n", "")
        )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(profile_name, "PROF_remote")

    def test_run_remote_profile_bench_selects_case_and_ignores_kernel_name(self) -> None:
        module = load_remote_profile_api_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text(
                "# api-name: kernel\n# api-kind: torch-function\n# kernels: kernel_a, kernel_b\n",
                encoding="utf-8",
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            copied_profile_dir = root / "PROF_remote"

            def _fake_copy(_spec, _remote_path, local_path, **_kwargs):
                output_dir = local_path / "mindstudio_profiler_output"
                output_dir.mkdir(parents=True)
                (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-profile"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "profile stdout\n", ""),
            ) as remote_run, patch.object(
                module,
                "run_remote_command_buffered",
                return_value=make_skill_result(
                    0,
                    json.dumps(
                        {
                            "result": {"return_code": 0, "stdout": "", "stderr": "", "stalled": False, "session_id": None},
                            "profile_name": "PROF_remote",
                        }
                    ),
                    "",
                ),
            ), patch.object(
                module,
                "copy_directory_from_remote",
                side_effect=_fake_copy,
            ), patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, resolved_profile_dir, remote_workspace = module.run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    "alice@example.com",
                    None,
                    case_id="case-b",
                    kernel_name="kernel_b",
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, copied_profile_dir)
        self.assertEqual(remote_workspace, "/tmp/remote-profile")
        remote_command = remote_run.call_args.args[2]
        self.assertEqual(remote_command[0:2], ["python3", "run_profile_remote_worker.py"])
        self.assertIn("case-b", remote_command)
        cleanup.assert_not_called()
