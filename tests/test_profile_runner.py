import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import load_profile_runner_module, make_skill_result


class ProfileRunnerTests(unittest.TestCase):
    def test_run_local_profile_bench_standalone_uses_case_id_runtime(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            bench_file.write_text("# bench-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "profile_local_standalone_case",
                create=True,
                return_value=make_skill_result(0, "profile stdout\n", ""),
            ) as helper, patch.object(
                module,
                "_resolve_local_profile_dir",
                return_value=profile_dir,
            ):
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        helper.assert_called_once_with(bench_file, operator_file, "case-b")

    def test_run_local_profile_bench_msprof_profiles_selected_case_without_kernel_filter(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")
            bench_file.write_text(
                """# bench-mode: msprof
# api-name: kernel
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

            profile_result = make_skill_result(0, "profile stdout\n", "")
            with patch.object(
                module,
                "run_streaming_process",
                return_value=profile_result,
            ) as mocked:
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    case_id="case-b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "msprof",
                sys.executable,
                "bench_runtime.py",
                "run-one",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "case-b",
            ],
        )

    def test_run_local_profile_bench_msprof_requires_case_id_when_multiple_cases_exist(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernels: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "_load_bench_runtime_module",
            ) as load_runtime:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [
                                    type("_Case", (), {"case_id": "case-a"})(),
                                    type("_Case", (), {"case_id": "case-b"})(),
                                ],
                                None,
                            )
                        ),
                    },
                )()
                with self.assertRaisesRegex(ValueError, "requires --case-id when multiple cases exist"):
                    module.run_local_profile_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                    )

    def test_run_local_profile_bench_msprof_defaults_to_single_case(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\n# kernels: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            profile_result = make_skill_result(0, "profile stdout\n", "")
            with patch.object(
                module,
                "run_streaming_process",
                return_value=profile_result,
            ) as mocked, patch.object(
                module,
                "_load_bench_runtime_module",
            ) as load_runtime:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [type("_Case", (), {"case_id": "only-case"})()],
                                None,
                            )
                        ),
                    },
                )()
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "msprof",
                sys.executable,
                "bench_runtime.py",
                "run-one",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "only-case",
            ],
        )

    def test_run_local_profile_bench_msprof_ignores_explicit_kernel_name(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")
            bench_file.write_text(
                """# bench-mode: msprof
# api-name: kernel
# api-kind: torch-function
# kernels: kernel_a, kernel_b

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

            profile_result = make_skill_result(0, "profile stdout\n", "")
            with patch.object(
                module,
                "run_streaming_process",
                return_value=profile_result,
            ) as mocked:
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    case_id="case-b",
                    kernel_name="kernel_b",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "msprof",
                sys.executable,
                "bench_runtime.py",
                "run-one",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "case-b",
            ],
        )

    def test_run_local_profile_bench_msprof_allows_multi_kernel_metadata_without_kernel_name(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")
            bench_file.write_text(
                "# bench-mode: msprof\n# kernels: kernel_a, kernel_b\n",
                encoding="utf-8",
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            profile_result = make_skill_result(0, "profile stdout\n", "")
            with patch.object(
                module,
                "run_streaming_process",
                return_value=profile_result,
            ) as mocked, patch.object(
                module,
                "_load_bench_runtime_module",
            ) as load_runtime:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [type("_Case", (), {"case_id": "only-case"})()],
                                None,
                            )
                        ),
                    },
                )()
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "msprof",
                sys.executable,
                "bench_runtime.py",
                "run-one",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "only-case",
            ],
        )

    def test_run_remote_profile_bench_msprof_profiles_selected_case_without_kernel_filter(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# kernels: kernel_a, kernel_b\n",
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
                "_load_bench_runtime_module",
            ) as load_runtime, patch.object(
                module,
                "run_remote_command_buffered",
                return_value=make_skill_result(0, "PROF_remote\n", ""),
            ), patch.object(
                module,
                "copy_directory_from_remote",
                side_effect=_fake_copy,
            ), patch.object(module, "cleanup_remote_workspace") as cleanup:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [
                                    type("_Case", (), {"case_id": "case-a"})(),
                                    type("_Case", (), {"case_id": "case-b"})(),
                                ],
                                None,
                            )
                        ),
                        "runtime_support_paths": staticmethod(
                            lambda: [root / "bench_runtime.py", root / "result_payload.py"]
                        ),
                    },
                )()
                result, resolved_profile_dir, remote_workspace = module.run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    "alice@example.com",
                    None,
                    case_id="case-b",
                    kernel_name="kernel_b",
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, copied_profile_dir)
        self.assertEqual(remote_workspace, "/tmp/remote-profile")
        self.assertEqual(
            remote_run.call_args.args[2],
            [
                "msprof",
                "python3",
                "bench_runtime.py",
                "run-one",
                "--bench-file",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--case-id",
                "case-b",
            ],
        )
        cleanup.assert_not_called()

    def test_run_remote_profile_bench_standalone_uses_case_id_runtime_helper(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: standalone\n", encoding="utf-8")
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
                return_value=make_skill_result(0, "PROF_remote\n", ""),
            ), patch.object(
                module,
                "copy_directory_from_remote",
                side_effect=_fake_copy,
            ) as copy_back, patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, resolved_profile_dir, remote_workspace = module.run_remote_profile_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                    case_id="case-b",
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, copied_profile_dir)
        self.assertEqual(remote_workspace, "/tmp/remote-profile")
        copy_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
        self.assertIn("bench_runtime.py", copy_targets)
        remote_command = remote_run.call_args.args[2]
        self.assertEqual(remote_command[0:2], ["python3", "-c"])
        self.assertIn("profile_local_bench_case", remote_command[2])
        self.assertEqual(remote_command[3:], ["bench_kernel.py", "kernel.py", "case-b"])
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-profile/PROF_remote",
            copied_profile_dir,
            verbose=False,
            stderr=None,
        )
        cleanup.assert_not_called()
