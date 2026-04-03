import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import load_profile_runner_module, make_skill_result


class ProfileRunnerTests(unittest.TestCase):
    def test_run_local_profile_bench_standalone_wraps_benchmark_with_msprof(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "run_streaming_process",
                return_value=make_skill_result(0, "profile stdout\n", ""),
            ) as mocked:
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        self.assertEqual(
            mocked.call_args.args[0],
            ["msprof", "python3", "bench_kernel.py", "--operator-file", "kernel.py"],
        )

    def test_run_local_profile_bench_msprof_requires_kernel_metadata_and_selected_case(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            profile_dir = root / "PROF_demo"
            output_dir = profile_dir / "mindstudio_profiler_output"
            output_dir.mkdir(parents=True)
            (output_dir / "op_statistic_1.csv").write_text("header\n", encoding="utf-8")
            bench_file.write_text("# bench-mode: msprof\n# kernel: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            count_result = make_skill_result(0, "3\n", "")
            profile_result = make_skill_result(0, "profile stdout\n", "")
            with patch.object(module, "run_buffered_process", return_value=count_result), patch.object(
                module,
                "run_streaming_process",
                return_value=profile_result,
            ) as mocked:
                result, resolved_profile_dir = module.run_local_profile_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    bench_case=2,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, profile_dir)
        self.assertEqual(
            mocked.call_args.args[0],
            [
                "msprof",
                "op",
                "--kernel-name=kernel_name",
                "python3",
                "bench_kernel.py",
                "--operator-file",
                "kernel.py",
                "--bench",
                "2",
            ],
        )

    def test_run_local_profile_bench_msprof_rejects_out_of_range_case(self) -> None:
        module = load_profile_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: kernel_name\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "1\n", "")):
                with self.assertRaises(ValueError):
                    module.run_local_profile_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                        bench_case=2,
                    )

    def test_run_remote_profile_bench_copies_profile_dir_back_and_keeps_workspace_when_requested(self) -> None:
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
            ), patch.object(module, "copy_file_to_remote"), patch.object(
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
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(resolved_profile_dir, copied_profile_dir)
        self.assertEqual(remote_workspace, "/tmp/remote-profile")
        self.assertEqual(
            remote_run.call_args.args[2],
            ["msprof", "python3", "bench_kernel.py", "--operator-file", "kernel.py"],
        )
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-profile/PROF_remote",
            copied_profile_dir,
            verbose=False,
            stderr=None,
        )
        cleanup.assert_not_called()
