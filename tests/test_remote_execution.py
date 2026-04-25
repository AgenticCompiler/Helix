import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skill_loader import load_operator_eval_script_module
from tests.run_skill_test_utils import (
    load_bench_runner_module,
    load_test_runner_module,
    make_skill_result,
)


class RemoteExecutionTests(unittest.TestCase):
    def test_app_remote_execution_module_has_been_removed(self) -> None:
        remote_execution = Path(__file__).resolve().parents[1] / "src" / "triton_agent" / "remote_execution.py"

        self.assertFalse(remote_execution.exists())

    def test_parse_remote_spec_supports_optional_port(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        spec = module.parse_remote_spec("alice@example.com:2200")

        self.assertEqual(spec["user_host"], "alice@example.com")
        self.assertEqual(spec["port"], 2200)

    def test_parse_remote_spec_without_port(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        spec = module.parse_remote_spec("alice@example.com")

        self.assertEqual(spec["user_host"], "alice@example.com")
        self.assertIsNone(spec["port"])

    def test_parse_remote_spec_rejects_invalid_port(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with self.assertRaises(ValueError):
            module.parse_remote_spec("alice@example.com:notaport")

    def test_verbose_remote_copy_logs_scp_command(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        stderr = StringIO()
        with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")):
            module.copy_file_to_remote(
                module.parse_remote_spec("alice@example.com:2200"),
                Path("/tmp/local.txt"),
                "/tmp/remote.txt",
                verbose=True,
                stderr=stderr,
            )

        self.assertIn("[remote]", stderr.getvalue())
        self.assertIn("scp -P 2200 /tmp/local.txt alice@example.com:/tmp/remote.txt", stderr.getvalue())

    def test_run_runtime_buffered_none_returncode_defaults_to_failure(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeStdout:
            def readline(self) -> str:
                return ""

        class _FakeStderr:
            def read(self) -> str:
                return ""

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStdout()
                self.stderr = _FakeStderr()
                self.returncode = None

            def poll(self):
                return 0

        with patch.object(module.subprocess, "Popen", return_value=_FakeProcess()):
            result = module.run_buffered_process(["ssh", "alice@example.com"], ".", 10)

        self.assertEqual(result["return_code"], 1)

    def test_run_remote_command_streaming_shell_joins_sequence_args(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with patch.object(module, "run_streaming_process", return_value=make_skill_result(0, "", "")) as mocked:
            module.run_remote_command_streaming(
                module.parse_remote_spec("alice@example.com"),
                "/tmp/remote dir",
                ["python3", "test kernel.py", "--operator-file", "kernel name.py"],
            )

        command = mocked.call_args.args[0]
        self.assertIn("/tmp/remote dir", command[-1])
        self.assertIn("python3", command[-1])
        self.assertIn("test kernel.py", command[-1])
        self.assertIn("kernel name.py", command[-1])
        self.assertNotIn("[", command[-1])

    def test_run_remote_test_keeps_workspace_when_requested(self) -> None:
        module = load_test_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test_kernel.py"
            operator_file = root / "kernel.py"
            test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-keep"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "", ""),
            ), patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, archived_result, remote_workspace = module.run_remote_test(
                    test_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                    keep_remote_workdir=True,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertIsNone(archived_result)
        self.assertEqual(remote_workspace, "/tmp/remote-keep")
        cleanup.assert_not_called()

    def test_run_remote_test_quotes_filenames_with_spaces(self) -> None:
        module = load_test_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "test kernel.py"
            operator_file = root / "kernel op.py"
            test_file.write_text("# test-mode: standalone\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-space"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "", ""),
            ) as remote_run, patch.object(module, "cleanup_remote_workspace"):
                module.run_remote_test(
                    test_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                )

        self.assertEqual(
            remote_run.call_args.args[2],
            ["python3", "test kernel.py", "--operator-file", "kernel op.py"],
        )

    def test_compare_remote_result_files_quotes_filenames_with_spaces(self) -> None:
        module = load_test_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "oracle result.pt"
            new = root / "new result.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-compare"),
            ), patch.object(module, "copy_file_to_remote") as copy_to_remote, patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "", ""),
            ) as remote_run, patch.object(module, "cleanup_remote_workspace"):
                module.compare_remote_result_files(
                    oracle,
                    new,
                    "balanced",
                    "alice@example.com",
                    None,
                )

        compare_script = copy_to_remote.call_args_list[0].args[1]
        self.assertEqual(
            compare_script,
            Path(__file__).resolve().parents[1]
            / "skills"
            / "triton-npu-run-eval"
            / "scripts"
            / "compare_result_payloads.py",
        )
        self.assertEqual(
            remote_run.call_args.args[2],
            [
                "python3",
                "compare_result_payloads.py",
                "--oracle-result",
                "oracle result.pt",
                "--new-result",
                "new result.pt",
                "--compare-level",
                "balanced",
            ],
        )

    def test_run_remote_bench_cleans_workspace_by_default(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: k\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-clean"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "latency-a: 1.0\n", ""),
            ), patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, perf_path, remote_workspace = module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertIsNotNone(perf_path)
        self.assertEqual(remote_workspace, "/tmp/remote-clean")
        cleanup.assert_called_once_with("spec", "/tmp/remote-clean", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_sums_avg_time_from_remote_csv_and_cleans_profiler_tmpdirs(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            issued_tmp_dirs: list[str] = []
            removed_tmp_dirs: list[str] = []
            next_tmp_index = 0
            next_payload_index = 0
            temp_dirs = ["/tmp/msprof-case-1", "/tmp/msprof-case-2"]
            metric_payloads = [
                '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                '{"kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelA","avg_time_us":3.0},{"op_type":"KernelB","avg_time_us":5.0}]}\n',
            ]

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                nonlocal next_tmp_index, next_payload_index
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                if command == ["python3", "bench_kernel.py", "--num-bench"]:
                    return make_skill_result(0, "2\n", "")
                if command == ["mktemp", "-d"]:
                    value = temp_dirs[next_tmp_index]
                    next_tmp_index += 1
                    return make_skill_result(0, f"{value}\n", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    value = metric_payloads[next_payload_index]
                    next_payload_index += 1
                    return make_skill_result(0, value, "")
                if isinstance(command, list) and command[:2] == ["rm", "-rf"]:
                    removed_tmp_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
                self.assertEqual(command[2:4], ["python3", "bench_kernel.py"])
                issued_tmp_dirs.append(command[1].split("=", 1)[1])
                return make_skill_result(0, "profile stdout\n", "")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-msprof"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_buffered",
                side_effect=_fake_remote_buffered,
            ), patch.object(
                module,
                "run_remote_command_streaming",
                side_effect=_fake_remote_streaming,
            ), patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, perf_path, remote_workspace = module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    "alice@example.com",
                    None,
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(remote_workspace, "/tmp/remote-msprof")
            self.assertEqual(issued_tmp_dirs, temp_dirs)
            self.assertEqual(removed_tmp_dirs, temp_dirs)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                (
                    'latency-case-1: 2.5\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n'
                    'latency-case-2: 5.0\n'
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"KernelA","avg_time_us":3.0},{"op_type":"KernelB","avg_time_us":5.0}]}\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_quotes_filenames_with_spaces(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench kernel.py"
            operator_file = root / "kernel op.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: k\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-bench-space"),
            ), patch.object(module, "copy_file_to_remote"), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "latency-a: 1.0\n", ""),
            ) as remote_run, patch.object(module, "cleanup_remote_workspace"):
                module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                )

        self.assertEqual(
            remote_run.call_args.args[2],
            ["python3", "bench kernel.py", "--operator-file", "kernel op.py"],
        )


if __name__ == "__main__":
    unittest.main()
