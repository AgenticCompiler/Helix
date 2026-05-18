import sys
import unittest
from io import StringIO
from os import environ
from pathlib import Path
from unittest.mock import patch
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.skill_loader import load_operator_eval_script_module
from tests.run_skill_test_utils import (
    load_compare_result_module,
    load_bench_runner_module,
    load_test_runner_module,
    make_skill_result,
)


class RemoteExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._bench_module = load_bench_runner_module()
        cls._monotonic_patcher = patch.object(cls._bench_module.time, "monotonic", return_value=0.0)
        cls._monotonic_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._monotonic_patcher.stop()

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

    def test_run_runtime_buffered_process_merges_extra_env(self) -> None:
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
                self.returncode = 0

            def poll(self):
                return 0

        with (
            patch.dict(environ, {"EXISTING_ENV": "base"}, clear=False),
            patch.object(module.subprocess, "Popen", return_value=_FakeProcess()) as mocked,
        ):
            module.run_buffered_process(
                ["python3", "bench.py"],
                ".",
                10,
                extra_env={"ASCEND_RT_VISIBLE_DEVICES": "4"},
            )

        self.assertEqual(mocked.call_args.kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "4")
        self.assertEqual(mocked.call_args.kwargs["env"]["EXISTING_ENV"], "base")

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

    def test_run_remote_command_streaming_prefixes_env_assignments(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with patch.object(module, "run_streaming_process", return_value=make_skill_result(0, "", "")) as mocked:
            module.run_remote_command_streaming(
                module.parse_remote_spec("alice@example.com"),
                "/tmp/workspace",
                ["python3", "bench.py"],
                extra_env={"ASCEND_RT_VISIBLE_DEVICES": "4", "TRITON_AGENT_ASSIGNED_NPU": "4"},
            )

        command = mocked.call_args.args[0]
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=4", command[-1])
        self.assertIn("TRITON_AGENT_ASSIGNED_NPU=4", command[-1])

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
        module = load_compare_result_module()
        runtime = load_operator_eval_script_module("run_runtime")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "oracle result.pt"
            new = root / "new result.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

            with patch.dict(sys.modules, {"run_runtime": runtime}, clear=False):
                with patch.object(
                    runtime,
                    "create_remote_workspace",
                    return_value=("spec", "/tmp/remote-compare"),
                ), patch.object(runtime, "copy_file_to_remote") as copy_to_remote, patch.object(
                    runtime,
                    "run_remote_command_streaming",
                    return_value=make_skill_result(0, "", ""),
                ) as remote_run, patch.object(runtime, "cleanup_remote_workspace"):
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
            / "compare_result.py",
        )
        self.assertEqual(
            remote_run.call_args.args[2],
            [
                "python3",
                "compare_result.py",
                "--oracle-result",
                "oracle result.pt",
                "--new-result",
                "new result.pt",
                "--compare-level",
                "balanced",
            ],
        )

    def test_run_remote_bench_standalone_uses_runtime_helper_and_copies_perf_back(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: k\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            local_perf_path = root / "kernel_perf.txt"

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-clean"),
            ), patch.object(module, "copy_file_to_remote") as copy_to_remote, patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "bench stdout\n", ""),
            ) as remote_run, patch.object(
                module,
                "copy_file_from_remote",
                create=True,
                side_effect=lambda _spec, _remote_path, local_path, **_kwargs: local_path.write_text(
                    "latency-case-a: 1.0\n",
                    encoding="utf-8",
                ),
            ) as copy_back, patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, perf_path, remote_workspace = module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(perf_path, local_perf_path)
        self.assertEqual(remote_workspace, "/tmp/remote-clean")
        self.assertIsNotNone(remote_run.call_args.kwargs.get("stdout"))
        copy_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
        self.assertIn("standalone_bench_runtime.py", copy_targets)
        self.assertEqual(
            remote_run.call_args.args[2],
            [
                "python3",
                "-c",
                remote_run.call_args.args[2][2],
                "bench_kernel.py",
                "kernel.py",
                "kernel_perf.txt",
            ],
        )
        self.assertIn("run_local_standalone_bench", remote_run.call_args.args[2][2])
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-clean/kernel_perf.txt",
            local_perf_path,
            verbose=False,
            stderr=None,
        )
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
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n'
                    '# resolved-kernels-case-1: KernelB\n'
                    '# kernel-source-case-1: metadata\n'
                    'latency-case-2: 5.0\n'
                    '# elapsed-seconds-case-2: 0.000000\n'
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"KernelA","avg_time_us":3.0},{"op_type":"KernelB","avg_time_us":5.0}]}\n'
                    '# resolved-kernels-case-2: KernelB\n'
                    '# kernel-source-case-2: metadata\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_continues_after_failed_case_and_persists_perf(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            removed_tmp_dirs: list[str] = []
            next_tmp_index = 0
            next_payload_index = 0
            temp_dirs = ["/tmp/msprof-case-1", "/tmp/msprof-case-2"]
            metric_payloads = [
                '{"kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelB","avg_time_us":5.0}]}\n',
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
                self.assertEqual(command[0], "msprof")
                if command[-1] == "1":
                    return make_skill_result(1, "", "case one failed\n")
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

            self.assertEqual(result["return_code"], 1)
            self.assertEqual(remote_workspace, "/tmp/remote-msprof")
            self.assertEqual(removed_tmp_dirs, temp_dirs)
            if perf_path is None:
                self.fail("expected msprof perf path even when one remote case fails")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                (
                    "latency-case-1: NA\n"
                    "# elapsed-seconds-case-1: 0.000000\n"
                    "# latency-error-case-1: msprof command failed with return code 1\n"
                    "# resolved-kernels-case-1: KernelB\n"
                    "# kernel-source-case-1: metadata\n"
                    "latency-case-2: 5.0\n"
                    "# elapsed-seconds-case-2: 0.000000\n"
                    '# raw-op-statistic-case-2: {"ops":[{"op_type":"KernelB","avg_time_us":5.0}]}\n'
                    "# resolved-kernels-case-2: KernelB\n"
                    "# kernel-source-case-2: metadata\n"
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_records_na_when_remote_kernel_row_is_missing(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: MissingKernel\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            removed_tmp_dirs: list[str] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                if command == ["python3", "bench_kernel.py", "--num-bench"]:
                    return make_skill_result(0, "1\n", "")
                if command == ["mktemp", "-d"]:
                    return make_skill_result(0, "/tmp/msprof-case-1\n", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    return make_skill_result(
                        0,
                        '{"kernel_avg_time_us":null,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                        "",
                    )
                if isinstance(command, list) and command[:2] == ["rm", "-rf"]:
                    removed_tmp_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
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
            self.assertEqual(removed_tmp_dirs, ["/tmp/msprof-case-1"])
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                (
                    'latency-case-1: NA\n'
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n'
                    '# latency-error-case-1: no resolved kernels matched op_statistic csv\n'
                    '# resolved-kernels-case-1: MissingKernel\n'
                    '# kernel-source-case-1: metadata\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_sums_multiple_declared_kernels(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernels: KernelA, KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            removed_tmp_dirs: list[str] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                if command == ["python3", "bench_kernel.py", "--num-bench"]:
                    return make_skill_result(0, "1\n", "")
                if command == ["mktemp", "-d"]:
                    return make_skill_result(0, "/tmp/msprof-case-1\n", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    return make_skill_result(
                        0,
                        '{"kernel_avg_time_us":4.0,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                        "",
                    )
                if isinstance(command, list) and command[:2] == ["rm", "-rf"]:
                    removed_tmp_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                self.assertEqual(command[0], "msprof")
                self.assertTrue(command[1].startswith("--output="))
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
            self.assertEqual(removed_tmp_dirs, ["/tmp/msprof-case-1"])
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                (
                    'latency-case-1: 4.0\n'
                    '# elapsed-seconds-case-1: 0.000000\n'
                    '# raw-op-statistic-case-1: {"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}]}\n'
                    '# resolved-kernels-case-1: KernelA,KernelB\n'
                    '# kernel-source-case-1: metadata\n'
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
                return_value=make_skill_result(0, "bench stdout\n", ""),
            ) as remote_run, patch.object(
                module,
                "copy_file_from_remote",
                create=True,
                side_effect=lambda _spec, _remote_path, local_path, **_kwargs: local_path.write_text(
                    "latency-case-a: 1.0\n",
                    encoding="utf-8",
                ),
            ), patch.object(module, "cleanup_remote_workspace"):
                module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "standalone",
                    "alice@example.com",
                    None,
                )

        self.assertEqual(
            remote_run.call_args.args[2],
            [
                "python3",
                "-c",
                remote_run.call_args.args[2][2],
                "bench kernel.py",
                "kernel op.py",
                "kernel op_perf.txt",
            ],
        )
        self.assertIn("run_local_standalone_bench", remote_run.call_args.args[2][2])

    def test_run_remote_bench_msprof_elapsed_seconds_in_perf_output_success(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: OpB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            buffered_payloads = ["1\n", "/tmp/msprof-case-1\n", '{"kernel_avg_time_us":3.0,"ops":[{"op_type":"OpB","avg_time_us":3.0}]}\n']

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                if command == ["python3", "bench_abs.py", "--num-bench"]:
                    return make_skill_result(0, buffered_payloads.pop(0), "")
                if command == ["mktemp", "-d"]:
                    return make_skill_result(0, buffered_payloads.pop(0), "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    return make_skill_result(0, buffered_payloads.pop(0), "")
                if isinstance(command, list) and command[:2] == ["rm", "-rf"]:
                    return make_skill_result(0, "", "")
                return make_skill_result(1, "", "unexpected command")

            self._monotonic_patcher.stop()
            try:
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
                    return_value=make_skill_result(0, "", ""),
                ), patch.object(
                    module.time, "monotonic", side_effect=[0.0, 1.5]
                ), patch.object(module, "copy_file_from_remote"), patch.object(
                    module, "cleanup_remote_workspace"
                ):
                    result, perf_path, _ws = module.run_remote_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                        "alice@example.com",
                        None,
                    )
            finally:
                self._monotonic_patcher.start()

            self.assertEqual(result["return_code"], 0)
            if perf_path is None:
                self.fail("expected remote msprof perf path")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertIn("latency-case-1: 3.0\n", perf_text)
            self.assertIn("# elapsed-seconds-case-1: 1.500000\n", perf_text)

    def test_run_remote_bench_msprof_elapsed_seconds_in_perf_output_failure(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            bench_file.write_text(
                "# bench-mode: msprof\n# api-name: abs_\n# kernel: OpB\n",
                encoding="utf-8",
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            self._monotonic_patcher.stop()
            try:
                with patch.object(
                    module,
                    "create_remote_workspace",
                    return_value=("spec", "/tmp/remote-msprof"),
                ), patch.object(module, "copy_file_to_remote"), patch.object(
                    module,
                    "run_remote_command_buffered",
                    return_value=make_skill_result(0, "1\n", ""),
                ), patch.object(
                    module,
                    "run_remote_command_streaming",
                    return_value=make_skill_result(1, "", "command failed"),
                ), patch.object(
                    module.time, "monotonic", side_effect=[0.0, 2.5]
                ), patch.object(module, "copy_file_from_remote"), patch.object(
                    module, "cleanup_remote_workspace"
                ):
                    result, perf_path, _ws = module.run_remote_bench(
                        bench_file,
                        operator_file,
                        "msprof",
                        "alice@example.com",
                        None,
                    )
            finally:
                self._monotonic_patcher.start()

            self.assertEqual(result["return_code"], 1)
            if perf_path is None:
                self.fail("expected remote msprof perf path for failed case")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertIn("latency-case-1: NA\n", perf_text)
            self.assertIn("# elapsed-seconds-case-1: 2.500000\n", perf_text)


if __name__ == "__main__":
    unittest.main()
