import sys
import unittest
import json
from io import StringIO
from os import environ
from pathlib import Path
from typing import Optional
from unittest.mock import patch
import tempfile
import subprocess
import shutil

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

    def test_parse_standalone_case_payload_accepts_metrics_on_python39_runtime(self) -> None:
        module = load_bench_runner_module()

        result = make_skill_result(
            0,
            (
                '{"case_label":"case-a","kernel_names":["KernelA"],"kernel_source":"metadata",'
                '"metrics":{"kernel_avg_time_us":1.0,"ops":[{"op_type":"KernelA","avg_time_us":1.0}]},'
                '"error_message":null,"case_wall_clock_seconds":0.0}\n'
            ),
            "",
        )

        record = module._standalone._parse_standalone_case_result_payload(
            result,
            case_id="case-a",
            fallback_kernel_source="metadata",
        )

        self.assertEqual(record.case_label, "case-a")
        self.assertIsNotNone(record.metrics)
        if record.metrics is None:
            self.fail("expected parsed metrics")
        self.assertEqual(record.metrics["kernel_avg_time_us"], 1.0)

    @unittest.skipIf(shutil.which("python3") is None, "requires python3")
    def test_parse_standalone_case_payload_accepts_metrics_in_system_python_subprocess(self) -> None:
        script = """
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from triton_agent.skill_loader import load_operator_eval_script_module
module = load_operator_eval_script_module("bench_runner")
result = {
    "return_code": 0,
    "stdout": '{"case_label":"case-a","kernel_names":["KernelA"],"kernel_source":"metadata","metrics":{"kernel_avg_time_us":1.0,"ops":[{"op_type":"KernelA","avg_time_us":1.0}]},"error_message":null,"case_wall_clock_seconds":0.0}\\n',
    "stderr": "",
    "stalled": False,
    "session_id": None,
}
record = module._standalone._parse_standalone_case_result_payload(
    result,
    case_id="case-a",
    fallback_kernel_source="metadata",
)
print(json.dumps({"case_label": record.case_label, "kernel_avg_time_us": record.metrics["kernel_avg_time_us"]}))
"""
        completed = subprocess.run(
            ["python3", "-c", script],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr or completed.stdout)
        parsed = json.loads(completed.stdout.strip())
        self.assertEqual(parsed["case_label"], "case-a")
        self.assertEqual(parsed["kernel_avg_time_us"], 1.0)

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
                extra_env={"ASCEND_RT_VISIBLE_DEVICES": "4"},
            )

        command = mocked.call_args.args[0]
        self.assertIn("ASCEND_RT_VISIBLE_DEVICES=4", command[-1])

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
                    '{"case_label":"case-a","kernel_names":["k"],"kernel_source":"metadata","kernel_avg_time_us":1.0,"ops":[{"op_type":"k","avg_time_us":1.0}],"total_op_avg_time_us":1.0,"error_message":null,"case_wall_clock_seconds":0.0}\n',
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

    def test_run_remote_bench_standalone_parallel_uses_isolated_case_workspaces_and_device_envs(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text("# bench-mode: standalone\n# kernel: KernelA\n", encoding="utf-8")
            operator_file.write_text("def build_api():\n    return None\n", encoding="utf-8")

            streamed_commands: list[tuple[str, Optional[str], list[str]]] = []
            buffered_case_dirs: list[str] = []
            mirrored_root = root.name

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if isinstance(command, list) and command[:3] == ["mkdir", "-p", "/tmp/remote-standalone/case-case-a"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-standalone")
                    buffered_case_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == ["mkdir", "-p", "/tmp/remote-standalone/case-case-b"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-standalone")
                    buffered_case_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == [
                    "mkdir",
                    "-p",
                    f"/tmp/remote-standalone/case-case-a/{mirrored_root}",
                ]:
                    self.assertEqual(remote_workspace, "/tmp/remote-standalone/case-case-a")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == [
                    "mkdir",
                    "-p",
                    f"/tmp/remote-standalone/case-case-b/{mirrored_root}",
                ]:
                    self.assertEqual(remote_workspace, "/tmp/remote-standalone/case-case-b")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["rm", "-rf"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-standalone")
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_copy(spec, local_path, remote_path, **kwargs):
                del spec, local_path, kwargs
                self.assertTrue(
                    remote_path.endswith("/bench_case.py")
                    or remote_path.endswith("/operator_case.py")
                    or remote_path.endswith("/standalone_bench_runtime.py")
                    or remote_path.endswith("/bench_contract.py")
                    or remote_path.endswith("/perf_artifacts.py")
                )

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                del spec
                streamed_commands.append(
                    (
                        remote_workspace,
                        (kwargs.get("extra_env") or {}).get("ASCEND_RT_VISIBLE_DEVICES"),
                        command,
                    )
                )
                case_id = command[-1]
                return make_skill_result(
                    0,
                    (
                        '{"case_label":"'
                        + case_id
                        + '","kernel_names":["KernelA"],"kernel_source":"metadata","metrics":{"kernel_avg_time_us":1.0,"ops":[{"op_type":"KernelA","avg_time_us":1.0}]},"error_message":null,"case_wall_clock_seconds":0.0}\n'
                    ),
                    "",
                )

            with patch.object(
                module,
                "_load_standalone_runtime_module",
            ) as load_runtime:
                load_runtime.return_value = type(
                    "_FakeRuntime",
                    (),
                    {
                        "load_standalone_bench_cases": staticmethod(
                            lambda *_args, **_kwargs: (
                                [
                                    type("_Case", (), {"case_id": "case-a"})(),
                                    type("_Case", (), {"case_id": "case-b"})(),
                                ],
                                type("_Resolution", (), {"kernel_names": ["KernelA"], "kernel_source": "metadata"})(),
                            )
                        ),
                        "runtime_support_paths": staticmethod(lambda: []),
                    },
                )()
                with patch.object(
                    module,
                    "create_remote_workspace",
                    return_value=("spec", "/tmp/remote-standalone"),
                ), patch.object(module, "copy_file_to_remote", side_effect=_fake_copy), patch.object(
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
                        "standalone",
                        "alice@example.com",
                        None,
                        npu_devices="0,2",
                    )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(remote_workspace, "/tmp/remote-standalone")
            self.assertEqual(
                set(buffered_case_dirs),
                {"/tmp/remote-standalone/case-case-a", "/tmp/remote-standalone/case-case-b"},
            )
            self.assertEqual(
                {remote_workspace for remote_workspace, _, _ in streamed_commands},
                {
                    f"/tmp/remote-standalone/case-case-a/{mirrored_root}",
                    f"/tmp/remote-standalone/case-case-b/{mirrored_root}",
                },
            )
            self.assertEqual({device for _, device, _ in streamed_commands}, {"0", "2"})
            self.assertEqual(
                {command[3] for _, _, command in streamed_commands},
                {"bench_case.py"},
            )
            if perf_path is None:
                self.fail("expected standalone perf path")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertLess(perf_text.index('"case_label":"case-a"'), perf_text.index('"case_label":"case-b"'))
            cleanup.assert_called_once_with("spec", "/tmp/remote-standalone", verbose=False, stderr=None)

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
                    '{"case_label":"case-1","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}],"total_op_avg_time_us":4.0,"error_message":null,"case_wall_clock_seconds":0.0}\n'
                    '{"case_label":"case-2","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelA","avg_time_us":3.0},{"op_type":"KernelB","avg_time_us":5.0}],"total_op_avg_time_us":8.0,"error_message":null,"case_wall_clock_seconds":0.0}\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_parallel_uses_isolated_case_workspaces_and_device_envs(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: msprof\n# kernel: KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            streamed_commands: list[tuple[str, Optional[str], list[str]]] = []
            buffered_case_dirs: list[str] = []
            mirrored_root = root.name
            metric_payloads = {
                f"/tmp/remote-msprof/case-1/{mirrored_root}/msprof-output": '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                f"/tmp/remote-msprof/case-2/{mirrored_root}/msprof-output": '{"kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelB","avg_time_us":5.0}]}\n',
            }

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if command == ["python3", "bench_kernel.py", "--num-bench"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "2\n", "")
                if isinstance(command, list) and command[:3] == ["mkdir", "-p", "/tmp/remote-msprof/case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    buffered_case_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == ["mkdir", "-p", "/tmp/remote-msprof/case-2"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    buffered_case_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == [
                    "mkdir",
                    "-p",
                    f"/tmp/remote-msprof/case-1/{mirrored_root}",
                ]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-1")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == [
                    "mkdir",
                    "-p",
                    f"/tmp/remote-msprof/case-2/{mirrored_root}",
                ]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-2")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    self.assertIn(
                        remote_workspace,
                        {
                            f"/tmp/remote-msprof/case-1/{mirrored_root}",
                            f"/tmp/remote-msprof/case-2/{mirrored_root}",
                        },
                    )
                    output_dir = command[3]
                    return make_skill_result(0, metric_payloads[output_dir], "")
                if isinstance(command, list) and command[:2] == ["rm", "-rf"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_copy(spec, local_path, remote_path, **kwargs):
                del spec, local_path, kwargs
                self.assertTrue(
                    remote_path.endswith("/bench_kernel.py")
                    or remote_path.endswith("/kernel.py")
                    or remote_path.endswith("/bench_kernel.json")
                )

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                del spec
                self.assertIn(
                    remote_workspace,
                    {
                        f"/tmp/remote-msprof/case-1/{mirrored_root}",
                        f"/tmp/remote-msprof/case-2/{mirrored_root}",
                    },
                )
                self.assertEqual(command[0], "msprof")
                output_dir = command[1].split("=", 1)[1]
                extra_env = kwargs.get("extra_env") or {}
                streamed_commands.append((output_dir, extra_env.get("ASCEND_RT_VISIBLE_DEVICES"), command))
                return make_skill_result(0, "profile stdout\n", "")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-msprof"),
            ), patch.object(module, "copy_file_to_remote", side_effect=_fake_copy), patch.object(
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
                    npu_devices="0,2",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(remote_workspace, "/tmp/remote-msprof")
            self.assertEqual(set(buffered_case_dirs), {"/tmp/remote-msprof/case-1", "/tmp/remote-msprof/case-2"})
            self.assertEqual({device for _, device, _ in streamed_commands}, {"0", "2"})
            self.assertEqual(
                {output_dir for output_dir, _, _ in streamed_commands},
                {
                    f"/tmp/remote-msprof/case-1/{mirrored_root}/msprof-output",
                    f"/tmp/remote-msprof/case-2/{mirrored_root}/msprof-output",
                },
            )
            if perf_path is None:
                self.fail("expected msprof perf path")
            perf_text = perf_path.read_text(encoding="utf-8")
            self.assertLess(
                perf_text.index('"case_label":"case-1"'),
                perf_text.index('"case_label":"case-2"'),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_parallel_stages_discovered_case_json_files(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_all_cases.py"
            operator_file = root / "kernel.py"
            discovered_json = root / "5_MoeInitRouting.json"
            bench_file.write_text("# bench-mode: msprof\n# kernel: KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            discovered_json.write_text('{"cases":[1]}\n', encoding="utf-8")

            copied_remote_paths: list[str] = []
            mirrored_root = root.name

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if command == ["python3", "bench_all_cases.py", "--num-bench"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "1\n", "")
                if command == ["mkdir", "-p", "/tmp/remote-msprof/case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                if command == ["mkdir", "-p", f"/tmp/remote-msprof/case-1/{mirrored_root}"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-1")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    self.assertEqual(remote_workspace, f"/tmp/remote-msprof/case-1/{mirrored_root}")
                    return make_skill_result(
                        0,
                        '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                        "",
                    )
                if command == ["rm", "-rf", "/tmp/remote-msprof/case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_copy(spec, local_path, remote_path, **kwargs):
                del spec, local_path, kwargs
                copied_remote_paths.append(remote_path)

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-msprof"),
            ), patch.object(module, "copy_file_to_remote", side_effect=_fake_copy), patch.object(
                module,
                "run_remote_command_buffered",
                side_effect=_fake_remote_buffered,
            ), patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "profile stdout\n", ""),
            ), patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, perf_path, remote_workspace = module.run_remote_bench(
                    bench_file,
                    operator_file,
                    "msprof",
                    "alice@example.com",
                    None,
                    npu_devices="1",
                )

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(remote_workspace, "/tmp/remote-msprof")
            self.assertIn(f"/tmp/remote-msprof/case-1/{mirrored_root}/bench_all_cases.py", copied_remote_paths)
            self.assertIn(f"/tmp/remote-msprof/case-1/{mirrored_root}/kernel.py", copied_remote_paths)
            self.assertIn(f"/tmp/remote-msprof/case-1/{mirrored_root}/5_MoeInitRouting.json", copied_remote_paths)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertIn('"kernel_avg_time_us":2.5', perf_path.read_text(encoding="utf-8"))
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_parallel_preserves_relative_operator_layout(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_all_cases.py"
            operator_dir = root / "opt-round-13"
            operator_dir.mkdir()
            operator_file = operator_dir / "opt_kernel.py"
            operator_json = operator_dir / "5_MoeInitRouting.json"
            discovered_json = root / "5_MoeInitRouting.json"
            bench_file.write_text("# bench-mode: msprof\n# kernel: KernelB\n", encoding="utf-8")
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            operator_json.write_text('{"from":"operator-dir"}\n', encoding="utf-8")
            discovered_json.write_text('{"cases":[1]}\n', encoding="utf-8")

            copied_remote_paths: list[str] = []
            streamed_commands: list[tuple[str, list[str]]] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if command == ["python3", "bench_all_cases.py", "--num-bench"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "1\n", "")
                if command == ["mkdir", "-p", "/tmp/remote-msprof/case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                if command == ["mkdir", "-p", f"/tmp/remote-msprof/case-1/{root.name}"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-1")
                    return make_skill_result(0, "", "")
                if command == ["mkdir", "-p", f"/tmp/remote-msprof/case-1/{root.name}/opt-round-13"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-1")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    self.assertEqual(remote_workspace, f"/tmp/remote-msprof/case-1/{root.name}")
                    return make_skill_result(
                        0,
                        '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                        "",
                    )
                if command == ["rm", "-rf", "/tmp/remote-msprof/case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                self.fail(f"unexpected buffered remote command: {command}")

            def _fake_copy(spec, local_path, remote_path, **kwargs):
                del spec, local_path, kwargs
                copied_remote_paths.append(remote_path)

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                del spec, kwargs
                streamed_commands.append((remote_workspace, command))
                return make_skill_result(0, "profile stdout\n", "")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-msprof"),
            ), patch.object(module, "copy_file_to_remote", side_effect=_fake_copy), patch.object(
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
                    npu_devices="1",
                )

            mirrored_root = root.name
            self.assertEqual(result["return_code"], 0)
            self.assertEqual(remote_workspace, "/tmp/remote-msprof")
            self.assertIn(f"/tmp/remote-msprof/case-1/{mirrored_root}/bench_all_cases.py", copied_remote_paths)
            self.assertIn(
                f"/tmp/remote-msprof/case-1/{mirrored_root}/opt-round-13/opt_kernel.py",
                copied_remote_paths,
            )
            self.assertIn(
                f"/tmp/remote-msprof/case-1/{mirrored_root}/opt-round-13/5_MoeInitRouting.json",
                copied_remote_paths,
            )
            self.assertIn(f"/tmp/remote-msprof/case-1/{mirrored_root}/5_MoeInitRouting.json", copied_remote_paths)
            self.assertEqual(
                streamed_commands,
                [
                    (
                        f"/tmp/remote-msprof/case-1/{mirrored_root}",
                        [
                            "msprof",
                            f"--output=/tmp/remote-msprof/case-1/{mirrored_root}/msprof-output",
                            "python3",
                            "bench_all_cases.py",
                            "--operator-file",
                            "opt-round-13/opt_kernel.py",
                            "--bench",
                            "1",
                        ],
                    )
                ],
            )
            if perf_path is None:
                self.fail("expected msprof perf path")
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
                    '{"case_label":"case-1","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":null,"ops":null,"total_op_avg_time_us":null,"error_message":"msprof command failed with return code 1","case_wall_clock_seconds":0.0}\n'
                    '{"case_label":"case-2","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelB","avg_time_us":5.0}],"total_op_avg_time_us":5.0,"error_message":null,"case_wall_clock_seconds":0.0}\n'
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
                    '{"case_label":"case-1","kernel_names":["MissingKernel"],"kernel_source":"metadata","kernel_avg_time_us":null,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}],"total_op_avg_time_us":4.0,"error_message":"no resolved kernels matched op_statistic csv","case_wall_clock_seconds":0.0}\n'
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
                    '{"case_label":"case-1","kernel_names":["KernelA","KernelB"],"kernel_source":"metadata","kernel_avg_time_us":4.0,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}],"total_op_avg_time_us":4.0,"error_message":null,"case_wall_clock_seconds":0.0}\n'
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
                    '{"case_label":"case-a","kernel_names":["k"],"kernel_source":"metadata","kernel_avg_time_us":1.0,"ops":[{"op_type":"k","avg_time_us":1.0}],"total_op_avg_time_us":1.0,"error_message":null,"case_wall_clock_seconds":0.0}\n',
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

    def test_run_remote_bench_msprof_case_wall_clock_seconds_in_perf_output_success(self) -> None:
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
            self.assertIn('"case_label":"case-1"', perf_text)
            self.assertIn('"kernel_avg_time_us":3.0', perf_text)
            self.assertIn('"case_wall_clock_seconds":1.5', perf_text)

    def test_run_remote_bench_msprof_case_wall_clock_seconds_in_perf_output_failure(self) -> None:
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
            self.assertIn('"kernel_avg_time_us":null', perf_text)
            self.assertIn('"case_wall_clock_seconds":2.5', perf_text)


if __name__ == "__main__":
    unittest.main()
