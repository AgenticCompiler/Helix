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
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.remote import env as remote_env_module
from triton_agent.skills.loader import load_operator_eval_script_module
from tests.run_skill_test_utils import (
    load_compare_result_module,
    load_bench_runner_module,
    load_test_runner_module,
    make_skill_result,
)


def _write_hooked_bench_file(
    path: Path,
    *,
    mode: str,
    api_name: str,
    kernel_names: tuple[str, ...],
    case_ids: tuple[str, ...] = ("case-1",),
) -> None:
    kernel_header = (
        f"# kernel: {kernel_names[0]}"
        if len(kernel_names) == 1
        else "# kernels: " + ", ".join(kernel_names)
    )
    cases_literal = ", ".join(f'{{"id": "{case_id}"}}' for case_id in case_ids)
    path.write_text(
        f"""# bench-mode: {mode}
# api-name: {api_name}
# api-kind: torch-function
{kernel_header}

def build_operator_api(operator_module):
    return getattr(operator_module, "{api_name}")

def build_bench_cases():
    return [{cases_literal}]

def build_bench_case_fn(operator_api, case):
    return lambda: operator_api(case["id"])
""",
        encoding="utf-8",
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
        self.assertIn("scp -P 2200 local.txt alice@example.com:/tmp/remote.txt", stderr.getvalue())

    def test_copy_file_to_remote_avoids_windows_drive_letter_in_scp_source_argument(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")) as mocked:
            module.copy_file_to_remote(
                module.parse_remote_spec("alice@example.com"),
                Path("D:/Project/input.py"),
                "/tmp/input.py",
            )

        self.assertEqual(
            mocked.call_args.args[0],
            ["scp", "input.py", "alice@example.com:/tmp/input.py"],
        )
        self.assertEqual(mocked.call_args.args[1], "D:/Project")

    def test_copy_file_from_remote_avoids_windows_drive_letter_in_scp_destination_argument(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")) as mocked:
            module.copy_file_from_remote(
                module.parse_remote_spec("alice@example.com"),
                "/tmp/result.pt",
                Path("D:/Project/result.pt"),
            )

        self.assertEqual(
            mocked.call_args.args[0],
            ["scp", "alice@example.com:/tmp/result.pt", "result.pt"],
        )
        self.assertEqual(mocked.call_args.args[1], "D:/Project")

    def test_remote_execution_env_prefers_explicit_flags_over_environment(self) -> None:
        remote, remote_workdir = remote_env_module.resolve_remote_execution(
            "bob@example.com",
            "/tmp/explicit",
            {
                remote_env_module.remote_target_env_name(): "alice@example.com",
                remote_env_module.remote_workdir_env_name(): "/tmp/from-env",
            },
        )

        self.assertEqual(remote, "bob@example.com")
        self.assertEqual(remote_workdir, "/tmp/explicit")

    def test_remote_execution_env_ignores_workdir_without_remote_target(self) -> None:
        remote, remote_workdir = remote_env_module.resolve_remote_execution(
            None,
            "/tmp/explicit",
            {},
        )

        self.assertIsNone(remote)
        self.assertIsNone(remote_workdir)

    def test_apply_remote_execution_env_sets_target_and_workdir(self) -> None:
        env: dict[str, str] = {}

        remote_env_module.apply_remote_execution_env(
            "alice@example.com",
            "/tmp/triton-agent",
            env,
        )

        self.assertEqual(env[remote_env_module.remote_target_env_name()], "alice@example.com")
        self.assertEqual(env[remote_env_module.remote_workdir_env_name()], "/tmp/triton-agent")

    def test_apply_remote_execution_env_clears_missing_workdir_and_remote(self) -> None:
        env = {
            remote_env_module.remote_target_env_name(): "alice@example.com",
            remote_env_module.remote_workdir_env_name(): "/tmp/old",
        }

        remote_env_module.apply_remote_execution_env("alice@example.com", None, env)
        self.assertEqual(env[remote_env_module.remote_target_env_name()], "alice@example.com")
        self.assertNotIn(remote_env_module.remote_workdir_env_name(), env)

        remote_env_module.apply_remote_execution_env(None, None, env)
        self.assertNotIn(remote_env_module.remote_target_env_name(), env)
        self.assertNotIn(remote_env_module.remote_workdir_env_name(), env)

    def test_parse_standalone_case_payload_accepts_metrics_on_python39_runtime(self) -> None:
        module = load_bench_runner_module()

        result = make_skill_result(
            0,
            (
                '{"case_label":"case-a","kernel_names":["KernelA"],"kernel_source":"metadata",'
                '"metrics":{"kernel_avg_time_us":1.0,"ops":[{"op_type":"KernelA","avg_time_us":1.0}]},'
                '"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
            ),
            "",
        )

        record = module._parse_torch_npu_profiler_case_result_payload(
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
from triton_agent.skills.loader import load_operator_eval_script_module
module = load_operator_eval_script_module("bench_runner")
result = {
    "return_code": 0,
    "stdout": '{"case_label":"case-a","kernel_names":["KernelA"],"kernel_source":"metadata","metrics":{"kernel_avg_time_us":1.0,"ops":[{"op_type":"KernelA","avg_time_us":1.0}]},"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\\n',
    "stderr": "",
    "stalled": False,
    "session_id": None,
}
record = module._parse_torch_npu_profiler_case_result_payload(
    result,
    case_id="case-a",
    fallback_kernel_source="metadata",
)
print(json.dumps({"case_label": record.case_label, "kernel_avg_time_us": record.metrics["kernel_avg_time_us"]}))
        """
        completed = subprocess.run(
            [sys.executable, "-c", script],
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

    def test_run_runtime_buffered_process_closes_pipe_streams(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeStream:
            def __init__(self, text: str = "") -> None:
                self._text = text
                self.closed = False

            def readline(self) -> str:
                return ""

            def read(self) -> str:
                text = self._text
                self._text = ""
                return text

            def close(self) -> None:
                self.closed = True

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStream()
                self.stderr = _FakeStream()
                self.returncode = 0

            def poll(self):
                return 0

        process = _FakeProcess()
        with patch.object(module.subprocess, "Popen", return_value=process):
            result = module.run_buffered_process(["python3", "bench.py"], ".", 10)

        self.assertEqual(result["return_code"], 0)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)

    def test_run_runtime_buffered_zero_timeout_disables_stall_termination(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeStdout:
            def readline(self) -> str:
                return ""

            def close(self) -> None:
                return None

        class _FakeStderr:
            def __iter__(self):
                return iter(())

            def close(self) -> None:
                return None

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStdout()
                self.stderr = _FakeStderr()
                self.returncode = 0
                self._poll_values = [None, 0]

            def poll(self):
                if self._poll_values:
                    return self._poll_values.pop(0)
                return 0

            def terminate(self) -> None:
                self.returncode = 1

        with patch.object(module.time, "monotonic", side_effect=[0.0, 1.0]), patch.object(
            module.subprocess,
            "Popen",
            return_value=_FakeProcess(),
        ):
            result = module.run_buffered_process(["python3", "bench.py"], ".", 0)

        self.assertFalse(result["stalled"])
        self.assertEqual(result["return_code"], 0)

    def test_run_runtime_buffered_process_decodes_utf8_output_from_bytes(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeStdout:
            def __init__(self) -> None:
                self._lines = [b"\xe8\xbf\x9c\xe7\xab\xaf\xe8\xbe\x93\xe5\x87\xba\n", b""]
                self.closed = False

            def readline(self) -> bytes:
                return self._lines.pop(0)

            def close(self) -> None:
                self.closed = True

        class _FakeStderr:
            def __init__(self) -> None:
                self.closed = False

            def __iter__(self):
                return iter([b"\xe6\x9d\x83\xe9\x99\x90\xe9\x94\x99\xe8\xaf\xaf\n"])

            def close(self) -> None:
                self.closed = True

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStdout()
                self.stderr = _FakeStderr()
                self.returncode = 1

            def poll(self):
                return 1

        with patch.object(module.subprocess, "Popen", return_value=_FakeProcess()):
            result = module.run_buffered_process(["ssh", "alice@example.com"], ".", 10)

        self.assertEqual(result["stdout"], "远端输出\n")
        self.assertEqual(result["stderr"], "权限错误\n")

    def test_run_runtime_streaming_windows_zero_timeout_disables_stall_termination(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeStdout:
            def read(self, _size: int) -> bytes:
                return b""

        class _FakeProcess:
            def __init__(self) -> None:
                self.stdout = _FakeStdout()
                self.returncode = 0
                self._poll_values = [None, 0]

            def poll(self):
                if self._poll_values:
                    return self._poll_values.pop(0)
                return 0

            def wait(self) -> int:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 1

        with patch.object(module.time, "monotonic", side_effect=[0.0, 1.0]), patch.object(
            module.subprocess,
            "Popen",
            return_value=_FakeProcess(),
        ):
            result = module._run_streaming_windows(["python3", "bench.py"], ".", 0)

        self.assertFalse(result["stalled"])
        self.assertEqual(result["return_code"], 0)

    def test_run_runtime_streaming_pty_zero_timeout_disables_stall_termination(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeProcess:
            def __init__(self) -> None:
                self.returncode = 0
                self._poll_values = [None, 0]

            def poll(self):
                if self._poll_values:
                    return self._poll_values.pop(0)
                return 0

            def wait(self, timeout=None) -> int:
                del timeout
                return self.returncode

            def terminate(self) -> None:
                self.returncode = 1

        fake_pty = SimpleNamespace(openpty=lambda: (11, 12))
        fake_select = SimpleNamespace(select=lambda _r, _w, _x, _t: ([], [], []))

        with patch.object(module, "pty", fake_pty), patch.object(
            module,
            "select",
            fake_select,
        ), patch.object(module.time, "monotonic", side_effect=[0.0, 1.0]), patch.object(
            module.subprocess,
            "Popen",
            return_value=_FakeProcess(),
        ), patch.object(module.os, "close"):
            result = module._run_streaming_pty(["python3", "bench.py"], ".", 0)

        self.assertFalse(result["stalled"])
        self.assertEqual(result["return_code"], 0)

    def test_eval_timeout_env_rejects_negative_values(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with patch.dict(module.os.environ, {"TRITON_AGENT_EVAL_TIMEOUT_SECONDS": "-1"}, clear=False):
            with self.assertRaises(ValueError):
                module.eval_stall_timeout_seconds()

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

    def test_run_remote_command_streaming_forces_blocks_parallel_zero_from_guarded_env(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with (
            patch.dict(environ, {"TRITON_ALL_BLOCKS_PARALLEL": "0"}, clear=False),
            patch.object(module, "run_streaming_process", return_value=make_skill_result(0, "", "")) as mocked,
        ):
            module.run_remote_command_streaming(
                module.parse_remote_spec("alice@example.com"),
                "/tmp/workspace",
                ["python3", "bench.py"],
            )

        command = mocked.call_args.args[0]
        self.assertIn("TRITON_ALL_BLOCKS_PARALLEL=0", command[-1])

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
            ["python3", "-c", remote_run.call_args.args[2][2]],
        )
        self.assertIn("test kernel.py", remote_run.call_args.args[2][2])
        self.assertIn("kernel op.py", remote_run.call_args.args[2][2])

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
                        "alice@example.com",
                        None,
                    )

        compare_script = copy_to_remote.call_args_list[0].args[1]
        self.assertEqual(
            compare_script,
            Path(__file__).resolve().parents[1]
            / "skills"
            / "common"
            / "ascend-npu-run-eval"
            / "scripts"
            / "compare_result.py",
        )
        self.assertEqual(
            remote_run.call_args.args[2],
            [
                "python3",
                "compare_result.py",
                "--ref-result",
                "oracle result.pt",
                "--new-result",
                "new result.pt",
            ],
        )

    def test_run_remote_bench_torch_npu_profiler_uses_runtime_helper_and_copies_perf_back(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            bench_file.write_text("# bench-mode: torch-npu-profiler\n# kernel: k\n", encoding="utf-8")
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
                    '{"case_label":"case-a","kernel_names":["k"],"kernel_source":"metadata","kernel_avg_time_us":1.0,"ops":[{"op_type":"k","avg_time_us":1.0}],"total_op_avg_time_us":1.0,"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n',
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
        self.assertIn("bench_runtime.py", copy_targets)
        self.assertIn("profile_csv_parser.py", copy_targets)
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
        self.assertIn("profile_all_bench_cases", remote_run.call_args.args[2][2])
        copy_back.assert_called_once_with(
            "spec",
            "/tmp/remote-clean/kernel_perf.txt",
            local_perf_path,
            verbose=False,
            stderr=None,
        )
        cleanup.assert_called_once_with("spec", "/tmp/remote-clean", verbose=False, stderr=None)

    def test_run_remote_bench_torch_npu_profiler_parallel_uses_isolated_case_workspaces_and_device_envs(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_case.py"
            operator_file = root / "operator_case.py"
            bench_file.write_text("# bench-mode: torch-npu-profiler\n# kernel: KernelA\n", encoding="utf-8")
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
                    or remote_path.endswith("/result_payload.py")
                    or remote_path.endswith("/bench_runtime.py")
                    or remote_path.endswith("/bench_contract.py")
                    or remote_path.endswith("/perf_artifacts.py")
                    or remote_path.endswith("/profile_csv_parser.py")
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
                case_id = (
                    command[command.index("--case-id") + 1]
                    if "--case-id" in command
                    else command[-2]
                )
                return make_skill_result(
                    0,
                    (
                        '{"case_label":"'
                        + case_id
                        + '","kernel_names":["KernelA"],"kernel_source":"metadata","metrics":{"kernel_avg_time_us":1.0,"ops":[{"op_type":"KernelA","avg_time_us":1.0}]},"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                    ),
                    "",
                )

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
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("KernelB",),
                case_ids=("case-1", "case-2"),
            )
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
                self.assertEqual(command[2:5], ["python3", "bench_runtime.py", "run-one"])
                issued_tmp_dirs.append(command[1].split("=", 1)[1])
                return make_skill_result(0, "profile stdout\n", "")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-msprof"),
            ), patch.object(module, "copy_file_to_remote") as copy_to_remote, patch.object(
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
            copied_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
            self.assertIn("bench_runtime.py", copied_targets)
            self.assertIn("profile_csv_parser.py", copied_targets)
            if perf_path is None:
                self.fail("expected msprof perf path")
            self.assertEqual(
                perf_path.read_text(encoding="utf-8"),
                (
                    '{"case_label":"case-1","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}],"total_op_avg_time_us":4.0,"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                    '{"case_label":"case-2","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelA","avg_time_us":3.0},{"op_type":"KernelB","avg_time_us":5.0}],"total_op_avg_time_us":8.0,"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_parallel_uses_isolated_case_workspaces_and_device_envs(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("KernelB",),
                case_ids=("case-1", "case-2"),
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            streamed_commands: list[tuple[str, Optional[str], list[str]]] = []
            buffered_case_dirs: list[str] = []
            mirrored_root = root.name
            metric_payloads = {
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/msprof-output": '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                f"/tmp/remote-msprof/case-case-2/{mirrored_root}/msprof-output": '{"kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelB","avg_time_us":5.0}]}\n',
            }

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if isinstance(command, list) and command[:3] == ["mkdir", "-p", "/tmp/remote-msprof/case-case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    buffered_case_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == ["mkdir", "-p", "/tmp/remote-msprof/case-case-2"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    buffered_case_dirs.append(command[2])
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == [
                    "mkdir",
                    "-p",
                    f"/tmp/remote-msprof/case-case-1/{mirrored_root}",
                ]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-case-1")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:3] == [
                    "mkdir",
                    "-p",
                    f"/tmp/remote-msprof/case-case-2/{mirrored_root}",
                ]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-case-2")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    self.assertIn(
                        remote_workspace,
                        {
                            f"/tmp/remote-msprof/case-case-1/{mirrored_root}",
                            f"/tmp/remote-msprof/case-case-2/{mirrored_root}",
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
                    or remote_path.endswith("/result_payload.py")
                    or remote_path.endswith("/bench_runtime.py")
                    or remote_path.endswith("/bench_contract.py")
                    or remote_path.endswith("/perf_artifacts.py")
                    or remote_path.endswith("/profile_csv_parser.py")
                )

            def _fake_remote_streaming(spec, remote_workspace, command, **kwargs):
                del spec
                self.assertIn(
                    remote_workspace,
                    {
                        f"/tmp/remote-msprof/case-case-1/{mirrored_root}",
                        f"/tmp/remote-msprof/case-case-2/{mirrored_root}",
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
            self.assertEqual(
                set(buffered_case_dirs),
                {"/tmp/remote-msprof/case-case-1", "/tmp/remote-msprof/case-case-2"},
            )
            self.assertEqual({device for _, device, _ in streamed_commands}, {"0", "2"})
            self.assertEqual(
                {output_dir for output_dir, _, _ in streamed_commands},
                {
                    f"/tmp/remote-msprof/case-case-1/{mirrored_root}/msprof-output",
                    f"/tmp/remote-msprof/case-case-2/{mirrored_root}/msprof-output",
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
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("KernelB",),
                case_ids=("case-1",),
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            discovered_json.write_text('{"cases":[1]}\n', encoding="utf-8")

            copied_remote_paths: list[str] = []
            mirrored_root = root.name
            streamed_commands: list[tuple[str, list[str]]] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if command == ["mkdir", "-p", "/tmp/remote-msprof/case-case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                if command == ["mkdir", "-p", f"/tmp/remote-msprof/case-case-1/{mirrored_root}"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-case-1")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    self.assertEqual(remote_workspace, f"/tmp/remote-msprof/case-case-1/{mirrored_root}")
                    return make_skill_result(
                        0,
                        '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                        "",
                    )
                if command == ["rm", "-rf", "/tmp/remote-msprof/case-case-1"]:
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

            self.assertEqual(result["return_code"], 0)
            self.assertEqual(remote_workspace, "/tmp/remote-msprof")
            self.assertIn(
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/bench_all_cases.py",
                copied_remote_paths,
            )
            self.assertIn(f"/tmp/remote-msprof/case-case-1/{mirrored_root}/kernel.py", copied_remote_paths)
            self.assertIn(
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/5_MoeInitRouting.json",
                copied_remote_paths,
            )
            self.assertEqual(
                streamed_commands,
                [
                    (
                        f"/tmp/remote-msprof/case-case-1/{mirrored_root}",
                        [
                            "msprof",
                            f"--output=/tmp/remote-msprof/case-case-1/{mirrored_root}/msprof-output",
                            "python3",
                            "bench_runtime.py",
                            "run-one",
                            "--bench-file",
                            "bench_all_cases.py",
                            "--operator-file",
                            "kernel.py",
                            "--case-id",
                            "case-1",
                            "--iterations",
                            "55",
                        ],
                    )
                ],
            )
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
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("KernelB",),
                case_ids=("case-1",),
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")
            operator_json.write_text('{"from":"operator-dir"}\n', encoding="utf-8")
            discovered_json.write_text('{"cases":[1]}\n', encoding="utf-8")

            copied_remote_paths: list[str] = []
            streamed_commands: list[tuple[str, list[str]]] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                if command == ["mkdir", "-p", "/tmp/remote-msprof/case-case-1"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof")
                    return make_skill_result(0, "", "")
                if command == ["mkdir", "-p", f"/tmp/remote-msprof/case-case-1/{root.name}"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-case-1")
                    return make_skill_result(0, "", "")
                if command == ["mkdir", "-p", f"/tmp/remote-msprof/case-case-1/{root.name}/opt-round-13"]:
                    self.assertEqual(remote_workspace, "/tmp/remote-msprof/case-case-1")
                    return make_skill_result(0, "", "")
                if isinstance(command, list) and command[:2] == ["python3", "-c"]:
                    self.assertEqual(remote_workspace, f"/tmp/remote-msprof/case-case-1/{root.name}")
                    return make_skill_result(
                        0,
                        '{"kernel_avg_time_us":2.5,"ops":[{"op_type":"KernelB","avg_time_us":2.5}]}\n',
                        "",
                    )
                if command == ["rm", "-rf", "/tmp/remote-msprof/case-case-1"]:
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
            self.assertIn(
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/bench_all_cases.py",
                copied_remote_paths,
            )
            self.assertIn(
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/opt-round-13/opt_kernel.py",
                copied_remote_paths,
            )
            self.assertIn(
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/opt-round-13/5_MoeInitRouting.json",
                copied_remote_paths,
            )
            self.assertIn(
                f"/tmp/remote-msprof/case-case-1/{mirrored_root}/5_MoeInitRouting.json",
                copied_remote_paths,
            )
            self.assertEqual(
                streamed_commands,
                [
                    (
                        f"/tmp/remote-msprof/case-case-1/{mirrored_root}",
                        [
                            "msprof",
                            f"--output=/tmp/remote-msprof/case-case-1/{mirrored_root}/msprof-output",
                            "python3",
                            "bench_runtime.py",
                            "run-one",
                            "--bench-file",
                            "bench_all_cases.py",
                            "--operator-file",
                            "opt-round-13/opt_kernel.py",
                            "--case-id",
                            "case-1",
                            "--iterations",
                            "55",
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
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("KernelB",),
                case_ids=("case-1", "case-2"),
            )
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
                case_id = command[command.index("--case-id") + 1]
                if case_id == "case-1":
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
                    '{"case_label":"case-1","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":null,"ops":null,"total_op_avg_time_us":null,"error_message":"msprof command failed with return code 1","case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                    '{"case_label":"case-2","kernel_names":["KernelB"],"kernel_source":"metadata","kernel_avg_time_us":5.0,"ops":[{"op_type":"KernelB","avg_time_us":5.0}],"total_op_avg_time_us":5.0,"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_read_remote_msprof_metrics_supports_kernel_suffix_alias_and_total_op(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            remote_workspace = Path(tmp) / "remote-workspace"
            remote_workspace.mkdir()
            for support_path in module._bench_runtime_support_paths():
                shutil.copy2(support_path, remote_workspace / support_path.name)

            output_dir = remote_workspace / "ASCEND_PROFILER_OUTPUT"
            output_dir.mkdir()
            (output_dir / "op_statistic_1.csv").write_text(
                "\n".join(
                    [
                        "Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)",
                        "0,KernelA_kernel,AI_CORE,2,13,5,6.5,8,65",
                        "0,HelperKernel,AI_VECTOR_CORE,2,7,3,3.5,4,35",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def _fake_remote_buffered(spec, remote_workspace_arg, command, **kwargs):
                del kwargs
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace_arg, str(remote_workspace))
                self.assertEqual(command[:2], ["python3", "-c"])
                completed = subprocess.run(
                    command,
                    cwd=remote_workspace,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return make_skill_result(completed.returncode, completed.stdout, completed.stderr)

            with patch.object(module, "run_remote_command_buffered", side_effect=_fake_remote_buffered):
                metrics = module._read_remote_msprof_metrics(
                    "spec",
                    str(remote_workspace),
                    str(output_dir),
                    ["KernelA"],
                )

        self.assertEqual(metrics["kernel_avg_time_us"], 6.5)
        self.assertEqual(
            metrics["ops"],
            [
                {"op_type": "KernelA_kernel", "avg_time_us": 6.5},
                {"op_type": "HelperKernel", "avg_time_us": 3.5},
            ],
        )
        self.assertEqual(metrics["total_op_avg_time_us"], 10.0)

    def test_run_remote_bench_msprof_records_na_when_remote_kernel_row_is_missing(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("MissingKernel",),
                case_ids=("case-1",),
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            removed_tmp_dirs: list[str] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace, "/tmp/remote-msprof")
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
                    '{"case_label":"case-1","kernel_names":["MissingKernel"],"kernel_source":"metadata","kernel_avg_time_us":null,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}],"total_op_avg_time_us":4.0,"error_message":"no resolved kernels matched op_statistic csv","case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_msprof_sums_multiple_declared_kernels(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_kernel.py"
            operator_file = root / "kernel.py"
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="kernel",
                kernel_names=("KernelA", "KernelB"),
                case_ids=("case-1",),
            )
            operator_file.write_text("def kernel():\n    pass\n", encoding="utf-8")

            removed_tmp_dirs: list[str] = []

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
                self.assertEqual(spec, "spec")
                self.assertEqual(remote_workspace, "/tmp/remote-msprof")
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
                    '{"case_label":"case-1","kernel_names":["KernelA","KernelB"],"kernel_source":"metadata","kernel_avg_time_us":4.0,"ops":[{"op_type":"KernelA","avg_time_us":1.5},{"op_type":"KernelB","avg_time_us":2.5}],"total_op_avg_time_us":4.0,"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n'
                ),
            )
            cleanup.assert_called_once_with("spec", "/tmp/remote-msprof", verbose=False, stderr=None)

    def test_run_remote_bench_quotes_filenames_with_spaces(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench kernel.py"
            operator_file = root / "kernel op.py"
            bench_file.write_text("# bench-mode: torch-npu-profiler\n# kernel: k\n", encoding="utf-8")
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
                    '{"case_label":"case-a","kernel_names":["k"],"kernel_source":"metadata","kernel_avg_time_us":1.0,"ops":[{"op_type":"k","avg_time_us":1.0}],"total_op_avg_time_us":1.0,"error_message":null,"case_wall_clock_seconds":0.0,"bench_mode":"msprof"}\n',
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
        self.assertIn("profile_all_bench_cases", remote_run.call_args.args[2][2])

    def test_run_remote_bench_msprof_case_wall_clock_seconds_in_perf_output_success(self) -> None:
        module = load_bench_runner_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bench_file = root / "bench_abs.py"
            operator_file = root / "abs.py"
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="abs_",
                kernel_names=("OpB",),
                case_ids=("case-1",),
            )
            operator_file.write_text("def abs_():\n    pass\n", encoding="utf-8")

            buffered_payloads = [
                "/tmp/msprof-case-1\n",
                '{"kernel_avg_time_us":3.0,"ops":[{"op_type":"OpB","avg_time_us":3.0}]}\n',
            ]

            def _fake_remote_buffered(spec, remote_workspace, command, **kwargs):
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
            _write_hooked_bench_file(
                bench_file,
                mode="msprof",
                api_name="abs_",
                kernel_names=("OpB",),
                case_ids=("case-1",),
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
                    side_effect=lambda _spec, _workspace, command, **_kwargs: (
                        make_skill_result(0, "/tmp/msprof-case-1\n", "")
                        if command == ["mktemp", "-d"]
                        else make_skill_result(0, "", "")
                    ),
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


class RemoteProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bench = load_bench_runner_module()
        run_runtime = load_operator_eval_script_module("run_runtime")
        self.spec = run_runtime.parse_remote_spec("user@host")

    def test_probe_remote_script_includes_clamp_and_preloaded(self) -> None:
        script = self.bench._build_remote_torch_npu_profiler_probe_run_all_script(
            verbose=False, warmup_cap=1, repeats_cap=3
        )
        self.assertIn("dataclasses.replace", script)
        self.assertIn("min(c.warmup, 1)", script)
        self.assertIn("min(c.repeats, 3)", script)
        self.assertIn("preloaded=(clamped, resolution)", script)
        self.assertIn("runtime.load_bench_cases", script)

    def test_canonical_remote_script_does_not_clamp(self) -> None:
        script = self.bench._build_remote_torch_npu_profiler_run_all_script(verbose=False)
        self.assertNotIn("dataclasses.replace", script)
        self.assertNotIn("preloaded=", script)

    def test_remote_profiler_uses_probe_script_when_caps_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.bench, "_stage_remote_bench_runtime_support_files"
        ), patch.object(
            self.bench, "run_remote_command_streaming", return_value=make_skill_result(0, "", "")
        ) as mocked_run, patch.object(self.bench, "copy_file_from_remote"):
            self.bench._run_remote_bench_torch_npu_profiler(
                self.spec,
                "ws",
                Path(tmp) / "bench.py",
                Path(tmp) / "op.py",
                probe_caps=(1, 3),
            )
        command = mocked_run.call_args.args[2]
        script = command[2]
        self.assertIn("preloaded=(clamped, resolution)", script)
        self.assertIn("min(c.warmup, 1)", script)

    def test_remote_profiler_uses_canonical_script_without_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.bench, "_stage_remote_bench_runtime_support_files"
        ), patch.object(
            self.bench, "run_remote_command_streaming", return_value=make_skill_result(0, "", "")
        ) as mocked_run, patch.object(self.bench, "copy_file_from_remote"):
            self.bench._run_remote_bench_torch_npu_profiler(
                self.spec,
                "ws",
                Path(tmp) / "bench.py",
                Path(tmp) / "op.py",
            )
        command = mocked_run.call_args.args[2]
        script = command[2]
        self.assertNotIn("preloaded=", script)
        self.assertNotIn("dataclasses.replace", script)

    def test_run_remote_probe_threads_probe_caps(self) -> None:
        sentinel = ({"return_code": 0, "stdout": "", "stderr": ""}, None, "ws")
        with patch.object(self.bench, "run_remote_bench", return_value=sentinel) as mocked:
            self.bench.run_remote_probe(
                Path("bench.py"),
                Path("op.py"),
                "torch-npu-profiler",
                "user@host",
                None,
                warmup_cap=1,
                repeats_cap=3,
            )
        self.assertEqual(mocked.call_args.kwargs.get("probe_caps"), (1, 3))

    def test_remote_profiler_passes_devices_env_when_caps_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.bench, "_stage_remote_bench_runtime_support_files"
        ), patch.object(
            self.bench, "run_remote_command_streaming", return_value=make_skill_result(0, "", "")
        ) as mocked_run, patch.object(self.bench, "copy_file_from_remote"):
            self.bench._run_remote_bench_torch_npu_profiler(
                self.spec,
                "ws",
                Path(tmp) / "bench.py",
                Path(tmp) / "op.py",
                probe_caps=(1, 3),
                devices=("4", "5"),
            )
        self.assertEqual(
            mocked_run.call_args.kwargs["extra_env"]["ASCEND_RT_VISIBLE_DEVICES"],
            "4,5",
        )

    def test_remote_profiler_omits_devices_env_without_devices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.bench, "_stage_remote_bench_runtime_support_files"
        ), patch.object(
            self.bench, "run_remote_command_streaming", return_value=make_skill_result(0, "", "")
        ) as mocked_run, patch.object(self.bench, "copy_file_from_remote"):
            self.bench._run_remote_bench_torch_npu_profiler(
                self.spec,
                "ws",
                Path(tmp) / "bench.py",
                Path(tmp) / "op.py",
                probe_caps=(1, 3),
            )
        self.assertNotIn(
            "ASCEND_RT_VISIBLE_DEVICES",
            mocked_run.call_args.kwargs["extra_env"],
        )


if __name__ == "__main__":
    unittest.main()
