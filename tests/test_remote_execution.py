import sys
import unittest
import json
from io import StringIO
from os import environ
from pathlib import Path
from unittest.mock import patch
import tempfile
import subprocess
import shutil
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.remote import env as remote_env_module
from helix.skills.loader import load_operator_eval_script_module
from tests.run_skill_test_utils import (
    load_compare_result_module,
    load_bench_modes_module,
    load_bench_remote_api_module,
    load_remote_api_module,
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
        cls._bench_module = load_bench_modes_module()
        cls._monotonic_patcher = patch.object(cls._bench_module.time, "monotonic", return_value=0.0)
        cls._monotonic_patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._monotonic_patcher.stop()

    def test_app_remote_execution_module_has_been_removed(self) -> None:
        remote_execution = Path(__file__).resolve().parents[1] / "src" / "helix" / "remote_execution.py"

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

    def test_parse_remote_spec_accepts_ssh_alias(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        spec = module.parse_remote_spec("R154_cdj")

        self.assertEqual(spec["user_host"], "R154_cdj")
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

    def test_copy_files_to_remote_uses_one_scp_command(self) -> None:
        module = load_operator_eval_script_module("run_runtime")
        sources = [Path("/tmp/first.py"), Path("/tmp/second.py")]

        with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")) as mocked:
            module.copy_files_to_remote(
                module.parse_remote_spec("alice@example.com:2200"),
                sources,
                "/tmp/remote",
            )

        self.assertEqual(
            mocked.call_args.args[0],
            [
                "scp", "-P", "2200", "/private/tmp/first.py", "/private/tmp/second.py",
                "alice@example.com:/tmp/remote/",
            ],
        )

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
            "/tmp/helix",
            env,
        )

        self.assertEqual(env[remote_env_module.remote_target_env_name()], "alice@example.com")
        self.assertEqual(env[remote_env_module.remote_workdir_env_name()], "/tmp/helix")

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
        module = load_bench_modes_module()

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
from helix.skills.loader import load_operator_eval_script_module
module = load_operator_eval_script_module("run_bench_modes")
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

    def test_run_runtime_buffered_wall_timeout_is_not_extended_by_output(self) -> None:
        module = load_operator_eval_script_module("run_runtime")
        command = [
            sys.executable,
            "-c",
            (
                "import time\n"
                "for _ in range(100):\n"
                "    print('still running', flush=True)\n"
                "    time.sleep(0.02)\n"
            ),
        ]

        self._monotonic_patcher.stop()
        try:
            result = module.run_buffered_process(
                command,
                ".",
                stall_timeout_seconds=0,
                timeout_seconds=0.1,
            )
        finally:
            self._monotonic_patcher.start()

        self.assertEqual(result["return_code"], 1)
        self.assertTrue(result["stalled"])
        self.assertIn("HELIX_EVAL_TIMEOUT_SECONDS", result["stderr"])
        self.assertIn("timed out after 0.1 seconds", result["stderr"])

    def test_run_runtime_windows_termination_kills_process_tree(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        class _FakeProcess:
            pid = 123

            def poll(self):
                return None

            def terminate(self) -> None:
                raise AssertionError("taskkill should be available")

        with (
            patch.object(module, "_IS_WINDOWS", True),
            patch.object(
                module.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=0),
            ) as taskkill,
        ):
            module._terminate_process_tree(_FakeProcess())

        taskkill.assert_called_once_with(
            ["taskkill", "/PID", "123", "/T", "/F"],
            check=False,
            stdout=module.subprocess.DEVNULL,
            stderr=module.subprocess.DEVNULL,
        )

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

        with patch.dict(module.os.environ, {"HELIX_EVAL_TIMEOUT_SECONDS": "-1"}, clear=False):
            with self.assertRaises(ValueError):
                module.eval_timeout_seconds()

    def test_run_remote_command_streaming_shell_joins_sequence_args(self) -> None:
        module = load_operator_eval_script_module("run_runtime")

        with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")) as mocked:
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

        with patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")) as mocked:
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
            patch.object(module, "run_buffered_process", return_value=make_skill_result(0, "", "")) as mocked,
        ):
            module.run_remote_command_streaming(
                module.parse_remote_spec("alice@example.com"),
                "/tmp/workspace",
                ["python3", "bench.py"],
            )

        command = mocked.call_args.args[0]
        self.assertIn("TRITON_ALL_BLOCKS_PARALLEL=0", command[-1])

    def test_run_remote_test_keeps_workspace_when_requested(self) -> None:
        module = load_remote_api_module()

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
                module, "copy_files_to_remote"
            ), patch.object(
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
        module = load_remote_api_module()

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
                module, "copy_files_to_remote"
            ), patch.object(
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
            [
                "python3", "run_test_remote_worker.py", "--test-file", "test kernel.py",
                "--operator-file", "kernel op.py", "--test-mode", "standalone",
            ],
        )

    def test_remote_differential_comparison_keeps_pt_files_remote(self) -> None:
        module = load_remote_api_module()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_file = root / "differential test.py"
            ref_operator = root / "reference kernel.py"
            operator = root / "candidate kernel.py"
            test_file.write_text("# test-mode: differential\n", encoding="utf-8")
            ref_operator.write_text("def kernel():\n    pass\n", encoding="utf-8")
            operator.write_text("def kernel():\n    pass\n", encoding="utf-8")

            with patch.object(
                module,
                "create_remote_workspace",
                return_value=("spec", "/tmp/remote-differential"),
            ), patch.object(module, "copy_file_to_remote") as copy_to_remote, patch.object(
                module, "copy_files_to_remote"
            ) as copy_files_to_remote, patch.object(module, "_stage_remote_python_bundle") as stage_bundle, patch.object(
                module,
                "copy_file_from_remote",
            ) as copy_from_remote, patch.object(
                module,
                "run_remote_command_streaming",
                return_value=make_skill_result(0, "PASS\n", ""),
            ) as remote_run, patch.object(module, "cleanup_remote_workspace") as cleanup:
                result, workspace = module.run_remote_differential_comparison(
                    test_file,
                    ref_operator,
                    operator,
                    "alice@example.com",
                    None,
                    accuracy_mode="dtype-close",
                )

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(workspace, "/tmp/remote-differential")
        copy_from_remote.assert_not_called()
        copied_sources = [call.args[1] for call in copy_to_remote.call_args_list]
        self.assertIn(test_file, copied_sources)
        self.assertIn(ref_operator, copied_sources)
        self.assertIn(operator, copied_sources)
        self.assertNotIn(ref_operator.with_name("reference kernel_result.pt"), copied_sources)
        copy_files_to_remote.assert_not_called()
        stage_bundle.assert_called_once()
        self.assertEqual(remote_run.call_count, 3)
        reference_command = remote_run.call_args_list[0].args[2]
        candidate_command = remote_run.call_args_list[1].args[2]
        compare_command = remote_run.call_args_list[2].args[2]
        self.assertEqual(reference_command[0:8], [
            "python3", "run_test_remote_worker.py", "--test-file", test_file.name,
            "--operator-file", "reference_reference kernel.py", "--test-mode", "differential",
        ])
        self.assertEqual(candidate_command[0:8], [
            "python3", "run_test_remote_worker.py", "--test-file", test_file.name,
            "--operator-file", "candidate_candidate kernel.py", "--test-mode", "differential",
        ])
        self.assertEqual(
            compare_command,
            [
                "python3",
                "compare_result.py",
                "--ref-result",
                "reference_reference kernel_result.pt",
                "--new-result",
                "candidate_candidate kernel_result.pt",
                "--accuracy-mode",
                "dtype-close",
            ],
        )
        cleanup.assert_called_once_with("spec", "/tmp/remote-differential", verbose=False, stderr=None)

    def test_compare_remote_result_files_quotes_filenames_with_spaces(self) -> None:
        module = load_compare_result_module()
        runtime = module.run_runtime

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "oracle result.pt"
            new = root / "new result.pt"
            oracle.write_text("oracle", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

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
                    accuracy_mode="dtype-close",
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
        copied_targets = [call.args[2].rsplit("/", 1)[-1] for call in copy_to_remote.call_args_list]
        self.assertEqual(
            copied_targets,
            [
                "compare_result.py",
                "npu_compare.py",
                "dtype_close_compare.py",
                "npu_compare_common.py",
                "npu_contract_compare.py",
                "env_registry.py",
                "oracle result.pt",
                "new result.pt",
            ],
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
                "--accuracy-mode",
                "dtype-close",
            ],
        )

    def test_run_bench_remote_api_owns_workspace_staging_and_cleanup(self) -> None:
        module = load_bench_remote_api_module()
        bench_file = Path("bench.py")
        operator_file = Path("operator.py")
        result = make_skill_result(0, "", "")
        with patch.object(module, "create_remote_workspace", return_value=("spec", "/tmp/remote")), \
            patch.object(module, "copy_file_to_remote") as copy_file, \
            patch.object(module, "stage_remote_bench_input_files") as stage_support, \
            patch.object(
                module,
                "execute_remote_bench_workspace",
                return_value=(result, Path("operator_perf.txt"), "/tmp/remote"),
            ) as execute, \
            patch.object(module, "cleanup_remote_workspace") as cleanup:
            actual, perf_path, workspace = module.run_remote_bench(
                bench_file, operator_file, "standalone", "user@host", None
            )

        self.assertEqual(actual, result)
        self.assertEqual(perf_path, Path("operator_perf.txt"))
        self.assertEqual(workspace, "/tmp/remote")
        self.assertEqual(
            [call.args[2] for call in copy_file.call_args_list],
            [
                "/tmp/remote/bench_contract.py",
                "/tmp/remote/env_registry.py",
                "/tmp/remote/perf_artifacts.py",
                "/tmp/remote/profile_csv_parser.py",
                "/tmp/remote/result_payload.py",
                "/tmp/remote/run_bench_execution.py",
                "/tmp/remote/run_bench_remote_worker.py",
                "/tmp/remote/torch_npu_warnings.py",
                "/tmp/remote/bench.py",
                "/tmp/remote/operator.py",
            ],
        )
        stage_support.assert_called_once()
        self.assertEqual(execute.call_args.args[:5], (bench_file, operator_file, "standalone", "spec", "/tmp/remote"))
        cleanup.assert_called_once_with("spec", "/tmp/remote", verbose=False, stderr=None)

    def test_run_bench_modes_uses_fixed_remote_worker(self) -> None:
        module = load_bench_modes_module()
        result = make_skill_result(0, "", "")
        with patch.object(module, "run_remote_command_streaming", return_value=result) as run_remote, \
            patch.object(module, "copy_file_from_remote"):
            actual, _perf_path, _workspace = module._run_remote_bench_torch_npu_profiler(
                "spec", "/tmp/remote", Path("bench.py"), Path("operator.py"), execution_limits=(1, 3)
            )

        self.assertEqual(actual, result)
        command = run_remote.call_args.args[2]
        self.assertEqual(command[:3], ["python3", "run_bench_remote_worker.py", "profile-all"])
        self.assertIn("--warmup-cap", command)
        self.assertIn("--repeats-cap", command)

    def test_run_bench_modes_reads_msprof_metrics_through_fixed_worker(self) -> None:
        module = load_bench_modes_module()
        payload = '{"kernel_avg_time_us": 2.5, "ops": [], "total_op_avg_time_us": 2.5}'
        with patch.object(
            module,
            "run_remote_command_buffered",
            return_value=make_skill_result(0, payload + "\n", ""),
        ) as run_remote:
            metrics = module._read_remote_msprof_metrics(
                "spec", "/tmp/remote", "/tmp/metrics", ["KernelA"]
            )

        self.assertEqual(metrics["kernel_avg_time_us"], 2.5)
        self.assertEqual(
            run_remote.call_args.args[2][:3],
            ["python3", "run_bench_remote_worker.py", "msprof-metrics"],
        )


class RemoteProbeTests(unittest.TestCase):
    def test_run_remote_bench_with_limits_threads_limits_through_remote_api(self) -> None:
        module = load_bench_remote_api_module()
        sentinel = (make_skill_result(0, "", ""), None, "workspace")
        with patch.object(module, "execute_remote_bench_workspace", return_value=sentinel) as run_bench, \
             patch.object(module, "create_remote_workspace", return_value=("spec", "workspace")), \
             patch.object(module, "_stage_remote_bench_inputs"), \
             patch.object(module, "cleanup_remote_workspace"):
            actual = module.run_remote_bench_with_limits(
                Path("bench.py"),
                Path("operator.py"),
                "torch-npu-profiler",
                "user@host",
                None,
                warmup_cap=1,
                repeats_cap=3,
            )

        self.assertEqual(actual, sentinel)
        self.assertEqual(run_bench.call_args.kwargs["execution_limits"], (1, 3))


if __name__ == "__main__":
    unittest.main()
