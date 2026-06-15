import pickle
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.run_skill_test_utils import load_compare_result_module, load_test_runner_module

_WARNING_LINE = "[WARNING] Please DO NOT tune args ['num_warps']!\n"
_ANOTHER_WARNING_LINE = "[WARNING] autotune fallback was used\n"


class LocalTestRunnerTests(unittest.TestCase):
    def test_run_local_test_prints_visible_devices_when_debug_enabled(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

def main(operator_api):
    print(operator_api())
""",
                encoding="utf-8",
            )
            stdout = StringIO()

            with patch.dict(
                module.os.environ,
                {
                    "TRITON_AGENT_DEBUG": "1",
                    "ASCEND_RT_VISIBLE_DEVICES": "3",
                },
                clear=False,
            ):
                with patch("sys.stdout", stdout):
                    result, archived = module.run_local_test(
                        test_file,
                        operator,
                        "standalone",
                    )

        self.assertEqual(result["return_code"], 0)
        self.assertIsNone(archived)
        self.assertIn("[TRITON_AGENT_DEBUG] ASCEND_RT_VISIBLE_DEVICES=3", stdout.getvalue())

    def test_run_local_test_executes_declarative_differential_cases(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    inputs_b = ("b",)
    return [
        {"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)},
        {"id": "case-b", "inputs": inputs_b, "fn": lambda: operator_api(*inputs_b)},
    ]
                """,
                encoding="utf-8",
            )

            def fake_save(obj, path) -> None:
                Path(path).write_bytes(pickle.dumps(obj))

            with patch.dict(sys.modules, {"torch": SimpleNamespace(save=fake_save)}, clear=False):
                result, archived = module.run_local_test(test_file, operator, "differential")

            payload = pickle.loads((root / "abs_result.pt").read_bytes())

        self.assertEqual(result["return_code"], 0)
        self.assertEqual(archived, root / "abs_result.pt")
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [
                    {"id": "case-a", "inputs": ("a",), "result": "A"},
                    {"id": "case-b", "inputs": ("b",), "result": "B"},
                ],
            },
        )
        self.assertFalse((root / "TEST_RESULT.pt").exists())

    def test_run_local_test_imports_standalone_module_and_calls_main_with_operator_api(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text(
                "import torch\n\n"
                "def abs_entry():\n"
                "    return torch.tensor([1, 2], dtype=torch.int32)\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: non-compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

from npu_compare import compare_case_result

def main(operator_api):
    expected = operator_api()
    actual = operator_api()
    result = compare_case_result(
        case_id="case-a",
        actual=actual,
        golden=expected,
        inputs=(),
        compute=False,
    )
    if not result.passed:
        raise AssertionError(result.message)
    print(f"path={result.comparison_path} compute={result.compute}")
""",
                encoding="utf-8",
            )
            result, archived = module.run_local_test(test_file, operator, "standalone")

        self.assertEqual(result["return_code"], 0)
        self.assertIn("path=non-compute compute=False", result["stdout"])
        self.assertEqual(result["stderr"], "")
        self.assertIsNone(archived)

    def test_run_local_test_reports_missing_differential_hooks_without_legacy_fallback(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text("def main():\n    return None\n", encoding="utf-8")
            result, archived = module.run_local_test(test_file, operator, "differential")

        self.assertEqual(result["return_code"], 1)
        self.assertIn("missing required hook 'build_operator_api'", result["stderr"])
        self.assertIsNone(archived)

    def test_run_local_test_filters_warning_prefix_lines_by_default(self) -> None:
        module = load_test_runner_module()
        fake_result = {
            "return_code": 0,
            "stdout": _WARNING_LINE + "useful stdout\n" + _ANOTHER_WARNING_LINE,
            "stderr": _ANOTHER_WARNING_LINE + "useful stderr\n",
            "stalled": False,
            "session_id": None,
        }
        result = module._filter_result_payload(fake_result, verbose=False)

        self.assertEqual(result["stdout"], "useful stdout\n")
        self.assertEqual(result["stderr"], "useful stderr\n")

    def test_run_local_test_preserves_warning_prefix_lines_in_verbose_mode(self) -> None:
        module = load_test_runner_module()
        fake_result = {
            "return_code": 0,
            "stdout": _WARNING_LINE + "useful stdout\n",
            "stderr": _ANOTHER_WARNING_LINE,
            "stalled": False,
            "session_id": None,
        }
        result = module._filter_result_payload(fake_result, verbose=True)

        self.assertEqual(result, fake_result)

    def test_run_remote_test_executes_declarative_differential_cases(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    return [{"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)}]
""",
                encoding="utf-8",
            )

            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }
            recorded_command: list[str] = []

            with patch.object(module, "create_remote_workspace", return_value=({"user_host": "user@host", "port": None}, "/tmp/remote")):
                with patch.object(module, "copy_file_to_remote"):
                    with patch.object(module, "run_remote_command_streaming", side_effect=lambda _spec, _workspace, command, **_kwargs: recorded_command.extend(command) or fake_result):
                        with patch.object(module, "_copy_remote_differential_archive", return_value=root / "abs_result.pt"):
                            with patch.object(module, "cleanup_remote_workspace"):
                                result, archived, remote_workspace = module.run_remote_test(
                                    test_file,
                                    operator,
                                    "differential",
                                    "user@host",
                                    None,
                                    keep_remote_workdir=False,
                                    verbose=False,
                                    stderr=None,
                                )

        self.assertEqual(result, fake_result)
        self.assertEqual(archived, root / "abs_result.pt")
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertEqual(recorded_command[0:2], ["python3", "-c"])
        self.assertIn("build_differential_test_cases", recorded_command[2])
        self.assertIn('"compute": compute', recorded_command[2])
        self.assertIn('"inputs"', recorded_command[2])
        self.assertIn('"cases"', recorded_command[2])
        self.assertNotIn('"results"', recorded_command[2])

    def test_remote_differential_generated_script_executes_successfully(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            (root / "torch.py").write_text(
                "import pickle\n\n"
                "def save(obj, path):\n"
                "    with open(path, 'wb') as handle:\n"
                "        pickle.dump(obj, handle)\n\n"
                "class _Npu:\n"
                "    def synchronize(self):\n"
                "        pass\n\n"
                "npu = _Npu()\n",
                encoding="utf-8",
            )
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    return [{"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)}]
""",
                encoding="utf-8",
            )

            command = module._build_remote_differential_command(test_file.name, operator.name)
            completed = subprocess.run(
                [sys.executable, *command[1:]],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = pickle.loads((root / "abs_result.pt").read_bytes()) if (root / "abs_result.pt").exists() else None

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stderr, "")
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-a", "inputs": ("a",), "result": "A"}],
            },
        )

    def test_remote_differential_generated_script_accepts_mapping_cases(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            (root / "torch.py").write_text(
                "import pickle\n\n"
                "def save(obj, path):\n"
                "    with open(path, 'wb') as handle:\n"
                "        pickle.dump(obj, handle)\n\n"
                "class _Npu:\n"
                "    def synchronize(self):\n"
                "        pass\n\n"
                "npu = _Npu()\n",
                encoding="utf-8",
            )
            operator.write_text(
                "def build_api():\n"
                "    return lambda value: value.upper()\n",
                encoding="utf-8",
            )
            test_file.write_text(
                """# test-mode: differential
# compute-kind: compute
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

from collections import UserDict

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    inputs_a = ("a",)
    return [UserDict({"id": "case-a", "inputs": inputs_a, "fn": lambda: operator_api(*inputs_a)})]
""",
                encoding="utf-8",
            )

            command = module._build_remote_differential_command(test_file.name, operator.name)
            completed = subprocess.run(
                [sys.executable, *command[1:]],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
            )

            payload = pickle.loads((root / "abs_result.pt").read_bytes()) if (root / "abs_result.pt").exists() else None

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            payload,
            {
                "compute": True,
                "cases": [{"id": "case-a", "inputs": ("a",), "result": "A"}],
            },
        )

    def test_run_remote_test_executes_import_only_standalone_main(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def abs_entry():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                """# test-mode: standalone
# compute-kind: compute
# api-name: abs_entry
# api-kind: torch-function
# kernels: KernelA

from npu_compare import compare_case_result

def main(operator_api):
    result = compare_case_result(
        case_id="case-a",
        actual=operator_api(),
        golden=operator_api(),
        inputs=(),
        compute=True,
    )
    if not result.passed:
        raise AssertionError(result.message)
""",
                encoding="utf-8",
            )

            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }
            copied_targets: list[str] = []
            recorded_command: list[str] = []

            with patch.object(module, "create_remote_workspace", return_value=({"user_host": "user@host", "port": None}, "/tmp/remote")):
                with patch.object(module, "copy_file_to_remote", side_effect=lambda _spec, _src, dst, **_kwargs: copied_targets.append(dst)):
                    with patch.object(module, "run_remote_command_streaming", side_effect=lambda _spec, _workspace, command, **_kwargs: recorded_command.extend(command) or fake_result):
                        with patch.object(module, "cleanup_remote_workspace"):
                            result, archived, remote_workspace = module.run_remote_test(
                                test_file,
                                operator,
                                "standalone",
                                "user@host",
                                None,
                                keep_remote_workdir=False,
                                verbose=False,
                                stderr=None,
                            )

        self.assertEqual(result, fake_result)
        self.assertIsNone(archived)
        self.assertEqual(remote_workspace, "/tmp/remote")
        self.assertIn("/tmp/remote/npu_compare.py", copied_targets)
        self.assertEqual(recorded_command[0:2], ["python3", "-c"])
        self.assertIn("main_fn(operator_api)", recorded_command[2])
        self.assertNotIn(test_file.name, recorded_command[2].split())

    def test_run_remote_test_filters_warning_prefix_lines_by_default(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")

            fake_result = {
                "return_code": 0,
                "stdout": _WARNING_LINE + "remote stdout\n",
                "stderr": _ANOTHER_WARNING_LINE + "remote stderr\n",
                "stalled": False,
                "session_id": None,
            }

            with patch.object(module, "create_remote_workspace", return_value=({"user_host": "user@host", "port": None}, "/tmp/remote")):
                with patch.object(module, "copy_file_to_remote"):
                    with patch.object(module, "run_remote_command_streaming", return_value=fake_result):
                        with patch.object(module, "cleanup_remote_workspace"):
                            result, archived, remote_workspace = module.run_remote_test(
                                test_file,
                                operator,
                                "standalone",
                                "user@host",
                                None,
                                keep_remote_workdir=False,
                                verbose=False,
                                stderr=None,
                            )

        self.assertEqual(result["stdout"], "remote stdout\n")
        self.assertEqual(result["stderr"], "remote stderr\n")
        self.assertIsNone(archived)
        self.assertEqual(remote_workspace, "/tmp/remote")

    def test_compare_result_files_compares_payloads_locally(self) -> None:
        module = load_compare_result_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("placeholder", encoding="utf-8")
            new.write_text("placeholder", encoding="utf-8")

            with patch.object(module, "compare_result_files", return_value=0) as compare_mock:
                return_code = module.compare_result_files(oracle, new)

            self.assertEqual(return_code, 0)
            compare_mock.assert_called_once_with(oracle, new)


if __name__ == "__main__":
    unittest.main()
