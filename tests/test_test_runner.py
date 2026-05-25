import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.run_skill_test_utils import load_compare_result_module, load_test_runner_module

_WARNING_LINE = "[WARNING] Please DO NOT tune args ['num_warps']!\n"
_ANOTHER_WARNING_LINE = "[WARNING] autotune fallback was used\n"


class LocalTestRunnerTests(unittest.TestCase):
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
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    def case_a():
        return operator_api("a")

    def case_b():
        return operator_api("b")

    return [{"id": "case-a", "fn": case_a}, {"id": "case-b", "fn": case_b}]
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
        self.assertEqual(payload, {"results": ["A", "B"]})
        self.assertFalse((root / "TEST_RESULT.pt").exists())

    def test_run_local_test_falls_back_to_legacy_script_mode_when_hooks_are_missing(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text(
                "def main():\n"
                "    return None\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n",
                encoding="utf-8",
            )

            fake_result = {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "stalled": False,
                "session_id": None,
            }

            with patch.object(module, "run_streaming_process", return_value=fake_result) as stream_mock:
                with patch.object(module, "archive_differential_result", return_value=root / "abs_result.pt") as archive_mock:
                    result, archived = module.run_local_test(test_file, operator, "differential")

        self.assertEqual(result, fake_result)
        self.assertEqual(archived, root / "abs_result.pt")
        stream_mock.assert_called_once()
        archive_mock.assert_called_once_with(test_file, operator)

    def test_run_local_test_filters_warning_prefix_lines_by_default(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")

            fake_result = {
                "return_code": 0,
                "stdout": _WARNING_LINE + "useful stdout\n" + _ANOTHER_WARNING_LINE,
                "stderr": _ANOTHER_WARNING_LINE + "useful stderr\n",
                "stalled": False,
                "session_id": None,
            }

            with patch.object(module, "run_streaming_process", return_value=fake_result):
                result, archived = module.run_local_test(test_file, operator, "standalone", verbose=False)

        self.assertEqual(result["stdout"], "useful stdout\n")
        self.assertEqual(result["stderr"], "useful stderr\n")
        self.assertIsNone(archived)

    def test_run_local_test_preserves_warning_prefix_lines_in_verbose_mode(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "test_abs.py"
            operator.write_text("def noop():\n    return 1\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")

            fake_result = {
                "return_code": 0,
                "stdout": _WARNING_LINE + "useful stdout\n",
                "stderr": _ANOTHER_WARNING_LINE,
                "stalled": False,
                "session_id": None,
            }

            with patch.object(module, "run_streaming_process", return_value=fake_result):
                result, archived = module.run_local_test(test_file, operator, "standalone", verbose=True)

        self.assertEqual(result, fake_result)
        self.assertIsNone(archived)

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
# api-name: build_api
# api-kind: torch-function
# kernels: KernelA

def build_operator_api(operator_module):
    return operator_module.build_api()

def build_differential_test_cases(operator_api):
    def case_a():
        return operator_api("a")
    return [{"id": "case-a", "fn": case_a}]
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
        self.assertIn('f"{operator_file.stem}_result.pt"', recorded_command[2])
        self.assertNotIn("TEST_RESULT.pt", recorded_command[2])

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

    def test_find_case_insensitive_result_file_matches_lowercase_payload(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "test_result.pt"
            payload.write_text("payload", encoding="utf-8")

            resolved = module.find_case_insensitive_result_file(root)

            self.assertEqual(resolved, payload)

    def test_archive_differential_result_uses_operator_filename_result_name(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            payload = root / "TEST_RESULT.pt"
            operator.write_text("def abs_():\n    pass\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")
            payload.write_text("payload", encoding="utf-8")

            archived = module.archive_differential_result(test_file, operator)

            self.assertEqual(archived, root / "abs_result.pt")
            self.assertEqual(archived.read_text(encoding="utf-8"), "payload")

    def test_archive_differential_result_uses_operator_filename_for_any_operator(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_abs.py"
            test_file = root / "differential_test_abs.py"
            payload = root / "Test_Result.PT"
            operator.write_text("def abs_():\n    pass\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")
            payload.write_text("payload", encoding="utf-8")

            archived = module.archive_differential_result(test_file, operator)

            self.assertEqual(archived, root / "opt_abs_result.pt")
            self.assertEqual(archived.read_text(encoding="utf-8"), "payload")

    def test_archive_differential_result_requires_payload_file(self) -> None:
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def abs_():\n    pass\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                module.archive_differential_result(test_file, operator)

    def test_compare_result_files_compares_payloads_locally(self) -> None:
        module = load_compare_result_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("placeholder", encoding="utf-8")
            new.write_text("placeholder", encoding="utf-8")

            with patch.object(module, "compare_result_files", return_value=0) as compare_mock:
                return_code = module.compare_result_files(oracle, new, "balanced")

            self.assertEqual(return_code, 0)
            compare_mock.assert_called_once_with(oracle, new, "balanced")


class ScalarComparisonRegressionTests(unittest.TestCase):
    def _assert_scalar_contract(self, module) -> None:
        self.assertIsNone(
            module._compare_values(float("nan"), float("nan"), "output", 1e-4, 1e-5)
        )
        self.assertEqual(
            module._compare_values(float("nan"), 1.0, "output", 1e-4, 1e-5),
            "output NaN mismatch: expected nan, got 1.0",
        )
        self.assertEqual(
            module._compare_values(1.0, float("nan"), "output", 1e-4, 1e-5),
            "output NaN mismatch: expected 1.0, got nan",
        )
        self.assertIsNone(module._compare_values(3.0, 3, "output", 1e-4, 1e-5))
        self.assertIsNone(module._compare_values(3, 3.0000001, "output", 1e-4, 1e-5))
        self.assertEqual(
            module._compare_values(True, 1.00001, "output", 1e-4, 1e-5),
            "output value mismatch: expected True, got 1.00001",
        )
        self.assertEqual(
            module._compare_values(False, 1e-5, "output", 1e-4, 1e-5),
            "output value mismatch: expected False, got 1e-05",
        )

    def test_compare_result_handles_scalar_edge_cases(self) -> None:
        self._assert_scalar_contract(load_compare_result_module())


if __name__ == "__main__":
    unittest.main()
