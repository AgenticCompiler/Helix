import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.run_skill_test_utils import load_test_runner_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_compare_result_payloads_module():
    script = REPO_ROOT / "scripts" / "compare_result_payloads.py"
    spec = importlib.util.spec_from_file_location("compare_result_payloads_test_module", script)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module spec for {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalTestRunnerTests(unittest.TestCase):
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
        module = load_test_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("placeholder", encoding="utf-8")
            new.write_text("placeholder", encoding="utf-8")

            with patch.object(
                module,
                "_load_result_payload",
                side_effect=[
                    {"results": [[1.0, 2.0], {"x": 3.0}]},
                    {"results": [[1.0, 2.0], {"x": 3.0}]},
                ],
            ):
                return_code = module.compare_result_files(oracle, new, "balanced")
            self.assertEqual(return_code, 0)


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

    def test_local_compare_values_handles_scalar_edge_cases(self) -> None:
        self._assert_scalar_contract(load_test_runner_module())

    def test_remote_compare_values_handles_scalar_edge_cases(self) -> None:
        self._assert_scalar_contract(_load_compare_result_payloads_module())


if __name__ == "__main__":
    unittest.main()
