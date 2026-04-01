import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.test_runner import (
    archive_differential_result,
    find_case_insensitive_result_file,
)


class LocalTestRunnerTests(unittest.TestCase):
    def test_find_case_insensitive_result_file_matches_lowercase_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = root / "test_result.pt"
            payload.write_text("payload", encoding="utf-8")

            resolved = find_case_insensitive_result_file(root)

            self.assertEqual(resolved, payload)

    def test_archive_differential_result_uses_operator_filename_result_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            payload = root / "TEST_RESULT.pt"
            operator.write_text("def abs_():\n    pass\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")
            payload.write_text("payload", encoding="utf-8")

            archived = archive_differential_result(test_file, operator)

            self.assertEqual(archived, root / "abs_result.pt")
            self.assertEqual(archived.read_text(encoding="utf-8"), "payload")

    def test_archive_differential_result_uses_operator_filename_for_any_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "opt_abs.py"
            test_file = root / "differential_test_abs.py"
            payload = root / "Test_Result.PT"
            operator.write_text("def abs_():\n    pass\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")
            payload.write_text("payload", encoding="utf-8")

            archived = archive_differential_result(test_file, operator)

            self.assertEqual(archived, root / "opt_abs_result.pt")
            self.assertEqual(archived.read_text(encoding="utf-8"), "payload")

    def test_archive_differential_result_requires_payload_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            operator = root / "abs.py"
            test_file = root / "differential_test_abs.py"
            operator.write_text("def abs_():\n    pass\n", encoding="utf-8")
            test_file.write_text("print('test')\n", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                archive_differential_result(test_file, operator)

    def test_compare_result_files_compares_payloads_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = root / "abs_result.pt"
            new = root / "opt_abs_result.pt"
            oracle.write_text("placeholder", encoding="utf-8")
            new.write_text("placeholder", encoding="utf-8")

            from triton_agent.test_runner import compare_result_files

            with patch(
                "triton_agent.test_runner._load_result_payload",
                side_effect=[
                    {"results": [[1.0, 2.0], {"x": 3.0}]},
                    {"results": [[1.0, 2.0], {"x": 3.0}]},
                ],
            ):
                return_code = compare_result_files(oracle, new, "balanced")
            self.assertEqual(return_code, 0)


if __name__ == "__main__":
    unittest.main()
