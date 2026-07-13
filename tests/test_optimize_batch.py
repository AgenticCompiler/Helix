import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.models import AgentResult
from helix.optimize.naming import (
    is_batch_optimize_operator_candidate,
)
from helix.optimize.batch import (
    resolve_batch_optimize_operator_file,
    summarize_batch_optimize_failure,
)


class OptimizeBatchHelpersTests(unittest.TestCase):
    def test_resolve_batch_optimize_operator_file_excludes_generated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (workspace / "test_kernel.py").write_text("", encoding="utf-8")
            (workspace / "differential_test_kernel.py").write_text("", encoding="utf-8")
            (workspace / "bench_kernel.py").write_text("", encoding="utf-8")
            (workspace / "opt_kernel.py").write_text("", encoding="utf-8")
            (workspace / "__init__.py").write_text("", encoding="utf-8")

            resolved = resolve_batch_optimize_operator_file(workspace)

            self.assertEqual(resolved, workspace / "kernel.py")

    def test_resolve_batch_optimize_operator_file_rejects_multiple_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "a.py").write_text("print('a')\n", encoding="utf-8")
            (workspace / "b.py").write_text("print('b')\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "multiple candidate operator files"):
                resolve_batch_optimize_operator_file(workspace)

    def test_is_batch_optimize_operator_candidate_filters_non_operator_names(self) -> None:
        workspace = Path("/tmp")

        self.assertTrue(is_batch_optimize_operator_candidate(workspace / "kernel.py"))
        self.assertFalse(is_batch_optimize_operator_candidate(workspace / "test_kernel.py"))
        self.assertFalse(
            is_batch_optimize_operator_candidate(workspace / "differential_test_kernel.py")
        )
        self.assertFalse(is_batch_optimize_operator_candidate(workspace / "bench_kernel.py"))
        self.assertFalse(is_batch_optimize_operator_candidate(workspace / "opt_kernel.py"))
        self.assertFalse(is_batch_optimize_operator_candidate(workspace / "__init__.py"))
        self.assertFalse(is_batch_optimize_operator_candidate(workspace / "kernel.txt"))

    def test_summarize_batch_optimize_failure_prefers_last_non_blank_stderr_line(self) -> None:
        result = AgentResult(return_code=1, stdout="stdout line\n", stderr="\nfirst\nsecond\n")

        summary = summarize_batch_optimize_failure(result)

        self.assertEqual(summary, "second")

    def test_summarize_batch_optimize_failure_falls_back_to_return_code(self) -> None:
        result = AgentResult(return_code=7, stdout="   \n", stderr="")

        summary = summarize_batch_optimize_failure(result)

        self.assertEqual(summary, "optimize exited with return code 7")


if __name__ == "__main__":
    unittest.main()
