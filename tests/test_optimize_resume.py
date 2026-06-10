import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.resume import classify_optimize_workspace


class OptimizeResumeTests(unittest.TestCase):
    def _write_baseline_state(
        self,
        workspace: Path,
        *,
        source_operator: str,
        test_file: str,
        bench_file: str,
        bench_mode: str = "torch-npu-profiler",
    ) -> None:
        baseline_dir = workspace / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "original",
                    "source_operator": source_operator,
                    "baseline_operator": "baseline/opt_kernel.py",
                    "test_file": test_file,
                    "test_mode": "differential",
                    "bench_file": bench_file,
                    "bench_mode": bench_mode,
                    "perf_artifact": "baseline/perf.txt",
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
        (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")

    def test_classify_optimize_workspace_prefers_matching_baseline_state_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "triton_7_Sum.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()
            self._write_baseline_state(
                workspace,
                source_operator="triton_7_Sum.py",
                test_file="differential_test_7_Sum.py",
                bench_file="bench_triton_7_Sum.py",
                bench_mode="msprof",
            )
            (workspace / "differential_test_7_Sum.py").write_text(
                "# test-mode: differential\nprint('test')\n",
                encoding="utf-8",
            )
            (workspace / "bench_triton_7_Sum.py").write_text(
                "# bench-mode: msprof\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            inspection = classify_optimize_workspace(operator, workspace)

            self.assertEqual(inspection.state, "resumable-session")
            self.assertEqual(inspection.test_mode, "differential")
            self.assertEqual(inspection.bench_mode, "msprof")

    def test_classify_optimize_workspace_ignores_baseline_state_for_different_source_operator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            operator = workspace / "triton_7_Sum.py"
            operator.write_text("print('source')\n", encoding="utf-8")
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()
            self._write_baseline_state(
                workspace,
                source_operator="7_Sum.py",
                test_file="differential_test_7_Sum.py",
                bench_file="bench_triton_7_Sum.py",
            )
            (workspace / "differential_test_7_Sum.py").write_text(
                "# test-mode: differential\nprint('test')\n",
                encoding="utf-8",
            )
            (workspace / "bench_triton_7_Sum.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            inspection = classify_optimize_workspace(operator, workspace)

            self.assertEqual(inspection.state, "partial-session")
            self.assertEqual(
                inspection.detail,
                "missing generated test harness for triton_7_Sum.py",
            )


if __name__ == "__main__":
    unittest.main()
