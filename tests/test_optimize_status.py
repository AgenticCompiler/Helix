import sys
import tempfile
import unittest
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.status import (
    inspect_optimize_status_workspace,
    parse_logged_best_round,
    workspace_has_optimize_artifacts,
)


class OptimizeStatusTests(unittest.TestCase):
    def _write_verify_state(
        self,
        workspace: Path,
        verify_name: str,
        *,
        test_status: str = "passed",
        baseline_bench_status: str = "passed",
        best_bench_status: str = "passed",
        compare_status: str = "passed",
    ) -> Path:
        verify_dir = workspace / "opt-verify" / verify_name
        verify_dir.mkdir(parents=True)
        state_path = verify_dir / "verify-state.json"
        state_path.write_text(
            json.dumps(
                {
                    "verify-result": {
                        "test": {"status": test_status},
                        "rerun_baseline_bench": {"status": baseline_bench_status},
                        "rerun_best_bench": {"status": best_bench_status},
                        "compare_perf": {"status": compare_status},
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return state_path

    def test_parse_logged_best_round_prefers_overall_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "opt-note.md"
            note.write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: current best",
                        "## Round 2",
                        "Best status: validated branch",
                        "",
                        "## Overall Summary",
                        "Final best round: round-2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            logged_best = parse_logged_best_round(note)

            self.assertEqual(logged_best, "round-2")

    def test_parse_logged_best_round_uses_latest_current_best_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "opt-note.md"
            note.write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: current best",
                        "## Round 2",
                        "Best status: validated branch",
                        "## Round 3",
                        "Best status: current best",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            logged_best = parse_logged_best_round(note)

            self.assertEqual(logged_best, "round-3")

    def test_inspect_optimize_status_workspace_returns_numeric_best_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt-note.md").write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: validated branch",
                        "## Round 2",
                        "Best status: validated branch",
                        "## Round 3",
                        "Best status: current best",
                        "",
                        "## Overall Summary",
                        "Final best round: round-1",
                        "Geomean speedup: 1.16x",
                        "Total speedup: 1.18x",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (round_two / "perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertEqual(status.best_round, "round-2")
            self.assertEqual(status.logged_best, "round-1")
            self.assertAlmostEqual(status.baseline_mean or 0.0, 15.0)
            self.assertAlmostEqual(status.best_mean or 0.0, 9.5)
            self.assertAlmostEqual(status.avg_improvement or 0.0, 0.3)
            self.assertAlmostEqual(status.geomean_speedup or 0.0, (10 / 9 * 20 / 10) ** 0.5)
            self.assertAlmostEqual(status.total_speedup or 0.0, 30 / 19)
            self.assertIn(
                "numeric best round != logged best. "
                "computed speedup: 1.49x, 1.58x; "
                "logged speedup: 1.16x, 1.18x",
                status.warnings,
            )

    def test_inspect_optimize_status_workspace_prefers_overall_summary_and_warns_on_legacy_mismatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt-note.md").write_text(
                "\n".join(
                    [
                        "## Round 1",
                        "Best status: current best",
                        "## Round 2",
                        "Best status: validated branch",
                        "",
                        "## Overall Summary",
                        "Final best round: round-2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            (round_two / "perf.txt").write_text(
                "latency-a: 9\nlatency-b: 10\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertEqual(status.best_round, "round-2")
            self.assertEqual(status.logged_best, "round-2")
            self.assertIn(
                "overall summary best round differs from legacy current best marker",
                status.warnings,
            )

    def test_inspect_optimize_status_workspace_reports_no_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "no-session")
            self.assertIsNone(status.best_round)
            self.assertEqual(status.warnings, ())
            self.assertIsNone(status.latest_verify_state)
            self.assertFalse(status.verified)

    def test_inspect_optimize_status_workspace_uses_latest_successful_verify_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            self._write_verify_state(
                workspace,
                "verify-20260421-100000",
                compare_status="failed",
            )
            latest_state = self._write_verify_state(
                workspace,
                "verify-20260421-120000",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertEqual(status.latest_verify_state, latest_state)
            self.assertTrue(status.verified)

    def test_inspect_optimize_status_workspace_marks_partial_latest_verify_as_unverified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            latest_state = self._write_verify_state(
                workspace,
                "verify-20260421-120000",
                compare_status="failed",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.latest_verify_state, latest_state)
            self.assertFalse(status.verified)

    def test_inspect_optimize_status_workspace_ignores_extra_round_perf_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 7\nmean_ms: 11.0\nlatency-b: 15\nnotes: strong round\n",
                encoding="utf-8",
            )
            (round_two / "perf.txt").write_text(
                "latency-a: 9\nlatency-b: 19\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertEqual(status.best_round, "round-1")
            self.assertAlmostEqual(status.best_mean or 0.0, 11.0)
            self.assertAlmostEqual(status.avg_improvement or 0.0, 0.275)
            self.assertAlmostEqual(status.geomean_speedup or 0.0, (10 / 7 * 20 / 15) ** 0.5)
            self.assertAlmostEqual(status.total_speedup or 0.0, 30 / 22)
            self.assertEqual(status.warnings, ())

    def test_inspect_optimize_status_workspace_prefers_best_geomean_speedup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 1\nlatency-b: 100\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_two = workspace / "opt-round-2"
            round_one.mkdir()
            round_two.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 0.5\nlatency-b: 100\n",
                encoding="utf-8",
            )
            (round_two / "perf.txt").write_text(
                "latency-a: 0.9\nlatency-b: 60\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertEqual(status.best_round, "round-1")
            self.assertAlmostEqual(status.avg_improvement or 0.0, 0.25)
            self.assertAlmostEqual(status.geomean_speedup or 0.0, (1 / 0.5 * 100 / 100) ** 0.5)
            self.assertAlmostEqual(status.total_speedup or 0.0, 101 / 100.5)

    def test_inspect_optimize_status_workspace_prefers_baseline_directory_perf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (baseline_dir / "state.json").write_text(
                "\n".join(
                    [
                        "{",
                        '  "baseline_kind": "prepared",',
                        '  "source_operator": "kernel.py",',
                        '  "baseline_operator": "baseline/kernel.py",',
                        '  "test_file": "differential_test_kernel.py",',
                        '  "test_mode": "differential",',
                        '  "bench_file": "bench_kernel.py",',
                        '  "bench_mode": "standalone",',
                        '  "perf_artifact": "baseline/perf.txt",',
                        '  "correctness_status": "passed",',
                        '  "benchmark_status": "passed",',
                        '  "baseline_established": true',
                        "}",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 999\nlatency-b: 999\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertAlmostEqual(status.baseline_mean or 0.0, 15.0)

    def test_inspect_optimize_status_workspace_prefers_non_opt_baseline_perf_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "opt_kernel_perf.txt").write_text(
                "latency-a: 8\nlatency-b: 18\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 9\nlatency-b: 15\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertEqual(status.best_round, "round-1")
            self.assertNotIn("found multiple baseline perf files", status.warnings)

    def test_inspect_optimize_status_workspace_prefers_operator_named_baseline_perf_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Gemm.py").write_text("print('x')\n", encoding="utf-8")
            (workspace / "baseline_perf.txt").write_text(
                "latency-a: 100\nlatency-b: 100\n",
                encoding="utf-8",
            )
            (workspace / "Gemm_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertAlmostEqual(status.baseline_mean or 0.0, 15.0)
            self.assertEqual(status.best_round, "round-1")
            self.assertEqual(status.warnings, ())

    def test_inspect_optimize_status_workspace_falls_back_to_baseline_perf_txt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "Gemm.py").write_text("print('x')\n", encoding="utf-8")
            (workspace / "baseline_perf.txt").write_text(
                "latency-a: 10\nlatency-b: 20\n",
                encoding="utf-8",
            )
            (workspace / "candidate_perf.txt").write_text(
                "latency-a: 100\nlatency-b: 100\n",
                encoding="utf-8",
            )
            round_one = workspace / "opt-round-1"
            round_one.mkdir()
            (round_one / "perf.txt").write_text(
                "latency-a: 8\nlatency-b: 16\n",
                encoding="utf-8",
            )

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "ok")
            self.assertAlmostEqual(status.baseline_mean or 0.0, 15.0)
            self.assertNotIn("found multiple baseline perf files", status.warnings)

    def test_inspect_optimize_status_workspace_ambiguous_baseline_does_not_repeat_missing_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_a_perf.txt").write_text("latency-a: 10\n", encoding="utf-8")
            (workspace / "kernel_b_perf.txt").write_text("latency-a: 11\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()

            status = inspect_optimize_status_workspace(workspace)

            self.assertEqual(status.state, "warning")
            self.assertIn("found multiple baseline perf files", status.warnings)
            self.assertNotIn("missing baseline perf data", status.warnings)

    def test_workspace_has_optimize_artifacts_detects_single_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel_perf.txt").write_text("latency-a: 10\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()

            self.assertTrue(workspace_has_optimize_artifacts(workspace))


if __name__ == "__main__":
    unittest.main()
