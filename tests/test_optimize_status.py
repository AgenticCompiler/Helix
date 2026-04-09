import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.status import inspect_optimize_status_workspace, parse_logged_best_round


class OptimizeStatusTests(unittest.TestCase):
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
                        "Best status: current best",
                        "## Round 2",
                        "Best status: validated branch",
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
            self.assertIn("numeric best round differs from logged best round", status.warnings)

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


if __name__ == "__main__":
    unittest.main()
