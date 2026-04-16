import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize import checks as optimize_checks


class OptimizeCheckTests(unittest.TestCase):
    def test_optimize_checks_delegate_to_optimize_check_script_module(self) -> None:
        module = SimpleNamespace(
            check_baseline=lambda path: {
                "ok": False,
                "kind": "baseline",
                "decision": "revise-required",
                "issues": ("baseline issue",),
                "summary": f"checked {path.name}",
            },
            check_round=lambda path: SimpleNamespace(
                ok=True,
                kind="round",
                decision="pass",
                issues=(),
                summary=f"checked {path.name}",
            ),
        )
        with patch("triton_agent.optimize.checks.load_skill_script_module", return_value=module) as mocked:
            baseline_result = optimize_checks.check_baseline(Path("/tmp/baseline"))
            round_result = optimize_checks.check_round(Path("/tmp/opt-round-1"))

        self.assertFalse(baseline_result.ok)
        self.assertEqual(baseline_result.kind, "baseline")
        self.assertEqual(baseline_result.decision, "revise-required")
        self.assertEqual(baseline_result.issues, ("baseline issue",))
        self.assertEqual(baseline_result.summary, "checked baseline")
        self.assertTrue(round_result.ok)
        self.assertEqual(round_result.kind, "round")
        self.assertEqual(round_result.decision, "pass")
        self.assertEqual(round_result.summary, "checked opt-round-1")
        mocked.assert_any_call("triton-npu-optimize-check", "optimize_check")

    def test_check_baseline_reports_missing_perf_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            baseline_dir = workdir / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "prepared",
                        "source_operator": "kernel.py",
                        "baseline_operator": "baseline/kernel.py",
                        "test_file": "differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "bench_kernel.py",
                        "bench_mode": "standalone",
                        "perf_artifact": "baseline/perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

            result = optimize_checks.check_baseline(baseline_dir)

            self.assertFalse(result.ok)
            self.assertEqual(result.kind, "baseline")
            self.assertEqual(result.decision, "revise-required")
            self.assertIn("missing baseline/perf.txt", result.issues)

    def test_check_round_passes_with_complete_round_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            self._write_baseline(workdir)
            round_dir = self._write_round(workdir, "opt-round-1", next_recommendation="continue")

            result = optimize_checks.check_round(round_dir)

            self.assertTrue(result.ok)
            self.assertEqual(result.kind, "round")
            self.assertEqual(result.decision, "pass")
            self.assertEqual(result.issues, ())

    def _write_baseline(self, workdir: Path) -> None:
        baseline_dir = workdir / "baseline"
        baseline_dir.mkdir(exist_ok=True)
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "prepared",
                    "source_operator": "kernel.py",
                    "baseline_operator": "baseline/kernel.py",
                    "test_file": "differential_test_kernel.py",
                    "test_mode": "differential",
                    "bench_file": "bench_kernel.py",
                    "bench_mode": "standalone",
                    "perf_artifact": "baseline/perf.txt",
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
        (baseline_dir / "kernel.py").write_text("print('baseline')\n", encoding="utf-8")

    def _write_round(self, workdir: Path, round_name: str, *, next_recommendation: str) -> Path:
        round_dir = workdir / round_name
        round_dir.mkdir(exist_ok=True)
        (workdir / "opt-note.md").write_text("## Round\n", encoding="utf-8")
        (round_dir / "kernel.py").write_text(f"print('{round_name}')\n", encoding="utf-8")
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": round_name,
                    "parent_round": "round-0",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"],
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "perf_artifact": "perf.txt",
                    "canonical_baseline": "baseline",
                    "comparison_target": "baseline/perf.txt",
                    "perf_summary_source": "compare-perf",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                    "next_recommendation": next_recommendation,
                }
            ),
            encoding="utf-8",
        )
        return round_dir


if __name__ == "__main__":
    unittest.main()
