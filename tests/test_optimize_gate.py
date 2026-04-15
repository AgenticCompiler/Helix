import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.gate import GateDecision, evaluate_round_gate


class OptimizeGateTests(unittest.TestCase):
    def test_evaluate_round_gate_returns_pass_continue_for_valid_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root)

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.PASS_CONTINUE)
            self.assertEqual(result.blocking_issues, ())

    def test_evaluate_round_gate_returns_pass_stop_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root)

            result = evaluate_round_gate(round_dir, stop_after_round=True)

            self.assertEqual(result.decision, GateDecision.PASS_STOP)

    def test_evaluate_round_gate_returns_revise_metadata_for_missing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root, include_summary=False)

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_METADATA)
            self.assertIn("missing summary.md", result.blocking_issues)

    def test_evaluate_round_gate_returns_revise_required_for_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root, evidence_sources=[])

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_REQUIRED)
            self.assertIn("missing supporting evidence sources", result.blocking_issues)

    def test_evaluate_round_gate_returns_hard_fail_for_failed_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root, correctness_status="failed")

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.HARD_FAIL)
            self.assertIn("correctness_status=failed", result.blocking_issues)

    def test_evaluate_round_gate_requires_established_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = self._create_round(Path(tmp))

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_REQUIRED)
            self.assertIn("missing baseline/state.json", result.blocking_issues)

    def test_evaluate_round_gate_requires_canonical_baseline_comparison_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root, comparison_target="opt-round-0/perf.txt")

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_REQUIRED)
            self.assertIn("comparison_target=opt-round-0/perf.txt", result.blocking_issues)

    def test_evaluate_round_gate_requires_compare_perf_summary_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(root)
            round_dir = self._create_round(root, perf_summary_source="hand-calculated")

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_REQUIRED)
            self.assertIn("perf_summary_source=hand-calculated", result.blocking_issues)

    def test_evaluate_round_gate_accepts_state_declared_round_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._create_baseline(
                root,
                baseline_operator="baseline/snapshots/chosen.py",
                perf_artifact="baseline/metrics/perf.txt",
            )
            round_dir = self._create_round(
                root,
                summary_path="reports/final.md",
                perf_artifact="bench/candidate_perf.txt",
            )

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.PASS_CONTINUE)
            self.assertEqual(result.blocking_issues, ())

    def _create_round(
        self,
        root: Path,
        *,
        include_summary: bool = True,
        evidence_sources: Optional[List[str]] = None,
        correctness_status: str = "passed",
        benchmark_status: str = "passed",
        comparison_target: str = "baseline/perf.txt",
        perf_summary_source: str = "compare-perf",
        summary_path: str = "summary.md",
        perf_artifact: str = "perf.txt",
    ) -> Path:
        round_dir = root / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        if include_summary:
            summary_file = round_dir / summary_path
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            summary_file.write_text("summary\n", encoding="utf-8")
        perf_file = round_dir / perf_artifact
        perf_file.parent.mkdir(parents=True, exist_ok=True)
        perf_file.write_text("case0: 1.0\n", encoding="utf-8")
        (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
        (round_dir / "round-state.json").write_text(
            json.dumps(
                {
                    "round": "opt-round-1",
                    "parent_round": "round-0",
                    "hypothesis": "vectorize loads",
                    "evidence_sources": ["benchmark"] if evidence_sources is None else evidence_sources,
                    "correctness_status": correctness_status,
                    "benchmark_status": benchmark_status,
                    "perf_artifact": perf_artifact,
                    "canonical_baseline": "baseline",
                    "comparison_target": comparison_target,
                    "perf_summary_source": perf_summary_source,
                    "summary_path": summary_path,
                    "opt_note_updated": True,
                    "next_recommendation": "continue",
                }
            ),
            encoding="utf-8",
        )
        return round_dir

    def _create_baseline(
        self,
        root: Path,
        *,
        baseline_operator: str = "baseline/kernel.py",
        perf_artifact: str = "baseline/perf.txt",
    ) -> None:
        baseline_dir = root / "baseline"
        baseline_dir.mkdir()
        (baseline_dir / "state.json").write_text(
            json.dumps(
                {
                    "baseline_kind": "prepared",
                    "source_operator": "kernel.py",
                    "baseline_operator": baseline_operator,
                    "test_file": "differential_test_kernel.py",
                    "test_mode": "differential",
                    "bench_file": "bench_kernel.py",
                    "bench_mode": "standalone",
                    "perf_artifact": perf_artifact,
                    "correctness_status": "passed",
                    "benchmark_status": "passed",
                    "baseline_established": True,
                }
            ),
            encoding="utf-8",
        )
        perf_file = root / perf_artifact
        perf_file.parent.mkdir(parents=True, exist_ok=True)
        perf_file.write_text("latency-a: 1.0\n", encoding="utf-8")
        operator_file = root / baseline_operator
        operator_file.parent.mkdir(parents=True, exist_ok=True)
        operator_file.write_text("print('baseline')\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
