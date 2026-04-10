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
            round_dir = self._create_round(Path(tmp))

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.PASS_CONTINUE)
            self.assertEqual(result.blocking_issues, ())

    def test_evaluate_round_gate_returns_pass_stop_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = self._create_round(Path(tmp))

            result = evaluate_round_gate(round_dir, stop_after_round=True)

            self.assertEqual(result.decision, GateDecision.PASS_STOP)

    def test_evaluate_round_gate_returns_revise_metadata_for_missing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = self._create_round(Path(tmp), include_summary=False)

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_METADATA)
            self.assertIn("missing summary.md", result.blocking_issues)

    def test_evaluate_round_gate_returns_revise_required_for_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = self._create_round(Path(tmp), evidence_sources=[])

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.REVISE_REQUIRED)
            self.assertIn("missing supporting evidence sources", result.blocking_issues)

    def test_evaluate_round_gate_returns_hard_fail_for_failed_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = self._create_round(Path(tmp), correctness_status="failed")

            result = evaluate_round_gate(round_dir)

            self.assertEqual(result.decision, GateDecision.HARD_FAIL)
            self.assertIn("correctness_status=failed", result.blocking_issues)

    def _create_round(
        self,
        root: Path,
        *,
        include_summary: bool = True,
        evidence_sources: Optional[List[str]] = None,
        correctness_status: str = "passed",
        benchmark_status: str = "passed",
    ) -> Path:
        round_dir = root / "opt-round-1"
        round_dir.mkdir()
        (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
        if include_summary:
            (round_dir / "summary.md").write_text("summary\n", encoding="utf-8")
        (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
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
                    "perf_artifact": "perf.txt",
                    "summary_path": "summary.md",
                    "opt_note_updated": True,
                    "next_recommendation": "continue",
                }
            ),
            encoding="utf-8",
        )
        return round_dir


if __name__ == "__main__":
    unittest.main()
