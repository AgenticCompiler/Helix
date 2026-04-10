import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.round_contract import inspect_round_artifacts, load_round_state


class OptimizeRoundContractTests(unittest.TestCase):
    def test_load_round_state_requires_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "round-state.json").write_text(
                json.dumps({"round": "opt-round-1"}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as ctx:
                load_round_state(round_dir)

            self.assertIn("missing required round-state fields", str(ctx.exception))

    def test_inspect_round_artifacts_flags_missing_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            round_dir = Path(tmp) / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "attempts.md").write_text("attempts\n", encoding="utf-8")
            (round_dir / "perf.txt").write_text("case0: 1.0\n", encoding="utf-8")
            (round_dir / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            (round_dir / "round-state.json").write_text(
                json.dumps(
                    {
                        "round": "opt-round-1",
                        "parent_round": "round-0",
                        "hypothesis": "vectorize loads",
                        "evidence_sources": ["benchmark"],
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "perf_artifact": "perf.txt",
                        "summary_path": "summary.md",
                        "opt_note_updated": True,
                        "next_recommendation": "continue",
                    }
                ),
                encoding="utf-8",
            )

            result = inspect_round_artifacts(round_dir)

            self.assertIn("missing summary.md", result.issues)


if __name__ == "__main__":
    unittest.main()
