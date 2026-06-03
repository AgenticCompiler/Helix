import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "triton-npu-pattern-validation-loop"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS))

from audit_batch import collect_workspace_evidence, main

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class AuditBatchEvidenceTests(unittest.TestCase):
    def test_collect_includes_round_excerpts_and_heuristic_hint(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp) / "chunk_o"
            workspace.mkdir()
            (workspace / "validation-meta.json").write_text(
                json.dumps(
                    {
                        "workspace": "chunk_o",
                        "expected_patterns": ["grid-flatten-and-ub-buffering"],
                        "validation_target": "forward_chunk_o",
                    }
                ),
                encoding="utf-8",
            )
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "summary.md").write_text(
                "Applied grid-flatten-and-ub-buffering on UB tiling.\n",
                encoding="utf-8",
            )
            report = collect_workspace_evidence(workspace)

        self.assertTrue(report["heuristic_suggested_pass"])
        self.assertTrue(report["agent_review_required"])
        self.assertEqual(report["heuristic_pattern_hits"], ["grid-flatten-and-ub-buffering"])
        rounds = report["rounds"]
        self.assertEqual(len(rounds), 1)
        summary = rounds[0]["artifacts"]["summary.md"]
        self.assertTrue(summary["exists"])
        self.assertIn("grid-flatten", str(summary["excerpt"]))

    def test_main_exits_zero_when_patterns_missing(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch_root = Path(tmp)
            workspace = batch_root / "chunk_o"
            workspace.mkdir()
            (workspace / "validation-meta.json").write_text(
                json.dumps({"workspace": "chunk_o", "expected_patterns": ["missing-pattern"]}),
                encoding="utf-8",
            )
            round_dir = workspace / "opt-round-1"
            round_dir.mkdir()
            (round_dir / "summary.md").write_text("no pattern ids here\n", encoding="utf-8")
            code = main(["--batch-root", batch_root.as_posix()])

        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
