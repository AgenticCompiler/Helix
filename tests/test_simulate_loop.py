import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.pattern_validation_loop.simulate_loop import run_commit_perf_extraction_if_needed
from triton_agent.pattern_validation_loop.simulate_plan import (
    batch_report_skills_aligned,
    build_simulate_plan_config,
    validate_simulate_report,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SimulateLoopTests(unittest.TestCase):
    def _aligned_concrete_report(self):
        return {
            "ranked_patterns": [
                {
                    "pattern_id": "grid-flatten-and-ub-buffering",
                    "priority": 1,
                    "hit": True,
                    "rationale": "visible in code",
                },
            ],
            "proposed_code_changes": {
                "unified_diff": "--- a/op.py\n+++ b/op.py\n@@\n-old\n+new\n",
                "edits_by_pattern": [
                    {
                        "pattern_id": "grid-flatten-and-ub-buffering",
                        "before_excerpt": "old",
                        "after_excerpt": "new",
                    },
                ],
            },
            "skills_alignment": "aligned",
            "code_plan_quality": "concrete",
        }

    def test_validate_simulate_report_requires_concrete_diff_and_edits(self) -> None:
        report = self._aligned_concrete_report()
        self.assertEqual(validate_simulate_report(report), [])
        bad = dict(report)
        bad["proposed_code_changes"] = {"unified_diff": "not a diff", "edits_by_pattern": []}
        self.assertTrue(validate_simulate_report(bad))

    def test_batch_report_skills_aligned_requires_all_ok_and_aligned(self) -> None:
        report = self._aligned_concrete_report()
        payload = {
            "workspaces": [
                {"status": "ok", "simulate_report": report},
                {"status": "ok", "simulate_report": report},
            ],
        }
        self.assertTrue(batch_report_skills_aligned(payload))

    def test_batch_report_skills_aligned_fails_on_mismatch(self) -> None:
        payload = {
            "workspaces": [
                {
                    "status": "ok",
                    "simulate_report": {"skills_alignment": "partial"},
                },
            ],
        }
        self.assertFalse(batch_report_skills_aligned(payload))

    def test_batch_report_skills_aligned_fails_without_concrete_code_plan(self) -> None:
        payload = {
            "workspaces": [
                {
                    "status": "ok",
                    "simulate_report": {
                        "skills_alignment": "aligned",
                        "code_plan_quality": "vague",
                    },
                },
            ],
        }
        self.assertFalse(batch_report_skills_aligned(payload))

    def test_run_commit_perf_extraction_skips_when_reports_exist(self) -> None:
        synth = WORKSPACE_ROOT / "tests/_simulate_extract_synth.md"
        knowledge = WORKSPACE_ROOT / "tests/_simulate_extract_knowledge.md"
        synth.write_text("# synth\n", encoding="utf-8")
        knowledge.write_text("# knowledge\n", encoding="utf-8")
        try:
            config = build_simulate_plan_config(
                target_path=WORKSPACE_ROOT,
                synthesis_output="tests/_simulate_extract_synth.md",
                knowledge_base="tests/_simulate_extract_knowledge.md",
            )
            with patch(
                "triton_agent.commit_perf_analysis.launcher.run_commit_perf_analysis",
            ) as extract_mock:
                code = run_commit_perf_extraction_if_needed(config)
            extract_mock.assert_not_called()
        finally:
            synth.unlink(missing_ok=True)
            knowledge.unlink(missing_ok=True)
        self.assertEqual(code, 0)

    def test_run_commit_perf_extraction_runs_when_synthesis_missing(self) -> None:
        config = build_simulate_plan_config(
            target_path=WORKSPACE_ROOT,
            synthesis_output="tests/_simulate_extract_missing_synth.md",
            knowledge_base="tests/_simulate_extract_missing_knowledge.md",
        )
        with patch(
            "triton_agent.commit_perf_analysis.launcher.run_commit_perf_analysis",
            return_value=0,
        ) as extract_mock:
            code = run_commit_perf_extraction_if_needed(config)
        extract_mock.assert_called_once()
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
