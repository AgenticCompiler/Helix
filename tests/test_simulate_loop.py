import unittest

from triton_agent.pattern_validation_loop.simulate_plan import (
    batch_report_skills_aligned,
    validate_simulate_report,
)


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


if __name__ == "__main__":
    unittest.main()
