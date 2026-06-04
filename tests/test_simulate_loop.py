import unittest

from triton_agent.pattern_validation_loop.simulate_plan import batch_report_skills_aligned


class SimulateLoopTests(unittest.TestCase):
    def _aligned_concrete_report(self):
        return {
            "skills_alignment": "aligned",
            "code_plan_quality": "concrete",
            "proposed_code_changes": {
                "unified_diff": "--- a/op.py\n+++ b/op.py\n@@\n-x\n+y\n",
            },
        }

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
