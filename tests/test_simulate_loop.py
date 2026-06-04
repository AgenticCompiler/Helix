import unittest

from triton_agent.pattern_validation_loop.simulate_plan import batch_report_skills_aligned


class SimulateLoopTests(unittest.TestCase):
    def test_batch_report_skills_aligned_requires_all_ok_and_aligned(self) -> None:
        payload = {
            "workspaces": [
                {
                    "status": "ok",
                    "simulate_report": {"skills_alignment": "aligned"},
                },
                {
                    "status": "ok",
                    "simulate_report": {"skills_alignment": "aligned"},
                },
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


if __name__ == "__main__":
    unittest.main()
