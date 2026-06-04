import json
import tempfile
import unittest
from pathlib import Path

from triton_agent.pattern_validation_loop.simulate_prompts import (
    SIMULATE_REPORT_FILENAME,
    build_simulate_plan_prompt,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SimulatePromptsTests(unittest.TestCase):
    def test_build_simulate_plan_prompt_requires_json_report_and_pattern_ranking(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            workspace = Path(tmp)
            operator = workspace / "demo.py"
            operator.write_text("def demo():\n    pass\n", encoding="utf-8")
            prompt = build_simulate_plan_prompt(
                operator_path=operator,
                workdir=workspace,
                test_mode="differential",
                bench_mode="standalone",
                target_chip="A5",
                optimize_target="kernel",
            )

        self.assertIn("SIMULATE OPTIMIZE PLAN", prompt)
        self.assertIn("PERF_PATTERN_SYNTHESIS.md", prompt)
        self.assertIn("validation-meta.json", prompt)
        self.assertIn("offline-eval-held", prompt)
        self.assertIn("pattern cards and operator code only", prompt)
        self.assertIn("ranked_patterns", prompt)
        self.assertIn("proposed_code_changes", prompt)
        self.assertIn("unified_diff", prompt)
        self.assertIn("code_plan_quality", prompt)
        self.assertIn("edits_by_pattern", prompt)
        self.assertIn(SIMULATE_REPORT_FILENAME, prompt)
        self.assertNotIn("grid-flatten-and-ub-buffering", prompt)
        self.assertIn("Do not create or update `baseline/`", prompt)
        self.assertIn("test_*.py.txt", prompt)


if __name__ == "__main__":
    unittest.main()
