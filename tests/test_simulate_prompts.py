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
            (workspace / "validation-meta.json").write_text(
                json.dumps(
                    {
                        "workspace": "demo",
                        "expected_patterns": ["grid-flatten-and-ub-buffering"],
                    },
                )
                + "\n",
                encoding="utf-8",
            )
            synthesis = workspace / "PERF_PATTERN_SYNTHESIS.md"
            knowledge = workspace / "PERF_KNOWLEDGE_BASE.md"
            synthesis.write_text("# synthesis\n", encoding="utf-8")
            knowledge.write_text("# knowledge\n", encoding="utf-8")
            prompt = build_simulate_plan_prompt(
                operator_path=operator,
                workdir=workspace,
                test_mode="differential",
                bench_mode="standalone",
                target_chip="A5",
                optimize_target="kernel",
                validation_meta=json.loads(
                    (workspace / "validation-meta.json").read_text(encoding="utf-8"),
                ),
                synthesis_path=synthesis,
                knowledge_path=knowledge,
            )

        self.assertIn("SIMULATE OPTIMIZE PLAN", prompt)
        self.assertIn("PERF_PATTERN_SYNTHESIS.md", prompt)
        self.assertIn("PERF_KNOWLEDGE_BASE.md", prompt)
        self.assertIn("ranked_patterns", prompt)
        self.assertIn(SIMULATE_REPORT_FILENAME, prompt)
        self.assertIn("grid-flatten-and-ub-buffering", prompt)
        self.assertIn("Do not create or update `baseline/`", prompt)
        self.assertIn("test_*.py.txt", prompt)


if __name__ == "__main__":
    unittest.main()
