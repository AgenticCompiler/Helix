import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.commands.pattern_validation_plan import handle_pattern_validation_plan

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class PatternValidationPlanCliTests(unittest.TestCase):
    def test_handle_generates_workspace_plan(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            repo = Path(tmp)
            knowledge = repo / "PERF_KNOWLEDGE_BASE.md"
            knowledge.write_text("## File Analyses\n", encoding="utf-8")
            batch = repo / "pattern-validation-batch"
            batch.mkdir()
            parser = unittest.mock.MagicMock()
            args = unittest.mock.MagicMock(
                input=repo.as_posix(),
                knowledge="PERF_KNOWLEDGE_BASE.md",
                batch_dir="pattern-validation-batch",
                output="",
                base="origin/main",
            )

            payload = {
                "workspace_count": 1,
                "warnings": [],
                "workspaces": [{"workspace": "demo_kernel"}],
            }

            with patch(
                "triton_agent.commands.pattern_validation_plan.generate_workspace_plan",
                return_value=(payload, []),
            ):
                code = handle_pattern_validation_plan(parser, args)

            self.assertEqual(code, 0)
            plan_path = batch / "workspace-plan.json"
            self.assertTrue(plan_path.is_file())
            self.assertEqual(json.loads(plan_path.read_text(encoding="utf-8"))["workspace_count"], 1)


if __name__ == "__main__":
    unittest.main()
