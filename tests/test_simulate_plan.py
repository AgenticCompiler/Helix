import json
import tempfile
import unittest
from pathlib import Path

from triton_agent.pattern_validation_loop.simulate_prompts import SIMULATE_PLAN_DIR
from triton_agent.pattern_validation_loop.simulate_plan import (
    WorkspaceSimulateResult,
    build_manual_optimize_command_hint,
    build_simulate_plan_config,
    remove_batch_workspace_simulate_plans,
    write_batch_simulate_report,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


class SimulatePlanTests(unittest.TestCase):
    def test_write_batch_simulate_report_aggregates_workspace_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch = Path(tmp)
            workspace = batch / "demo"
            plan_dir = workspace / "simulate-plan"
            plan_dir.mkdir(parents=True)
            (plan_dir / "report.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "workspace": "demo",
                        "ranked_patterns": [
                            {
                                "pattern_id": "grid-flatten-and-ub-buffering",
                                "priority": 1,
                                "hit": True,
                                "rationale": "grid flatten visible",
                            },
                        ],
                        "skills_alignment": "aligned",
                    },
                )
                + "\n",
                encoding="utf-8",
            )
            results = [
                WorkspaceSimulateResult(
                    workspace=workspace,
                    status="ok",
                    message="ok",
                    report_path=plan_dir / "report.json",
                ),
            ]
            path = write_batch_simulate_report(batch, results)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["ok_count"], 1)
        self.assertIn("next_step_manual_optimize", payload)
        self.assertEqual(
            payload["workspaces"][0]["simulate_report"]["ranked_patterns"][0]["pattern_id"],
            "grid-flatten-and-ub-buffering",
        )

    def test_manual_optimize_hint_mentions_optimize_batch(self) -> None:
        hint = build_manual_optimize_command_hint(Path("/tmp/batch"))
        self.assertIn("optimize-batch", hint)
        self.assertIn("/tmp/batch", hint)

    def test_remove_batch_workspace_simulate_plans_deletes_active_workspace_dirs(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as tmp:
            batch = Path(tmp)
            workspace = batch / "demo"
            workspace.mkdir()
            (workspace / "demo.py").write_text("def demo():\n    pass\n", encoding="utf-8")
            plan_dir = workspace / SIMULATE_PLAN_DIR
            plan_dir.mkdir()
            (plan_dir / "report.json").write_text("{}\n", encoding="utf-8")

            removed = remove_batch_workspace_simulate_plans(batch)

        self.assertEqual(removed, ["demo"])
        self.assertFalse(plan_dir.exists())

    def test_build_simulate_plan_config_requires_synthesis_file(self) -> None:
        missing = "tests/_simulate_missing_synthesis.md"
        with self.assertRaises(ValueError) as ctx:
            build_simulate_plan_config(
                target_path=WORKSPACE_ROOT,
                synthesis_output=missing,
            )
        self.assertIn("Synthesis report not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
