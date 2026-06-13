import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_agent.pattern_validation_loop.simulate_prompts import SIMULATE_PLAN_DIR
from triton_agent.pattern_validation_loop.simulate_plan import (
    WorkspaceSimulateResult,
    build_manual_optimize_command_hint,
    build_simulate_plan_config,
    remove_batch_workspace_simulate_plans,
    run_simulate_workspace_agents,
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

    def test_build_simulate_plan_config_accepts_missing_synthesis_file(self) -> None:
        missing = "tests/_simulate_missing_synthesis.md"
        config = build_simulate_plan_config(
            target_path=WORKSPACE_ROOT,
            synthesis_output=missing,
        )
        self.assertFalse(config.synthesis_path.is_file())

    def test_ensure_simulate_synthesis_ready_requires_synthesis_file(self) -> None:
        from triton_agent.pattern_validation_loop.simulate_plan import ensure_simulate_synthesis_ready

        missing = "tests/_simulate_missing_synthesis2.md"
        config = build_simulate_plan_config(
            target_path=WORKSPACE_ROOT,
            synthesis_output=missing,
            skip_extract=True,
        )
        with self.assertRaises(ValueError) as ctx:
            ensure_simulate_synthesis_ready(config)
        self.assertIn("Synthesis report not found", str(ctx.exception))

    def test_run_simulate_workspace_agents_runs_in_parallel_when_concurrency_gt_one(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            batch = Path(tmp)
            synth = WORKSPACE_ROOT / "tests/_simulate_parallel_synth.md"
            synth.write_text("# synth\n", encoding="utf-8")
            active = threading.active_count()
            peak = {"count": active}

            def _track_and_run(*_args, **_kwargs) -> int:
                peak["count"] = max(peak["count"], threading.active_count())
                return 0

            for name in ("alpha", "beta", "gamma"):
                workspace = batch / name
                workspace.mkdir()
                (workspace / f"{name}.py").write_text("def kernel():\n    pass\n", encoding="utf-8")

            config = build_simulate_plan_config(
                target_path=WORKSPACE_ROOT,
                batch_dir=batch.relative_to(WORKSPACE_ROOT).as_posix(),
                synthesis_output="tests/_simulate_parallel_synth.md",
                concurrency=2,
            )
            try:
                with patch(
                    "triton_agent.pattern_validation_loop.simulate_plan.run_simulate_plan_request",
                    side_effect=_track_and_run,
                ):
                    results, code = run_simulate_workspace_agents(config)
            finally:
                synth.unlink(missing_ok=True)

        self.assertEqual(code, 1)
        self.assertEqual(len(results), 3)
        self.assertGreater(peak["count"], active)

    def test_run_simulate_workspace_agents_reuses_existing_report_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT / "tests") as tmp:
            batch = Path(tmp)
            synth = WORKSPACE_ROOT / "tests/_simulate_reuse_synth.md"
            synth.write_text("# synth\n", encoding="utf-8")
            done = batch / "done"
            done.mkdir()
            (done / "done.py").write_text("def done():\n    pass\n", encoding="utf-8")
            plan_dir = done / SIMULATE_PLAN_DIR
            plan_dir.mkdir()
            (plan_dir / "report.json").write_text(
                json.dumps(
                    {
                        "ranked_patterns": [
                            {
                                "pattern_id": "grid-flatten-and-ub-buffering",
                                "priority": 1,
                                "hit": True,
                                "rationale": "visible",
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
                    },
                )
                + "\n",
                encoding="utf-8",
            )
            pending = batch / "pending"
            pending.mkdir()
            (pending / "pending.py").write_text("def pending():\n    pass\n", encoding="utf-8")

            config = build_simulate_plan_config(
                target_path=WORKSPACE_ROOT,
                batch_dir=batch.relative_to(WORKSPACE_ROOT).as_posix(),
                synthesis_output="tests/_simulate_reuse_synth.md",
            )
            try:
                with patch(
                    "triton_agent.pattern_validation_loop.simulate_plan.run_simulate_plan_request",
                    return_value=0,
                ) as run_mock:
                    results, code = run_simulate_workspace_agents(config)
            finally:
                synth.unlink(missing_ok=True)

        self.assertEqual(code, 1)
        self.assertEqual(run_mock.call_count, 1)
        by_name = {item.workspace.name: item for item in results}
        self.assertEqual(by_name["done"].status, "ok")
        self.assertIn("reused existing", by_name["done"].message)
        self.assertEqual(by_name["pending"].status, "failed")


if __name__ == "__main__":
    unittest.main()
