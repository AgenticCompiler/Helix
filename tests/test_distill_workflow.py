import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.distill.agent import _prefixed_stream
from triton_agent.distill.git_repo_workspaces import GIT_REPO_PLAN_SKILL_NAME
from triton_agent.distill.models import DiscoveryResult
from triton_agent.distill.models import DistillConfig
from triton_agent.distill.git_repo_workspaces import build_workspace_plan_prompt
from triton_agent.distill.workflow import run_distill
from triton_agent.models import AgentResult


class DistillWorkflowTests(unittest.TestCase):
    def test_git_repo_plan_prompt_delegates_workflow_to_common_skill(self) -> None:
        prompt = build_workspace_plan_prompt(
            repo_root=Path("/repo"),
            language="tilelang",
            base_revision="origin/main",
            fork_revision="abc123",
            plan_path=Path("/repo/.triton-agent/workspace-plan.json"),
        )

        self.assertEqual(GIT_REPO_PLAN_SKILL_NAME, "ascend-npu-plan-git-operator-workspaces")
        self.assertIn("Use the staged ascend-npu-plan-git-operator-workspaces skill", prompt)
        self.assertIn("Operator language:\n  tilelang", prompt)
        self.assertIn("Fork point", prompt)
        self.assertIn("/repo/.triton-agent/workspace-plan.json", prompt)
        self.assertNotIn("### Step 2: For EACH changed file", prompt)
        self.assertNotIn("## What NOT to do", prompt)

    def test_prefixed_stream_marks_each_agent_output_line(self) -> None:
        stream = StringIO()
        prefixed = _prefixed_stream(stream, "[op] [simulate-iter-1/3]")

        prefixed.write("first\nsecond")
        prefixed.write(" continued\n")

        self.assertEqual(
            stream.getvalue(),
            "[op] [simulate-iter-1/3] first\n[op] [simulate-iter-1/3] second continued\n",
        )

    def test_not_aligned_analysis_triggers_next_iteration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op_dir = root / "op"
            op_dir.mkdir()
            (op_dir / "foo.py").write_text("x = 1\n", encoding="utf-8")
            (op_dir / "opt_foo.py").write_text("x = 2\n", encoding="utf-8")
            skills_dir = root / "skills"
            knowledge_dir = skills_dir / "triton-npu-optimize-knowledge"
            knowledge_dir.mkdir(parents=True)
            calls: list[dict[str, object]] = []
            analysis_count = 0

            def agent_runner(**kwargs: object) -> AgentResult:
                nonlocal analysis_count
                prompt = str(kwargs["prompt"])
                workdir = Path(kwargs["workdir"])  # type: ignore[arg-type]
                calls.append(kwargs)
                output_label = str(kwargs["output_label"])
                if output_label == "[op] [distill]":
                    self.assertEqual(kwargs["output_label"], "[op] [distill]")
                    self.assertIn("ascend-npu-distill-patterns", prompt)
                    _json_path_from_prompt(prompt).write_text(
                        json.dumps(
                            {
                                "matched_patterns": ["tiling"],
                                "updated_patterns": ["tiling"],
                                "summary": "uses tiling",
                            }
                        ),
                        encoding="utf-8",
                    )
                elif "[simulate-iter-" in output_label:
                    self.assertIn("[op] [simulate-iter-", str(kwargs["output_label"]))
                    self.assertIn("ascend-npu-distill-patterns", prompt)
                    self.assertNotIn("opt_foo.py", prompt)
                    self.assertNotIn("Unified diff", prompt)
                    self.assertIn("tiling", prompt)
                    if "simulate-foo-2.json" in prompt:
                        self.assertFalse(
                            (workdir / "generated_foo.py").exists(),
                            "unaligned candidate should be deleted before the next simulate iteration",
                        )
                    (workdir / "generated_foo.py").write_text("x = 2\n", encoding="utf-8")
                    _json_path_from_prompt(prompt).write_text(
                        json.dumps({"summary": "generated", "applied_patterns": ["tiling"]}),
                        encoding="utf-8",
                    )
                elif "[analyze-iter-" in output_label:
                    self.assertIn("[op] [analyze-iter-", str(kwargs["output_label"]))
                    self.assertIn("ascend-npu-distill-patterns", prompt)
                    analysis_count += 1
                    _json_path_from_prompt(prompt).write_text(
                        json.dumps(
                            {
                                "aligned": analysis_count == 2,
                                "summary": "ok" if analysis_count == 2 else "needs more guidance",
                                "updated_patterns": ["loop-invariant-hoisting"] if analysis_count == 1 else [],
                                "skill_updates": ["tiling"],
                            }
                        ),
                        encoding="utf-8",
                    )
                return AgentResult(return_code=0, stdout="", stderr="")

            config = DistillConfig(
                input_root=root,
                skills_dir=skills_dir,
                update_skills_dir=root / "update_skills",
                source="code-diff",
                agent_name="codex",
                max_iterations=2,
                concurrency=1,
                stream_output=False,
                verbose=False,
                force=False,
                skip_existing=False,
                promote_converged_skills=True,
            )

            with (
                patch(
                    "triton_agent.distill.workflow.ensure_editable_knowledge_skill",
                    return_value=knowledge_dir,
                ),
                patch("triton_agent.distill.workflow.rebuild_pattern_index"),
                patch("triton_agent.distill.workflow.promote_editable_knowledge_skill") as promote,
            ):
                promote.return_value = knowledge_dir
                results = run_distill(config, agent_runner=agent_runner)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "aligned")
            self.assertEqual(len(results[0].iterations), 2)
            self.assertEqual(results[0].matched_patterns, ["tiling", "loop-invariant-hoisting"])
            self.assertEqual(results[0].updated_patterns, ["tiling", "loop-invariant-hoisting"])
            self.assertTrue((op_dir / "simulate" / "foo.py").exists())
            self.assertTrue((op_dir / "simulate" / "generated_foo.py").exists())
            report = json.loads((op_dir / "simulate" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "aligned")
            self.assertEqual(report["updated_patterns"], ["tiling", "loop-invariant-hoisting"])
            self.assertEqual(report["iterations"][0]["updated_patterns"], ["loop-invariant-hoisting"])
            self.assertTrue(all(call.get("skills_root") == skills_dir for call in calls))
            simulate_calls = [
                call for call in calls if "[simulate-iter-" in str(call.get("output_label"))
            ]
            self.assertEqual(len(simulate_calls), 2)
            promote.assert_called_once_with(knowledge_dir, language="triton")

    def test_git_repo_plan_agent_receives_staged_skill_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skills_dir = root / "skills"
            knowledge_dir = skills_dir / "triton-npu-optimize-knowledge"
            knowledge_dir.mkdir(parents=True)
            calls: list[dict[str, object]] = []

            def agent_runner(**kwargs: object) -> AgentResult:
                calls.append(kwargs)
                self.assertEqual(kwargs["output_label"], "[git-repo]")
                self.assertEqual(kwargs["skills_root"], skills_dir)
                self.assertIn(
                    "Use the staged ascend-npu-plan-git-operator-workspaces skill",
                    str(kwargs["prompt"]),
                )
                (root / ".triton-agent" / "workspace-plan.json").write_text(
                    json.dumps({"schema_version": 1, "operators": []}) + "\n",
                    encoding="utf-8",
                )
                return AgentResult(return_code=0, stdout="", stderr="")

            config = DistillConfig(
                input_root=root,
                skills_dir=skills_dir,
                update_skills_dir=root / "update_skills",
                source="git-repo",
                agent_name="opencode",
                max_iterations=1,
                concurrency=1,
                stream_output=False,
                verbose=False,
                force=False,
                skip_existing=False,
                promote_converged_skills=False,
            )

            with (
                patch(
                    "triton_agent.distill.workflow.detect_git_worktree",
                    return_value=(root, "headsha"),
                ),
                patch(
                    "triton_agent.distill.workflow.detect_default_base_branch",
                    return_value="origin/main",
                ),
                patch(
                    "triton_agent.distill.workflow.compute_fork_point",
                    return_value="abc123",
                ),
                patch("triton_agent.distill.workflow.scaffold_operator_workspaces", return_value=0),
                patch(
                    "triton_agent.distill.workflow.operator_workspaces_created",
                    return_value=True,
                ),
                patch(
                    "triton_agent.distill.workflow.discover_operator_pairs",
                    return_value=DiscoveryResult(pairs=(), skips=()),
                ),
                patch(
                    "triton_agent.distill.workflow.ensure_editable_knowledge_skill",
                    return_value=knowledge_dir,
                ),
                patch(
                    "triton_agent.distill.workflow.snapshot_pattern_card_texts",
                    return_value={},
                ),
            ):
                results = run_distill(config, agent_runner=agent_runner)

            self.assertEqual(results, [])
            self.assertEqual(len(calls), 1)


def _json_path_from_prompt(prompt: str) -> Path:
    for line in prompt.splitlines():
        if line.startswith("Write JSON to "):
            value = line.removeprefix("Write JSON to ").removesuffix(" with this shape:")
            return Path(value)
        if line.startswith("Also write JSON to "):
            value = line.removeprefix("Also write JSON to ").removesuffix(" with this shape:")
            return Path(value)
    raise AssertionError("prompt did not include JSON output path")


if __name__ == "__main__":
    unittest.main()
