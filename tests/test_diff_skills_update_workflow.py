import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.diff_skills_update.agent import _prefixed_stream
from triton_agent.diff_skills_update.models import DiffSkillsUpdateConfig
from triton_agent.diff_skills_update.workflow import run_diff_skills_update
from triton_agent.models import AgentResult


class DiffSkillsUpdateWorkflowTests(unittest.TestCase):
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
                if "updating Triton Ascend NPU optimization knowledge" in prompt:
                    self.assertEqual(kwargs["output_label"], "[op] [diff-skills]")
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
                elif "You are simulating an optimizer worker" in prompt:
                    self.assertIn("[op] [simulate-iter-", str(kwargs["output_label"]))
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
                elif "You are auditing a simulated optimization result" in prompt:
                    self.assertIn("[op] [analyze-iter-", str(kwargs["output_label"]))
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

            config = DiffSkillsUpdateConfig(
                input_root=root,
                skills_dir=skills_dir,
                update_skills_dir=root / "update_skills",
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
                    "triton_agent.diff_skills_update.workflow.ensure_skills_workspace",
                    return_value=knowledge_dir,
                ),
                patch("triton_agent.diff_skills_update.workflow.regenerate_pattern_index"),
                patch("triton_agent.diff_skills_update.workflow.promote_converged_knowledge_workspace") as promote,
            ):
                promote.return_value = knowledge_dir
                results = run_diff_skills_update(config, agent_runner=agent_runner)

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "aligned")
            self.assertEqual(len(results[0].iterations), 2)
            self.assertEqual(results[0].matched_patterns, ["tiling"])
            self.assertEqual(results[0].updated_patterns, ["tiling", "loop-invariant-hoisting"])
            self.assertTrue((op_dir / "simulate" / "foo.py").exists())
            self.assertTrue((op_dir / "simulate" / "generated_foo.py").exists())
            report = json.loads((op_dir / "simulate" / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "aligned")
            self.assertEqual(report["updated_patterns"], ["tiling", "loop-invariant-hoisting"])
            self.assertEqual(report["iterations"][0]["updated_patterns"], ["loop-invariant-hoisting"])
            simulate_calls = [call for call in calls if call.get("skills_root") == skills_dir]
            self.assertEqual(len(simulate_calls), 2)
            promote.assert_called_once_with(knowledge_dir)


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
