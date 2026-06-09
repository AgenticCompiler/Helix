import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.optimize.subagents import perf_diagnosis_subagent_definition
from triton_agent.subagents import SubagentManager


class SubagentManagerTests(unittest.TestCase):
    def test_prepare_codex_subagent_stages_toml_and_cleans_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            manager = SubagentManager()
            definition = perf_diagnosis_subagent_definition(
                optimize_target="kernel",
                enable_cann_ext_api=False,
            )

            state = manager.prepare("codex", workspace, (definition,))

            agent_path = (
                workspace
                / ".codex"
                / "agents"
                / "triton-agent-perf-diagnosis-advisor.toml"
            )
            self.assertTrue(agent_path.exists())
            content = agent_path.read_text(encoding="utf-8")
            self.assertIn("triton-agent-perf-diagnosis-advisor", content)
            self.assertIn("Start with skill `triton-npu-optimize-knowledge`", content)
            self.assertIn("Read its `pattern_index.md` before detailed pattern cards.", content)
            self.assertIn(
                "Use its `symptom_index.md` when profile or IR evidence needs symptom routing.",
                content,
            )
            self.assertNotIn(".codex/skills/triton-npu-optimize-knowledge/", content)
            self.assertIn("must not perform optimization work", content)

            self.assertEqual(manager.cleanup(state), [])
            self.assertFalse(agent_path.exists())
            self.assertFalse((workspace / ".codex").exists())

    def test_prepare_claude_subagent_stages_markdown_and_preserves_existing_parent_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            agent_dir = workspace / ".claude" / "agents"
            agent_dir.mkdir(parents=True)
            manager = SubagentManager()
            definition = perf_diagnosis_subagent_definition(
                optimize_target="operator",
                enable_cann_ext_api=True,
            )

            state = manager.prepare("claude", workspace, (definition,))

            agent_path = agent_dir / "triton-agent-perf-diagnosis-advisor.md"
            self.assertTrue(agent_path.exists())
            content = agent_path.read_text(encoding="utf-8")
            self.assertIn("Start with skill `triton-npu-optimize-knowledge`", content)
            self.assertIn(
                "Also use skill `torch-npu-optimize-knowledge` and its `pattern_index.md`",
                content,
            )
            self.assertIn(
                "Also use skill `triton-npu-cann-ext-api-patterns` and its `index.md`",
                content,
            )
            self.assertNotIn(".claude/skills/", content)
            self.assertIn("collect fresh benchmark, profiler, or IR evidence", content)
            self.assertIn("must not perform optimization work", content)

            self.assertEqual(manager.cleanup(state), [])
            self.assertTrue(agent_dir.exists())
            self.assertFalse(agent_path.exists())

    def test_prepare_opencode_subagent_stages_markdown_with_subagent_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            manager = SubagentManager()
            definition = perf_diagnosis_subagent_definition(
                optimize_target="kernel",
                enable_cann_ext_api=False,
            )

            state = manager.prepare("opencode", workspace, (definition,))

            agent_path = (
                workspace
                / ".opencode"
                / "agents"
                / "triton-agent-perf-diagnosis-advisor.md"
            )
            self.assertTrue(agent_path.exists())
            content = agent_path.read_text(encoding="utf-8")
            self.assertIn("mode: subagent", content)
            self.assertIn("Start with skill `triton-npu-optimize-knowledge`", content)
            self.assertIn(
                "Use skill `triton-npu-run-eval`, `triton-npu-profile-operator`, and `triton-npu-analyze-ir`",
                content,
            )
            self.assertIn("must not perform optimization work", content)
            self.assertEqual(manager.cleanup(state), [])
            self.assertFalse((workspace / ".opencode").exists())

    def test_prepare_rejects_existing_subagent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            agent_dir = workspace / ".codex" / "agents"
            agent_dir.mkdir(parents=True)
            (agent_dir / "triton-agent-perf-diagnosis-advisor.toml").write_text(
                "user owned\n",
                encoding="utf-8",
            )
            manager = SubagentManager()
            definition = perf_diagnosis_subagent_definition(
                optimize_target="kernel",
                enable_cann_ext_api=False,
            )

            with self.assertRaisesRegex(RuntimeError, "Existing subagent file must not be overwritten"):
                manager.prepare("codex", workspace, (definition,))


if __name__ == "__main__":
    unittest.main()
