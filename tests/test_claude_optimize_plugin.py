import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from triton_agent.models import CommandKind
from triton_agent.skill_staging import resolve_staged_skills

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build-claude-optimize-plugin.py"


def _load_builder_script_module() -> ModuleType:
    module_name = "build_claude_optimize_plugin_script"
    spec = importlib.util.spec_from_file_location(
        module_name,
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load build-claude-optimize-plugin.py for tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BUILDER_SCRIPT = _load_builder_script_module()
build_claude_optimize_plugin = getattr(_BUILDER_SCRIPT, "build_claude_optimize_plugin")
build_claude_optimize_plugin_assets = getattr(
    _BUILDER_SCRIPT,
    "build_claude_optimize_plugin_assets",
)


class ClaudeOptimizePluginBuilderTests(unittest.TestCase):
    def test_plugin_builder_uses_optimize_skill_staging_contract(self) -> None:
        skill_names, skill_sources = resolve_staged_skills(CommandKind.OPTIMIZE)
        self.assertIsNotNone(skill_names)

        assets = build_claude_optimize_plugin_assets()

        self.assertEqual(
            tuple(sorted(assets.skill_names)),
            tuple(sorted(skill_names or ())),
        )
        self.assertEqual(assets.skill_sources, skill_sources)

    def test_plugin_builder_renders_single_optimize_agent_without_standalone_prompt_files(self) -> None:
        assets = build_claude_optimize_plugin_assets()

        self.assertIn("agents/triton-agent-optimize.md", assets.text_files)
        self.assertNotIn("CLAUDE.md", assets.text_files)
        self.assertNotIn("prompts.md", assets.text_files)
        agent_text = assets.text_files["agents/triton-agent-optimize.md"]
        readme_text = assets.text_files["README.md"]
        self.assertIn("name: triton-agent-optimize", agent_text)
        self.assertIn("Use `triton-npu-optimize` as the primary workflow skill.", agent_text)
        self.assertIn("## Critical Workflow Rules", agent_text)
        self.assertIn(
            "Use `ascend-npu-optimize-state` `start-round` immediately before beginning a new `opt-round-N/`.",
            agent_text,
        )
        self.assertIn(
            "Use `ascend-npu-optimize-state` `submit-round` after each complete round before stopping or opening the next round.",
            agent_text,
        )
        self.assertIn("Keep exactly one optimize round active at a time.", agent_text)
        self.assertNotIn("## Embedded Optimize Guidance", agent_text)
        self.assertNotIn("## Embedded Optimize Prompt Rules", agent_text)
        self.assertNotIn("Complete optimize rounds strictly one at a time in sequence.", agent_text)
        self.assertNotIn("This workspace is under an optimize round loop.", agent_text)
        self.assertIn("optimize workflow", readme_text)
        self.assertNotIn("optimize hook flow", readme_text)

    def test_build_plugin_writes_expected_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "triton-agent-optimize"

            built_dir = build_claude_optimize_plugin(output_dir)

            self.assertEqual(built_dir, output_dir.resolve())
            self.assertTrue((built_dir / ".claude-plugin" / "plugin.json").exists())
            self.assertTrue((built_dir / "agents" / "triton-agent-optimize.md").exists())
            self.assertTrue((built_dir / "hooks" / "hooks.json").exists())
            self.assertTrue((built_dir / "skills").is_dir())
            self.assertFalse((built_dir / "CLAUDE.md").exists())
            self.assertFalse((built_dir / "prompts.md").exists())

            plugin_manifest = json.loads(
                (built_dir / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plugin_manifest["name"], "triton-agent-optimize")
            self.assertEqual(plugin_manifest["version"], "0.1.0")
            self.assertTrue((built_dir / "hooks" / "state_bootstrap.py").exists())
            self.assertTrue((built_dir / "hooks" / "session_start.py").exists())
            self.assertTrue(
                (built_dir / "hooks" / "hook_runtime" / "__init__.py").exists()
            )
            self.assertTrue(
                (built_dir / "hooks" / "hook_runtime" / "optimize" / "workflow_state.py").exists()
            )
            self.assertFalse((built_dir / "python_support").exists())
            self.assertTrue((built_dir / "skills" / "ascend-npu-optimize-state").is_dir())

    def test_build_plugin_copies_latest_hook_runtime_tool_use_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "triton-agent-optimize"

            built_dir = build_claude_optimize_plugin(output_dir)

            shared_guard = (
                Path(__file__).resolve().parents[1]
                / "src"
                / "hook_runtime"
                / "tool_use_decision.py"
            )
            built_guard = built_dir / "hooks" / "hook_runtime" / "tool_use_decision.py"

            self.assertTrue(built_guard.exists())
            self.assertEqual(
                built_guard.read_text(encoding="utf-8"),
                shared_guard.read_text(encoding="utf-8"),
            )

    def test_build_plugin_copies_hook_runtime_pretooluse_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "triton-agent-optimize"

            built_dir = build_claude_optimize_plugin(output_dir)

            shared_helper = (
                Path(__file__).resolve().parents[1]
                / "src"
                / "hook_runtime"
                / "pretooluse_adapter.py"
            )
            built_helper = built_dir / "hooks" / "hook_runtime" / "pretooluse_adapter.py"

            self.assertTrue(built_helper.exists())
            self.assertEqual(
                built_helper.read_text(encoding="utf-8"),
                shared_helper.read_text(encoding="utf-8"),
            )

    def test_built_plugin_session_start_bootstraps_baseline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-agent-optimize")
            workspace = tmpdir / "workspace"
            workspace.mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "session_start.py")],
                input=json.dumps(
                    {
                        "agent_type": "triton-agent-optimize:triton-agent-optimize",
                        "cwd": str(workspace),
                    }
                ),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state_payload = json.loads(
                (workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(state_payload["baseline"], {"status": "pending", "submitted_at": None})


if __name__ == "__main__":
    unittest.main()
