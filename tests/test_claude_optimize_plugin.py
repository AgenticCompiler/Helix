import inspect
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helix.models import CommandKind
from helix.skills.selection import resolve_staged_skills

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
    def test_plugin_builder_uses_optimize_and_triton_convert_skill_staging_contract(self) -> None:
        optimize_skill_names, optimize_skill_sources = resolve_staged_skills(CommandKind.OPTIMIZE)
        convert_skill_names, convert_skill_sources = resolve_staged_skills(
            CommandKind.CONVERT,
            language="triton",
        )
        self.assertIsNotNone(optimize_skill_names)
        self.assertIsNotNone(convert_skill_names)

        assets = build_claude_optimize_plugin_assets()
        expected_optimize_skill_names = tuple(
            skill_name
            for skill_name in (optimize_skill_names or ())
            if skill_name != "torch-npu-optimize-knowledge"
        )

        self.assertEqual(
            tuple(sorted(assets.optimize_skill_names)),
            tuple(sorted(expected_optimize_skill_names)),
        )
        self.assertEqual(
            tuple(sorted(assets.convert_skill_names)),
            tuple(sorted(convert_skill_names or ())),
        )
        expected_skill_names = tuple(
            dict.fromkeys(expected_optimize_skill_names + (convert_skill_names or ()))
        )
        self.assertEqual(
            tuple(sorted(assets.skill_names)),
            tuple(sorted(expected_skill_names)),
        )
        expected_skill_sources = {}
        if optimize_skill_sources:
            expected_skill_sources.update(optimize_skill_sources)
        if convert_skill_sources:
            expected_skill_sources.update(convert_skill_sources)
        self.assertEqual(assets.skill_sources, expected_skill_sources or None)
        self.assertNotIn("torch-npu-optimize-knowledge", assets.optimize_skill_names)
        self.assertNotIn("torch-npu-optimize-knowledge", assets.skill_names)

    def test_plugin_builder_api_does_not_expose_optimize_target_or_subagent_option(self) -> None:
        asset_parameters = inspect.signature(build_claude_optimize_plugin_assets).parameters
        builder_parameters = inspect.signature(build_claude_optimize_plugin).parameters

        self.assertNotIn("optimize_target", asset_parameters)
        self.assertNotIn("optimize_target", builder_parameters)
        self.assertNotIn("enable_subagent", asset_parameters)
        self.assertNotIn("enable_subagent", builder_parameters)

    def test_plugin_builder_omits_agent_definitions(self) -> None:
        assets = build_claude_optimize_plugin_assets()

        self.assertEqual(set(assets.text_files), {"README.md"})
        self.assertNotIn("agents/helix-optimizer.md", assets.text_files)
        self.assertNotIn("agents/helix-convert.md", assets.text_files)
        self.assertNotIn("CLAUDE.md", assets.text_files)
        self.assertNotIn("prompts.md", assets.text_files)
        readme_text = assets.text_files["README.md"]
        self.assertIn("optimize workflow", readme_text)
        self.assertIn("convert workflow", readme_text)
        self.assertIn("bundled skills available directly", readme_text)
        self.assertNotIn("--agent", readme_text)
        self.assertNotIn("helix-optimizer", readme_text)

    def test_build_plugin_writes_expected_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "triton-optimizer"

            built_dir = build_claude_optimize_plugin(output_dir)

            self.assertEqual(built_dir, output_dir.resolve())
            self.assertTrue((built_dir / ".claude-plugin" / "plugin.json").exists())
            self.assertFalse((built_dir / "agents").exists())
            self.assertTrue((built_dir / "hooks" / "hooks.json").exists())
            self.assertFalse((built_dir / "hooks" / "subagent_start.py").exists())
            self.assertFalse((built_dir / "hooks" / "subagent_stop.py").exists())
            self.assertTrue((built_dir / "skills").is_dir())
            self.assertFalse((built_dir / "CLAUDE.md").exists())
            self.assertFalse((built_dir / "prompts.md").exists())

            plugin_manifest = json.loads(
                (built_dir / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plugin_manifest["name"], "triton-optimizer")
            self.assertEqual(plugin_manifest["version"], "0.1.0")
            self.assertTrue((built_dir / "hooks" / "state_bootstrap.py").exists())
            self.assertTrue((built_dir / "hooks" / "session_start.py").exists())
            self.assertTrue(
                (built_dir / "hooks" / "hook_runtime" / "__init__.py").exists()
            )
            self.assertTrue(
                (built_dir / "hooks" / "hook_runtime" / "optimize" / "workflow_state.py").exists()
            )
            self.assertTrue(
                (built_dir / "hooks" / "hook_runtime" / "optimize" / "compiler_source.py").exists()
            )
            self.assertFalse((built_dir / "python_support").exists())
            self.assertTrue((built_dir / "skills" / "ascend-npu-optimize-state").is_dir())
            self.assertTrue((built_dir / "skills" / "triton-npu-convert-pytorch-operator").is_dir())
            self.assertFalse((built_dir / "skills" / "torch-npu-optimize-knowledge").exists())
            self.assertFalse((built_dir / "skills" / "tilelang-npu-convert-pytorch-operator").exists())

            hooks_manifest = json.loads((built_dir / "hooks" / "hooks.json").read_text(encoding="utf-8"))
            self.assertIn("SessionStart", hooks_manifest["hooks"])
            self.assertIn("SessionEnd", hooks_manifest["hooks"])
            self.assertNotIn("SubagentStart", hooks_manifest["hooks"])
            self.assertNotIn("SubagentStop", hooks_manifest["hooks"])

    def test_build_plugin_copies_latest_hook_runtime_tool_use_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "triton-optimizer"

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
            output_dir = Path(tmp) / "triton-optimizer"

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

    def test_built_plugin_session_start_bootstraps_baseline_state_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-optimizer")
            workspace = tmpdir / "workspace"
            workspace.mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "session_start.py")],
                input=json.dumps(
                    {
                        "cwd": str(workspace),
                    }
                ),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state_payload = json.loads(
                (workspace / ".helix" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(state_payload["baseline"], {"status": "pending", "submitted_at": None})

    def test_built_plugin_session_end_removes_runtime_dir_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-optimizer")
            workspace = tmpdir / "workspace"
            workspace.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("{}", encoding="utf-8")
            (workspace / "baseline").mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "session_end.py")],
                input=json.dumps(
                    {
                        "cwd": str(workspace),
                    }
                ),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((workspace / ".helix").exists())
            self.assertTrue((workspace / "baseline").exists())

if __name__ == "__main__":
    unittest.main()
