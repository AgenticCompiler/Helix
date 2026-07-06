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
from triton_agent.models import CommandKind
from triton_agent.skills.selection import resolve_staged_skills

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

    def test_plugin_builder_api_does_not_expose_optimize_target(self) -> None:
        asset_parameters = inspect.signature(build_claude_optimize_plugin_assets).parameters
        builder_parameters = inspect.signature(build_claude_optimize_plugin).parameters

        self.assertNotIn("optimize_target", asset_parameters)
        self.assertNotIn("optimize_target", builder_parameters)

    def test_plugin_builder_renders_optimize_and_convert_agents_without_standalone_prompt_files(self) -> None:
        assets = build_claude_optimize_plugin_assets()

        self.assertIn("agents/triton-agent-optimize.md", assets.text_files)
        self.assertIn("agents/triton-agent-convert.md", assets.text_files)
        self.assertNotIn("CLAUDE.md", assets.text_files)
        self.assertNotIn("prompts.md", assets.text_files)
        agent_text = assets.text_files["agents/triton-agent-optimize.md"]
        convert_agent_text = assets.text_files["agents/triton-agent-convert.md"]
        readme_text = assets.text_files["README.md"]
        self.assertIn("name: triton-agent-optimize", agent_text)
        self.assertIn("Use `triton-npu-optimize` as the primary workflow skill.", agent_text)
        self.assertIn("## Fixed Optimize Modes", agent_text)
        self.assertIn("test-mode: `differential`", agent_text)
        self.assertIn("bench-mode: `torch-npu-profiler`", agent_text)
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
        self.assertIn(
            "Read files cautiously. Do not read unrelated files speculatively or just in case.",
            agent_text,
        )
        self.assertIn("Use the staged workspace skills as the workflow source of truth.", agent_text)
        self.assertIn("Invocation-specific behavior comes from the user prompt, SessionStart context, workflow state, and existing round artifacts.", agent_text)
        self.assertIn("Treat `baseline/` as the canonical optimize baseline.", agent_text)
        self.assertIn("Use `compare-perf` as the authoritative source for round performance summaries.", agent_text)
        self.assertIn("Choose the analysis level for each round before editing code.", agent_text)
        self.assertIn(
            "Escalate analysis in this order: pattern triage, profiling diagnosis, IR attribution, compiler-source escalation.",
            agent_text,
        )
        self.assertIn("Do not begin with blind tiling or launch-parameter search.", agent_text)
        self.assertIn("a5-force-simt-only-discrete-access", agent_text)
        self.assertIn("autotune", agent_text)
        self.assertIn("grid-flatten-and-ub-buffering", agent_text)
        self.assertNotIn("torch-npu-optimize-knowledge", agent_text)
        self.assertNotIn("triton-npu-optimize-submit-baseline", agent_text)
        self.assertNotIn("triton-npu-optimize-start-round", agent_text)
        self.assertNotIn("triton-npu-optimize-submit-round", agent_text)
        self.assertNotIn("## Embedded Optimize Guidance", agent_text)
        self.assertNotIn("## Embedded Optimize Prompt Rules", agent_text)
        self.assertNotIn("Complete optimize rounds strictly one at a time in sequence.", agent_text)
        self.assertNotIn("This workspace is under an optimize round loop.", agent_text)
        self.assertIn("name: triton-agent-convert", convert_agent_text)
        self.assertIn(
            "Use `triton-npu-convert-pytorch-operator` as the primary workflow skill.",
            convert_agent_text,
        )
        self.assertIn("Treat the original input operator file as immutable source material.", convert_agent_text)
        self.assertIn("Use `ascend-npu-gen-test` when no suitable reusable test exists.", convert_agent_text)
        self.assertIn("Use `ascend-npu-run-eval` to execute validation.", convert_agent_text)
        self.assertNotIn("submit-round", convert_agent_text)
        self.assertNotIn("Fixed Optimize Modes", convert_agent_text)
        self.assertNotIn("High-priority generic pattern reminders", convert_agent_text)
        self.assertIn("optimize workflow", readme_text)
        self.assertIn("convert workflow", readme_text)
        self.assertNotIn("optimize hook flow", readme_text)

    def test_build_plugin_writes_expected_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "triton-optimizer"

            built_dir = build_claude_optimize_plugin(output_dir)

            self.assertEqual(built_dir, output_dir.resolve())
            self.assertTrue((built_dir / ".claude-plugin" / "plugin.json").exists())
            self.assertTrue((built_dir / "agents" / "triton-agent-optimize.md").exists())
            self.assertTrue((built_dir / "agents" / "triton-agent-convert.md").exists())
            self.assertTrue((built_dir / "hooks" / "hooks.json").exists())
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

    def test_built_plugin_session_start_bootstraps_baseline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-optimizer")
            workspace = tmpdir / "workspace"
            workspace.mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "session_start.py")],
                input=json.dumps(
                    {
                        "agent_type": "triton-optimizer:triton-agent-optimize",
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
