import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from helix.backends.claude_hooks import prepare_claude_hooks
from helix.backends.codex_hooks import prepare_codex_hooks
from helix.backends.hook_common import cleanup_hook_stage
from helix.backends.opencode_hooks import prepare_opencode_hooks


class AgentHookStageTests(unittest.TestCase):
    def test_prepare_codex_hooks_includes_extra_allowed_read_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            compiler_source = Path(tmp) / "compiler-sources" / "AscendNPU-IR"
            workspace.mkdir()
            compiler_source.mkdir(parents=True)

            state = prepare_codex_hooks(
                Path(__file__).resolve().parents[1] / "hooks",
                workspace,
                extra_allowed_read_roots=(compiler_source,),
            )

            policy_path = workspace / ".codex" / "helix-hooks" / "policy.json"
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

            self.assertEqual(
                policy["allow_read_roots"],
                [str(workspace.resolve()), str(compiler_source.resolve())],
            )

            self.assertEqual(cleanup_hook_stage(state), [])

    def test_prepare_codex_hooks_stages_workspace_policy_and_cleans_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            templates_root = Path(__file__).resolve().parents[1] / "hooks"
            self.assertFalse((templates_root / "codex" / "policy.json").exists())
            self.assertTrue((templates_root / "codex" / "pretooluse_guard.py").exists())
            self.assertIn(
                "${CODEX_PROJECT_DIR}",
                (templates_root / "codex" / "hooks.json").read_text(encoding="utf-8"),
            )
            self.assertTrue((templates_root / "claude" / "pretooluse_guard.py").exists())
            self.assertTrue((templates_root / "claude" / "settings.json").exists())
            self.assertTrue((Path(__file__).resolve().parents[1] / "src" / "hook_runtime").is_dir())

            state = prepare_codex_hooks(templates_root, workspace)

            hooks_json = workspace / ".codex" / "hooks.json"
            hook_dir = workspace / ".codex" / "helix-hooks"
            policy_json = hook_dir / "policy.json"
            resolved_workspace = workspace.resolve()
            resolved_hook_dir = resolved_workspace / ".codex" / "helix-hooks"
            resolved_policy_json = resolved_hook_dir / "policy.json"
            guard_script = hook_dir / "pretooluse_guard.py"
            hook_runtime_dir = hook_dir / "hook_runtime"
            pretooluse_adapter_module = hook_runtime_dir / "pretooluse_adapter.py"
            tool_use_decision_module = hook_runtime_dir / "tool_use_decision.py"
            trace_script = hook_dir / "tool_trace_hook.py"
            self.assertTrue(hooks_json.exists())
            self.assertTrue(policy_json.exists())
            self.assertTrue(guard_script.exists())
            self.assertTrue(hook_runtime_dir.exists())
            self.assertTrue(pretooluse_adapter_module.exists())
            self.assertTrue(tool_use_decision_module.exists())
            self.assertTrue(trace_script.exists())
            self.assertEqual(state.created_paths, [hooks_json, hook_dir])

            hooks_config = json.loads(hooks_json.read_text(encoding="utf-8"))
            self.assertEqual(
                hooks_config["hooks"]["PreToolUse"],
                [
                    {
                        "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"python \"{resolved_hook_dir / 'tool_trace_hook.py'}\" "
                                    f"--policy \"{resolved_policy_json}\" --event PreToolUse"
                                ),
                            },
                            {
                                "type": "command",
                                "command": (
                                    f"python3 \"{resolved_hook_dir / 'pretooluse_guard.py'}\" "
                                    f"--policy \"{resolved_policy_json}\""
                                ),
                            }
                        ],
                    },
                ],
            )

            policy = json.loads(policy_json.read_text(encoding="utf-8"))
            self.assertEqual(policy["workspace_root"], str(workspace.resolve()))
            self.assertEqual(policy["allow_read_roots"], [str(workspace.resolve())])
            self.assertEqual(
                policy["deny_read_globs"],
                [
                    str(workspace.resolve() / ".helix"),
                    str(workspace.resolve() / ".helix" / "**"),
                    str(workspace.resolve() / "helix-logs" / "**"),
                    str(workspace.resolve() / ".codex" / "helix-hooks"),
                    str(workspace.resolve() / ".codex" / "helix-hooks" / "**"),
                    str(workspace.resolve() / ".codex" / "skills" / "*" / "scripts" / "**"),
                ],
            )
            self.assertIn("temporary optimize runtime files", policy["deny_message"])
            self.assertIn("helix-logs", policy["deny_message"])
            self.assertIn("helix workspace policy", policy["deny_message"])

            warnings = cleanup_hook_stage(state)

            self.assertEqual(warnings, [])
            self.assertFalse(hooks_json.exists())
            self.assertFalse(hook_dir.exists())
            self.assertTrue((workspace / ".codex").exists())

    def test_prepare_codex_hooks_quotes_absolute_command_paths_when_workspace_contains_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace with spaces"
            workspace.mkdir()

            state = prepare_codex_hooks(Path(__file__).resolve().parents[1] / "hooks", workspace)

            hooks_json = workspace / ".codex" / "hooks.json"
            resolved_workspace = workspace.resolve()
            resolved_hook_dir = resolved_workspace / ".codex" / "helix-hooks"
            resolved_policy_json = resolved_hook_dir / "policy.json"
            hooks_config = json.loads(hooks_json.read_text(encoding="utf-8"))

            self.assertEqual(
                hooks_config["hooks"]["PreToolUse"],
                [
                    {
                        "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"python \"{resolved_hook_dir / 'tool_trace_hook.py'}\" "
                                    f"--policy \"{resolved_policy_json}\" --event PreToolUse"
                                ),
                            },
                            {
                                "type": "command",
                                "command": (
                                    f"python3 \"{resolved_hook_dir / 'pretooluse_guard.py'}\" "
                                    f"--policy \"{resolved_policy_json}\""
                                ),
                            },
                        ],
                    },
                ],
            )

            self.assertEqual(cleanup_hook_stage(state), [])

    def test_prepare_opencode_hooks_stages_workspace_policy_and_cleans_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            templates_root = Path(__file__).resolve().parents[1] / "hooks"

            state = prepare_opencode_hooks(templates_root, workspace)

            plugin_file = workspace / ".opencode" / "plugins" / "helix-hook-guard.js"
            hook_dir = workspace / ".opencode" / "helix-hooks"
            policy_json = hook_dir / "policy.json"
            self.assertTrue(plugin_file.exists())
            self.assertTrue(policy_json.exists())
            self.assertEqual(state.created_paths, [plugin_file, hook_dir])

            policy = json.loads(policy_json.read_text(encoding="utf-8"))
            self.assertEqual(policy["workspace_root"], str(workspace.resolve()))
            self.assertEqual(policy["allow_read_roots"], [str(workspace.resolve())])
            self.assertEqual(
                policy["deny_read_globs"],
                [
                    str(workspace.resolve() / ".helix"),
                    str(workspace.resolve() / ".helix" / "**"),
                    str(workspace.resolve() / "helix-logs" / "**"),
                    str(workspace.resolve() / ".opencode" / "plugins" / "helix-hook-guard.js"),
                    str(workspace.resolve() / ".opencode" / "helix-hooks"),
                    str(workspace.resolve() / ".opencode" / "helix-hooks" / "**"),
                    str(workspace.resolve() / ".opencode" / "skills" / "*" / "scripts" / "**"),
                ],
            )
            self.assertIn("temporary optimize runtime files", policy["deny_message"])
            self.assertIn("helix-logs", policy["deny_message"])
            self.assertIn("helix workspace policy", policy["deny_message"])
            self.assertIn(".opencode/skills/*/scripts/", policy["deny_message"])

            warnings = cleanup_hook_stage(state)

            self.assertEqual(warnings, [])
            self.assertFalse(plugin_file.exists())
            self.assertFalse(hook_dir.exists())
            self.assertTrue((workspace / ".opencode").exists())
            self.assertTrue((workspace / ".opencode" / "plugins").exists())

    def test_prepare_opencode_hooks_includes_extra_allowed_read_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            compiler_source = Path(tmp) / "compiler-sources" / "AscendNPU-IR"
            workspace.mkdir()
            compiler_source.mkdir(parents=True)

            state = prepare_opencode_hooks(
                Path(__file__).resolve().parents[1] / "hooks",
                workspace,
                extra_allowed_read_roots=(compiler_source,),
            )

            policy_path = workspace / ".opencode" / "helix-hooks" / "policy.json"
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

            self.assertEqual(
                policy["allow_read_roots"],
                [str(workspace.resolve()), str(compiler_source.resolve())],
            )

            self.assertEqual(cleanup_hook_stage(state), [])

    def test_prepare_claude_hooks_stages_workspace_settings_and_cleans_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            templates_root = Path(__file__).resolve().parents[1] / "hooks"
            self.assertIn(
                "${CLAUDE_PROJECT_DIR}",
                (templates_root / "claude" / "settings.json").read_text(encoding="utf-8"),
            )

            state = prepare_claude_hooks(templates_root, workspace)

            hook_dir = workspace / ".claude" / "helix-hooks"
            settings_json = hook_dir / "settings.json"
            policy_json = hook_dir / "policy.json"
            resolved_workspace = workspace.resolve()
            resolved_hook_dir = resolved_workspace / ".claude" / "helix-hooks"
            resolved_policy_json = resolved_hook_dir / "policy.json"
            guard_script = hook_dir / "pretooluse_guard.py"
            hook_runtime_dir = hook_dir / "hook_runtime"
            pretooluse_adapter_module = hook_runtime_dir / "pretooluse_adapter.py"
            tool_use_decision_module = hook_runtime_dir / "tool_use_decision.py"
            self.assertTrue(settings_json.exists())
            self.assertTrue(policy_json.exists())
            self.assertTrue(guard_script.exists())
            self.assertTrue(hook_runtime_dir.exists())
            self.assertTrue(pretooluse_adapter_module.exists())
            self.assertTrue(tool_use_decision_module.exists())
            self.assertEqual(state.created_paths, [settings_json, hook_dir])

            settings = json.loads(settings_json.read_text(encoding="utf-8"))
            self.assertEqual(
                settings["hooks"]["PreToolUse"],
                [
                    {
                        "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python3",
                                "args": [
                                    str(resolved_hook_dir / "pretooluse_guard.py"),
                                    "--policy",
                                    str(resolved_policy_json),
                                ],
                            }
                        ],
                    }
                ],
            )

            policy = json.loads(policy_json.read_text(encoding="utf-8"))
            self.assertEqual(policy["workspace_root"], str(workspace.resolve()))
            self.assertEqual(policy["allow_read_roots"], [str(workspace.resolve())])
            self.assertEqual(
                policy["deny_read_globs"],
                [
                    str(workspace.resolve() / ".helix"),
                    str(workspace.resolve() / ".helix" / "**"),
                    str(workspace.resolve() / "helix-logs" / "**"),
                    str(workspace.resolve() / ".claude" / "helix-hooks"),
                    str(workspace.resolve() / ".claude" / "helix-hooks" / "**"),
                    str(workspace.resolve() / ".claude" / "skills" / "*" / "scripts" / "**"),
                ],
            )
            self.assertIn("temporary optimize runtime files", policy["deny_message"])
            self.assertIn(".claude/skills/*/scripts/", policy["deny_message"])

            warnings = cleanup_hook_stage(state)

            self.assertEqual(warnings, [])
            self.assertFalse(hook_dir.exists())
            self.assertTrue((workspace / ".claude").exists())

    def test_prepare_claude_hooks_includes_extra_allowed_read_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            compiler_source = Path(tmp) / "compiler-sources" / "AscendNPU-IR"
            workspace.mkdir()
            compiler_source.mkdir(parents=True)

            state = prepare_claude_hooks(
                Path(__file__).resolve().parents[1] / "hooks",
                workspace,
                extra_allowed_read_roots=(compiler_source,),
            )

            policy_path = workspace / ".claude" / "helix-hooks" / "policy.json"
            policy = json.loads(policy_path.read_text(encoding="utf-8"))

            self.assertEqual(
                policy["allow_read_roots"],
                [str(workspace.resolve()), str(compiler_source.resolve())],
            )

            self.assertEqual(cleanup_hook_stage(state), [])

    def test_prepare_codex_hooks_rejects_existing_hooks_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hooks_json = workspace / ".codex" / "hooks.json"
            hooks_json.parent.mkdir()
            hooks_json.write_text('{"hooks": {}}\n', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Existing Codex hooks config"):
                prepare_codex_hooks(Path(__file__).resolve().parents[1] / "hooks", workspace)

    def test_prepare_codex_hooks_rejects_existing_owned_hook_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hook_dir = workspace / ".codex" / "helix-hooks"
            hook_dir.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "Existing Codex hook directory"):
                prepare_codex_hooks(Path(__file__).resolve().parents[1] / "hooks", workspace)

    def test_prepare_opencode_hooks_rejects_existing_plugin_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            plugin_file = workspace / ".opencode" / "plugins" / "helix-hook-guard.js"
            plugin_file.parent.mkdir(parents=True)
            plugin_file.write_text("export default async function Plugin() {}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Existing OpenCode hook plugin"):
                prepare_opencode_hooks(Path(__file__).resolve().parents[1] / "hooks", workspace)

    def test_prepare_opencode_hooks_rejects_existing_owned_hook_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hook_dir = workspace / ".opencode" / "helix-hooks"
            hook_dir.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "Existing OpenCode hook directory"):
                prepare_opencode_hooks(Path(__file__).resolve().parents[1] / "hooks", workspace)

    def test_prepare_claude_hooks_rejects_existing_owned_hook_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hook_dir = workspace / ".claude" / "helix-hooks"
            hook_dir.mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "Existing Claude hook directory"):
                prepare_claude_hooks(Path(__file__).resolve().parents[1] / "hooks", workspace)


if __name__ == "__main__":
    unittest.main()
