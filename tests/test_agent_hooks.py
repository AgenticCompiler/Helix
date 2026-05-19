import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triton_agent.agent_hooks import AgentHookManager


class AgentHookManagerTests(unittest.TestCase):
    def test_prepare_codex_hooks_stages_workspace_policy_and_cleans_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            templates_root = Path(__file__).resolve().parents[1] / "hooks"
            manager = AgentHookManager(templates_root)
            self.assertFalse((templates_root / "codex" / "policy.json").exists())

            state = manager.prepare_hooks("codex", workspace)

            hooks_json = workspace / ".codex" / "hooks.json"
            hook_dir = workspace / ".codex" / "triton-agent-hooks"
            policy_json = hook_dir / "policy.json"
            guard_script = hook_dir / "pretooluse_guard.py"
            self.assertTrue(hooks_json.exists())
            self.assertTrue(policy_json.exists())
            self.assertTrue(guard_script.exists())
            self.assertEqual(state.created_paths, [hooks_json, hook_dir])

            hooks_config = json.loads(hooks_json.read_text(encoding="utf-8"))
            self.assertEqual(
                hooks_config["hooks"]["PreToolUse"],
                [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python3 .codex/triton-agent-hooks/pretooluse_guard.py --policy .codex/triton-agent-hooks/policy.json",
                            }
                        ],
                    },
                    {
                        "matcher": "Read",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "python3 .codex/triton-agent-hooks/pretooluse_guard.py --policy .codex/triton-agent-hooks/policy.json",
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
                    str(workspace.resolve() / "triton-agent-logs" / "**"),
                    str(workspace.resolve() / ".codex" / "skills" / "*" / "scripts" / "**"),
                ],
            )
            self.assertIn("triton-agent-logs", policy["deny_message"])
            self.assertIn("triton-agent workspace policy", policy["deny_message"])

            warnings = manager.cleanup(state)

            self.assertEqual(warnings, [])
            self.assertFalse(hooks_json.exists())
            self.assertFalse(hook_dir.exists())
            self.assertTrue((workspace / ".codex").exists())

    def test_prepare_opencode_hooks_stages_workspace_policy_and_cleans_owned_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            templates_root = Path(__file__).resolve().parents[1] / "hooks"
            manager = AgentHookManager(templates_root)

            state = manager.prepare_hooks("opencode", workspace)

            plugin_file = workspace / ".opencode" / "plugins" / "triton-agent-hook-guard.js"
            hook_dir = workspace / ".opencode" / "triton-agent-hooks"
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
                    str(workspace.resolve() / "triton-agent-logs" / "**"),
                    str(workspace.resolve() / ".opencode" / "skills" / "*" / "scripts" / "**"),
                ],
            )
            self.assertIn("triton-agent-logs", policy["deny_message"])
            self.assertIn("triton-agent workspace policy", policy["deny_message"])
            self.assertIn(".opencode/skills/*/scripts/", policy["deny_message"])

            warnings = manager.cleanup(state)

            self.assertEqual(warnings, [])
            self.assertFalse(plugin_file.exists())
            self.assertFalse(hook_dir.exists())
            self.assertTrue((workspace / ".opencode").exists())
            self.assertTrue((workspace / ".opencode" / "plugins").exists())

    def test_prepare_codex_hooks_rejects_existing_hooks_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hooks_json = workspace / ".codex" / "hooks.json"
            hooks_json.parent.mkdir()
            hooks_json.write_text('{"hooks": {}}\n', encoding="utf-8")

            manager = AgentHookManager(Path(__file__).resolve().parents[1] / "hooks")

            with self.assertRaisesRegex(RuntimeError, "Existing Codex hooks config"):
                manager.prepare_hooks("codex", workspace)

    def test_prepare_codex_hooks_rejects_existing_owned_hook_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hook_dir = workspace / ".codex" / "triton-agent-hooks"
            hook_dir.mkdir(parents=True)

            manager = AgentHookManager(Path(__file__).resolve().parents[1] / "hooks")

            with self.assertRaisesRegex(RuntimeError, "Existing Codex hook directory"):
                manager.prepare_hooks("codex", workspace)

    def test_prepare_opencode_hooks_rejects_existing_plugin_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            plugin_file = workspace / ".opencode" / "plugins" / "triton-agent-hook-guard.js"
            plugin_file.parent.mkdir(parents=True)
            plugin_file.write_text("export default async function Plugin() {}\n", encoding="utf-8")

            manager = AgentHookManager(Path(__file__).resolve().parents[1] / "hooks")

            with self.assertRaisesRegex(RuntimeError, "Existing OpenCode hook plugin"):
                manager.prepare_hooks("opencode", workspace)

    def test_prepare_opencode_hooks_rejects_existing_owned_hook_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            hook_dir = workspace / ".opencode" / "triton-agent-hooks"
            hook_dir.mkdir(parents=True)

            manager = AgentHookManager(Path(__file__).resolve().parents[1] / "hooks")

            with self.assertRaisesRegex(RuntimeError, "Existing OpenCode hook directory"):
                manager.prepare_hooks("opencode", workspace)

    def test_prepare_hooks_for_non_codex_backend_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            manager = AgentHookManager(Path(__file__).resolve().parents[1] / "hooks")

            state = manager.prepare_hooks("claude", workspace)

            self.assertEqual(state.created_paths, [])
            self.assertFalse((workspace / ".codex").exists())
            self.assertFalse((workspace / ".opencode").exists())
            self.assertEqual(manager.cleanup(state), [])


if __name__ == "__main__":
    unittest.main()
