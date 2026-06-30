from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Callable, Protocol, cast

HOOKS_ROOT = Path(__file__).resolve().parents[1] / "hooks" / "claude_plugin"


class BootstrapResultLike(Protocol):
    additional_context: str | None


def _load_state_bootstrap_module() -> ModuleType:
    module_name = "claude_plugin_state_bootstrap"
    spec = importlib.util.spec_from_file_location(
        module_name,
        HOOKS_ROOT / "state_bootstrap.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Claude plugin state bootstrap module for tests.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_STATE_BOOTSTRAP = _load_state_bootstrap_module()
bootstrap_runtime_state = cast(
    Callable[[Path], BootstrapResultLike],
    getattr(_STATE_BOOTSTRAP, "bootstrap_runtime_state"),
)
validate_existing_state = cast(
    Callable[[Path], BootstrapResultLike],
    getattr(_STATE_BOOTSTRAP, "validate_existing_state"),
)


class ClaudeOptimizePluginHookTests(unittest.TestCase):
    def test_bootstrap_runtime_state_creates_runtime_dir_without_inferred_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            result = bootstrap_runtime_state(workspace)

            self.assertIsNotNone(result.additional_context)
            assert result.additional_context is not None
            self.assertTrue((workspace / ".triton-agent").is_dir())
            self.assertFalse((workspace / ".triton-agent" / "state.json").exists())
            self.assertIn("submit-baseline", result.additional_context)
            self.assertIn("start-round", result.additional_context)

    def test_validate_existing_state_reports_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / ".triton-agent" / "state.json"
            state_path.parent.mkdir()
            state_path.write_text("{", encoding="utf-8")

            result = validate_existing_state(state_path)

            self.assertIsNotNone(result.additional_context)
            assert result.additional_context is not None
            self.assertIn("malformed", result.additional_context)
            self.assertIn("submit-baseline", result.additional_context)
            self.assertNotIn("Remove it", result.additional_context)

    def test_session_start_returns_repair_guidance_for_namespaced_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _run_hook(
                "session_start.py",
                {
                    "agent_type": "triton-agent-optimize:triton-agent-optimize",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            payload = json.loads(result.stdout)
            self.assertTrue((workspace / ".triton-agent").is_dir())
            self.assertFalse((workspace / ".triton-agent" / "state.json").exists())
            self.assertIn("submit-baseline", payload["hookSpecificOutput"]["additionalContext"])

    def test_session_end_removes_runtime_dir_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".triton-agent").mkdir()
            (workspace / ".triton-agent" / "state.json").write_text("{}", encoding="utf-8")
            (workspace / "baseline").mkdir()

            result = _run_hook(
                "session_end.py",
                {
                    "agent_type": "triton-agent-optimize:triton-agent-optimize",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse((workspace / ".triton-agent").exists())
            self.assertTrue((workspace / "baseline").exists())

    def test_pretooluse_guard_denies_edit_when_workflow_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "triton-agent-optimize:triton-agent-optimize",
                    "cwd": str(workspace),
                    "tool_name": "Edit",
                    "tool_input": {
                        "file_path": str(workspace / "kernel.py"),
                    },
                },
            )

            self.assertEqual(result.returncode, 0)
            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["hookSpecificOutput"]["permissionDecision"],
                "deny",
            )
            reason = payload["hookSpecificOutput"]["permissionDecisionReason"]
            self.assertIn("submit-baseline", reason)
            self.assertIn("start-round", reason)
            self.assertNotIn(".triton-agent/state.json", reason)

    def test_pretooluse_guard_falls_back_to_shared_policy_in_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            protected_path = workspace / ".triton-agent" / "state.json"

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "triton-agent-optimize:triton-agent-optimize",
                    "cwd": str(workspace),
                    "tool_name": "Read",
                    "tool_input": {
                        "file_path": str(protected_path),
                    },
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stderr, "")
            payload = json.loads(result.stdout)
            self.assertEqual(
                payload["hookSpecificOutput"]["permissionDecision"],
                "deny",
            )
            self.assertIn(
                "blocked by triton-agent workspace policy",
                payload["hookSpecificOutput"]["permissionDecisionReason"],
            )


def _run_hook(script_name: str, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOKS_ROOT / script_name)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )


if __name__ == "__main__":
    unittest.main()
