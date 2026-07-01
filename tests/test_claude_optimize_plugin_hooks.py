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
    def test_bootstrap_runtime_state_bootstraps_fresh_baseline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = bootstrap_runtime_state(workspace)

            self.assertTrue((workspace / ".triton-agent").is_dir())
            state_path = workspace / ".triton-agent" / "state.json"
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "baseline")
            self.assertEqual(payload["baseline"], {"status": "pending", "submitted_at": None})
            self.assertNotIn("source_operator", payload)
            self.assertIsNone(result.additional_context)

    def test_bootstrap_runtime_state_rebuilds_resumable_workspace_without_operator_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text(
                json.dumps(
                    {
                        "baseline_kind": "original",
                        "source_operator": "../kernel.py",
                        "baseline_operator": "opt_kernel.py",
                        "test_file": "../differential_test_kernel.py",
                        "test_mode": "differential",
                        "bench_file": "../bench_kernel.py",
                        "bench_mode": "torch-npu-profiler",
                        "perf_artifact": "perf.txt",
                        "correctness_status": "passed",
                        "benchmark_status": "passed",
                        "baseline_established": True,
                    }
                ),
                encoding="utf-8",
            )
            (baseline_dir / "perf.txt").write_text("latency-a: 1.0\n", encoding="utf-8")
            (baseline_dir / "opt_kernel.py").write_text("print('baseline')\n", encoding="utf-8")
            (workspace / "kernel.py").write_text("print('source')\n", encoding="utf-8")
            (workspace / "differential_test_kernel.py").write_text(
                "# test-mode: differential\nprint('test')\n",
                encoding="utf-8",
            )
            (workspace / "bench_kernel.py").write_text(
                "# bench-mode: torch-npu-profiler\n# kernel: k\nprint('bench')\n",
                encoding="utf-8",
            )

            result = bootstrap_runtime_state(workspace)

            state_path = workspace / ".triton-agent" / "state.json"
            self.assertTrue(state_path.exists())
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "awaiting_round_start")
            self.assertEqual(payload["baseline"], {"status": "passed", "submitted_at": None})
            self.assertIsNone(payload["current_round"])
            self.assertNotIn("source_operator", payload)
            self.assertIsNone(result.additional_context)

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

    def test_bootstrap_runtime_state_returns_repair_guidance_when_resumable_markers_exist_but_baseline_state_is_invalid(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline_dir = workspace / "baseline"
            baseline_dir.mkdir()
            (baseline_dir / "state.json").write_text("{", encoding="utf-8")
            (workspace / "opt-note.md").write_text("history\n", encoding="utf-8")
            (workspace / "opt-round-1").mkdir()

            result = bootstrap_runtime_state(workspace)

            self.assertIsNotNone(result.additional_context)
            assert result.additional_context is not None
            self.assertIn("cannot determine source operator from baseline/state.json", result.additional_context)
            self.assertFalse((workspace / ".triton-agent" / "state.json").exists())

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
            self.assertTrue((workspace / ".triton-agent").is_dir())
            state_payload = json.loads(
                (workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(state_payload["baseline"], {"status": "pending", "submitted_at": None})
            self.assertEqual(result.stdout, "")

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
