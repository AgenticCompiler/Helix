from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional, Protocol, cast

HOOKS_ROOT = Path(__file__).resolve().parents[1] / "hooks" / "claude_plugin"


class BootstrapResultLike(Protocol):
    additional_context: str | None


class RunGitLike(Protocol):
    def __call__(self, args: list[str], cwd: Optional[Path] = None) -> str: ...


class BootstrapRuntimeStateLike(Protocol):
    def __call__(
        self,
        workspace: Path,
        *,
        compiler_source_enabled: bool | None = None,
        compiler_source_cache_dir: Path | None = None,
        run_git: RunGitLike | None = None,
    ) -> BootstrapResultLike: ...


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
    BootstrapRuntimeStateLike,
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

            result = bootstrap_runtime_state(workspace, compiler_source_enabled=False)

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

            result = bootstrap_runtime_state(workspace, compiler_source_enabled=False)

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

            result = bootstrap_runtime_state(workspace, compiler_source_enabled=False)

            self.assertIsNotNone(result.additional_context)
            assert result.additional_context is not None
            self.assertIn("cannot determine source operator from baseline/state.json", result.additional_context)
            self.assertFalse((workspace / ".triton-agent" / "state.json").exists())

    def test_bootstrap_runtime_state_prepares_compiler_source_on_first_session_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            cache_dir = root / "cache"
            checkout = cache_dir / "compiler-sources" / "AscendNPU-IR"
            calls: list[list[str]] = []

            def fake_run(args: list[str], cwd: Optional[Path] = None) -> str:
                calls.append(args)
                if args[:2] == ["git", "clone"]:
                    target = Path(args[-1])
                    target.mkdir(parents=True)
                    (target / ".git").mkdir()
                    return ""
                if args == ["git", "rev-parse", "HEAD"]:
                    self.assertEqual(cwd, checkout)
                    return "abc123\n"
                raise AssertionError(args)

            result = bootstrap_runtime_state(
                workspace,
                compiler_source_enabled=True,
                compiler_source_cache_dir=cache_dir,
                run_git=fake_run,
            )

            self.assertTrue((workspace / ".triton-agent" / "state.json").exists())
            self.assertTrue(checkout.is_dir())
            self.assertIn(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://gitcode.com/Ascend/AscendNPU-IR.git",
                    str(checkout),
                ],
                calls,
            )
            self.assertIsNotNone(result.additional_context)
            assert result.additional_context is not None
            self.assertIn("Compiler source analysis is enabled", result.additional_context)
            self.assertIn(str(checkout), result.additional_context)
            self.assertIn("abc123", result.additional_context)

    def test_bootstrap_runtime_state_reports_compiler_source_failure_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            def fake_run(args: list[str], cwd: Optional[Path] = None) -> str:
                del args, cwd
                raise ValueError("network unavailable")

            result = bootstrap_runtime_state(
                workspace,
                compiler_source_enabled=True,
                compiler_source_cache_dir=Path(tmp) / "cache",
                run_git=fake_run,
            )

            self.assertTrue((workspace / ".triton-agent" / "state.json").exists())
            self.assertIsNotNone(result.additional_context)
            assert result.additional_context is not None
            self.assertIn("Compiler source analysis is unavailable", result.additional_context)
            self.assertIn("network unavailable", result.additional_context)

    def test_session_start_returns_repair_guidance_for_namespaced_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _run_hook(
                "session_start.py",
                {
                    "agent_type": "triton-agent-optimizer:triton-agent-optimizer",
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

    def test_session_start_bootstraps_baseline_state_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _run_hook(
                "session_start.py",
                {
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            state_payload = json.loads(
                (workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(state_payload["baseline"], {"status": "pending", "submitted_at": None})

    def test_subagent_start_bootstraps_baseline_state_for_optimize_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _run_hook(
                "subagent_start.py",
                {
                    "hook_event_name": "SubagentStart",
                    "subagent_type": "triton-agent-optimizer",
                    "agent_id": "agent-opt-1",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            state_payload = json.loads(
                (workspace / ".triton-agent" / "state.json").read_text(encoding="utf-8")
            )
            owner_payload = json.loads(
                (workspace / ".triton-agent" / "plugin-owner.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(owner_payload, {"agent_id": "agent-opt-1", "agent_type": "triton-agent-optimizer"})

    def test_subagent_start_ignores_unrelated_subagent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _run_hook(
                "subagent_start.py",
                {
                    "hook_event_name": "SubagentStart",
                    "subagent_type": "researcher",
                    "agent_id": "agent-other-1",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse((workspace / ".triton-agent").exists())

    def test_session_end_removes_runtime_dir_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".triton-agent").mkdir()
            (workspace / ".triton-agent" / "state.json").write_text("{}", encoding="utf-8")
            (workspace / "baseline").mkdir()

            result = _run_hook(
                "session_end.py",
                {
                    "agent_type": "triton-agent-optimizer:triton-agent-optimizer",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse((workspace / ".triton-agent").exists())
            self.assertTrue((workspace / "baseline").exists())

    def test_session_end_removes_runtime_dir_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".triton-agent").mkdir()
            (workspace / ".triton-agent" / "state.json").write_text("{}", encoding="utf-8")
            (workspace / "baseline").mkdir()

            result = _run_hook(
                "session_end.py",
                {
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse((workspace / ".triton-agent").exists())
            self.assertTrue((workspace / "baseline").exists())

    def test_subagent_stop_removes_runtime_dir_only_for_matching_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runtime_dir = workspace / ".triton-agent"
            runtime_dir.mkdir()
            (runtime_dir / "state.json").write_text("{}", encoding="utf-8")
            (runtime_dir / "plugin-owner.json").write_text(
                json.dumps({"agent_id": "agent-opt-1", "agent_type": "triton-agent-optimizer"}),
                encoding="utf-8",
            )
            (workspace / "baseline").mkdir()

            result = _run_hook(
                "subagent_stop.py",
                {
                    "hook_event_name": "SubagentStop",
                    "subagent_type": "triton-agent-optimizer",
                    "agent_id": "agent-opt-1",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse(runtime_dir.exists())
            self.assertTrue((workspace / "baseline").exists())

    def test_subagent_stop_ignores_non_owner_agent_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runtime_dir = workspace / ".triton-agent"
            runtime_dir.mkdir()
            (runtime_dir / "state.json").write_text("{}", encoding="utf-8")
            (runtime_dir / "plugin-owner.json").write_text(
                json.dumps({"agent_id": "agent-opt-1", "agent_type": "triton-agent-optimizer"}),
                encoding="utf-8",
            )

            result = _run_hook(
                "subagent_stop.py",
                {
                    "hook_event_name": "SubagentStop",
                    "subagent_type": "triton-agent-optimizer",
                    "agent_id": "agent-opt-2",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(runtime_dir.exists())

    def test_pretooluse_guard_allows_edit_when_workflow_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "triton-agent-optimizer:triton-agent-optimizer",
                    "cwd": str(workspace),
                    "tool_name": "Edit",
                    "tool_input": {
                        "file_path": str(workspace / "kernel.py"),
                    },
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stderr, "")
            self.assertEqual(result.stdout, "")

    def test_pretooluse_guard_falls_back_to_shared_policy_in_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            protected_path = workspace / ".triton-agent" / "state.json"

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "triton-agent-optimizer:triton-agent-optimizer",
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

    def test_pretooluse_guard_denies_protected_runtime_read_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            protected_path = workspace / ".triton-agent" / "state.json"

            result = _run_hook(
                "pretooluse_guard.py",
                {
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

    def test_pretooluse_guard_allows_workspace_read_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source_file = workspace / "kernel.py"
            source_file.write_text("print('x')\n", encoding="utf-8")

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "cwd": str(workspace),
                    "tool_name": "Read",
                    "tool_input": {
                        "file_path": str(source_file),
                    },
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stderr, "")
            self.assertEqual(result.stdout, "")

    def test_pretooluse_guard_allows_read_from_existing_compiler_source_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            checkout = root / "cache" / "compiler-sources" / "AscendNPU-IR"
            source_file = checkout / "bishengir" / "lib" / "pass.cc"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("// source\n", encoding="utf-8")
            (checkout / ".git").mkdir()

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "triton-agent-optimizer:triton-agent-optimizer",
                    "cwd": str(workspace),
                    "tool_name": "Read",
                    "tool_input": {
                        "file_path": str(source_file),
                    },
                },
                env={
                    "TRITON_AGENT_CLAUDE_PLUGIN_COMPILER_SOURCE": "auto",
                    "TRITON_AGENT_COMPILER_SOURCE_CACHE_DIR": str(root / "cache"),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stderr, "")
            self.assertEqual(result.stdout, "")


def _run_hook(
    script_name: str,
    payload: dict[str, object],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    hook_env = os.environ.copy()
    hook_env.setdefault("TRITON_AGENT_CLAUDE_PLUGIN_COMPILER_SOURCE", "off")
    if env is not None:
        hook_env.update(env)
    return subprocess.run(
        [sys.executable, str(HOOKS_ROOT / script_name)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env=hook_env,
    )


if __name__ == "__main__":
    unittest.main()
