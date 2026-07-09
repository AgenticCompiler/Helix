import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


_MODULE_NAME = "codex_pretooluse_guard"


def _load_guard_module():
    guard_path = Path(__file__).resolve().parents[1] / "src" / "hook_runtime" / "tool_use_decision.py"
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load guard script: {guard_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CodexPreToolUseGuardTests(unittest.TestCase):
    def test_deny_reason_for_tool_use_matches_read_denial_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"cat {outside}"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_in_workspace_non_protected_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("hello\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"sed -n '1,20p' {readme}"),
            )

            self.assertIsNone(reason)

    def test_blocks_outside_workspace_absolute_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"cat {outside}"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_outside_workspace_parent_escape_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, "sed -n '1,20p' ../outside.txt"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_staged_skill_script_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"sed -n '1,80p' {script}"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_python_one_liner_opening_protected_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"python3 -c \"print(open('{script}').read())\""),
            )

            self.assertIsNone(reason)

    def test_allows_python_entrypoint_for_staged_helper_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"python3 {script} run-test-optimize --test-file differential_test_file.py"),
            )

            self.assertIsNone(reason)

    def test_blocks_staged_claude_skill_script_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".claude" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace, skill_backend_root=".claude"),
                _payload(workspace, f"sed -n '1,80p' {script}"),
            )

            self.assertEqual(reason, _deny_message(".claude"))

    def test_allows_python_entrypoint_for_staged_claude_helper_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".claude" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace, skill_backend_root=".claude"),
                _payload(workspace, f"python3 {script} run-test-optimize --test-file differential_test_file.py"),
            )

            self.assertIsNone(reason)

    def test_allows_relative_python_entrypoint_for_staged_helper_script_with_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "common" / "ascend-npu-run-eval" / "scripts" / "cli.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            bench_file = workspace / "bench_triton_5_MoeInitRouting.py"
            bench_file.write_text("pass\n", encoding="utf-8")
            operator_dir = workspace / "baseline"
            operator_dir.mkdir()
            operator_file = operator_dir / "opt_triton_5_MoeInitRouting.py"
            operator_file.write_text("pass\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(
                    workspace,
                    "python3 .codex/skills/ascend-npu-run-eval/scripts/cli.py "
                    "run-bench --bench-file bench_triton_5_MoeInitRouting.py "
                    "--operator-file baseline/opt_triton_5_MoeInitRouting.py "
                    "--bench-mode msprof 2>&1",
                ),
            )

            self.assertIsNone(reason)

    def test_allows_non_read_python_entrypoint_piped_to_head(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = Path(tmp) / "dist" / "triton-optimizer" / "skills" / "ascend-npu-run-eval" / "scripts" / "cli.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"python3 {script} --help 2>&1 | head -60"),
            )

            self.assertIsNone(reason)

    def test_allows_chained_non_read_commands_before_grep_and_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            cache_path = Path(tmp) / ".triton" / "cache" / "cache-key"
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(
                    workspace,
                    "rm -rf "
                    f"{cache_path} "
                    "2>/dev/null; "
                    "python3 -c \"from importlib.machinery import SourceFileLoader; "
                    "SourceFileLoader('triton_mod', 'triton_16_Repeat.py').load_module()\" "
                    "> /tmp/test_out.txt 2>&1; "
                    "grep -c 'CASE0:OK' /tmp/test_out.txt; "
                    "tail -5 /tmp/test_out.txt",
                ),
            )

            self.assertIsNone(reason)

    def test_blocks_nested_bash_lc_read_of_protected_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"bash -lc \"sed -n '1,20p' {script}\""),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_triton_agent_logs_bash_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"sed -n '1,20p' {log_file}"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_triton_agent_logs_bare_relative_bash_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, "cat triton-agent-logs/gen-test.show-output.log"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_triton_agent_logs_nested_script_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / "triton-agent-logs" / "triton-agent" / "opt_kernel.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('opt')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"python3 -c \"print(open('{script}').read())\""),
            )

            self.assertIsNone(reason)

    def test_allows_python_one_liner_opening_relative_triton_agent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(
                    workspace,
                    'python3 -c "print(open(\'triton-agent-logs/gen-test.show-output.log\').read())"',
                ),
            )

            self.assertIsNone(reason)

    def test_allows_read_outside_triton_agent_logs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "triton-agent-readme.md"
            readme.write_text("not a log\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace),
                _payload(workspace, f"cat {readme}"),
            )

            self.assertIsNone(reason)

    def test_allows_read_from_extra_allow_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            compiler_source = Path(tmp) / "compiler-sources" / "AscendNPU-IR"
            source_file = compiler_source / "passes" / "lowering.cc"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("pass\n", encoding="utf-8")
            workspace.mkdir()
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(
                _policy(workspace, extra_allow_roots=[compiler_source]),
                _payload(workspace, f"sed -n '1,20p' {source_file}"),
            )

            self.assertIsNone(reason)

    def test_blocks_protected_read_tool_payload_with_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            guard = _load_guard_module()
            script = workspace / ".codex" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            payload = _read_payload(script)

            reason = guard.deny_reason_for_tool_use(_policy(workspace), payload)

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_read_tool_payload_outside_workspace_with_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _read_payload(outside))

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_in_workspace_read_tool_payload_with_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("hello\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _read_payload(readme))

            self.assertIsNone(reason)

    def test_blocks_protected_read_tool_payload_with_file_path_camel_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            guard = _load_guard_module()
            script = workspace / ".codex" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            payload = _read_payload(script, key="filePath")

            reason = guard.deny_reason_for_tool_use(_policy(workspace), payload)

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_baseline_phase_allows_native_write_to_regular_workspace_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            notes_file = workspace / "notes.md"
            notes_file.write_text("draft\n", encoding="utf-8")
            _write_workflow_state(
                workspace,
                phase="baseline",
                baseline_status="pending",
            )
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(notes_file))

            self.assertIsNone(reason)

    def test_baseline_phase_blocks_native_write_to_hidden_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            runtime_state = workspace / ".triton-agent" / "state.json"
            runtime_state.parent.mkdir(parents=True)
            _write_workflow_state(
                workspace,
                phase="baseline",
                baseline_status="pending",
            )
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(runtime_state))

            assert reason is not None
            self.assertIn("protected", reason)
            self.assertIn(".triton-agent/", reason)

    def test_awaiting_round_start_blocks_native_write_with_start_round_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-1" / "opt_kernel.py"
            round_file.parent.mkdir(parents=True)
            _write_workflow_state(
                workspace,
                phase="awaiting_round_start",
                baseline_status="passed",
            )
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(round_file))

            assert reason is not None
            self.assertIn("awaiting_round_start", reason)
            self.assertIn("ascend-npu-optimize-state", reason)
            self.assertIn("start-round", reason)

    def test_round_active_allows_native_write_inside_current_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-2" / "attempts.md"
            round_file.parent.mkdir(parents=True)
            _write_workflow_state(
                workspace,
                phase="round_active",
                baseline_status="passed",
                current_round=2,
                rounds={
                    "2": {
                        "status": "active",
                        "round_dir": "opt-round-2",
                        "started_at": "2026-06-23T08:00:00Z",
                        "ended_at": None,
                    }
                },
            )
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(round_file))

            self.assertIsNone(reason)

    def test_round_active_allows_native_write_to_top_level_progress_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            _write_workflow_state(
                workspace,
                phase="round_active",
                baseline_status="passed",
                current_round=2,
                rounds={
                    "2": {
                        "status": "active",
                        "round_dir": "opt-round-2",
                        "started_at": "2026-06-23T08:00:00Z",
                        "ended_at": None,
                    }
                },
            )
            guard = _load_guard_module()

            for file_name in ("opt-note.md", "learned_lessons.md", "supervisor-report.md"):
                with self.subTest(file_name=file_name):
                    reason = guard.deny_reason_for_tool_use(
                        _policy(workspace),
                        _write_payload(workspace / file_name),
                    )

                    self.assertIsNone(reason)

    def test_round_active_blocks_native_write_outside_current_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            operator_file = workspace / "kernel.py"
            operator_file.write_text("pass\n", encoding="utf-8")
            _write_workflow_state(
                workspace,
                phase="round_active",
                baseline_status="passed",
                current_round=2,
                rounds={
                    "2": {
                        "status": "active",
                        "round_dir": "opt-round-2",
                        "started_at": "2026-06-23T08:00:00Z",
                        "ended_at": None,
                    }
                },
            )
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(operator_file))

            assert reason is not None
            self.assertIn("Current active round is opt-round-2", reason)
            self.assertIn("must stay inside `opt-round-2/`", reason)
            self.assertIn("ascend-npu-optimize-state", reason)
            self.assertIn("set-current-round-state", reason)
            self.assertIn("submit-round", reason)

    def test_missing_workflow_state_allows_native_write_after_runtime_path_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-1" / "attempts.md"
            round_file.parent.mkdir(parents=True)
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(round_file))

            self.assertIsNone(reason)

    def test_blocks_runtime_state_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            state_path = workspace / ".triton-agent" / "state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text("{}", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _read_payload(state_path))

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_hidden_runtime_directory_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            runtime_dir = workspace / ".triton-agent"
            runtime_dir.mkdir(parents=True)
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _payload(workspace, "ls .triton-agent"))

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_malformed_payload_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), {"tool_name": "Bash"})

            self.assertIsNone(reason)

_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected runner-managed files (temporary optimize runtime files, "
    "staged skill implementation files under .codex/skills/*/scripts/, or triton-agent-logs/ output). "
    "Use the skill instructions and documented command interface instead."
)


def _deny_message(skill_backend_root: str) -> str:
    return (
        "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
        "and do not inspect protected runner-managed files (temporary optimize runtime files, "
        f"staged skill implementation files under {skill_backend_root}/skills/*/scripts/, or "
        "triton-agent-logs/ output). "
        "Use the skill instructions and documented command interface instead."
    )


def _policy(
    workspace: Path,
    extra_allow_roots: Optional[list[Path]] = None,
    *,
    skill_backend_root: str = ".codex",
) -> dict[str, object]:
    root = workspace.resolve()
    allow_read_roots = [str(root)]
    if extra_allow_roots is not None:
        allow_read_roots.extend(str(path.resolve()) for path in extra_allow_roots)
    return {
        "workspace_root": str(root),
        "allow_read_roots": allow_read_roots,
        "deny_read_globs": [
            str(root / ".triton-agent"),
            str(root / ".triton-agent" / "**"),
            str(root / skill_backend_root / "triton-agent-hooks"),
            str(root / skill_backend_root / "triton-agent-hooks" / "**"),
            str(root / "triton-agent-logs" / "**"),
            str(root / skill_backend_root / "skills" / "*" / "scripts" / "**"),
        ],
        "deny_message": _deny_message(skill_backend_root),
    }


def _payload(workspace: Path, command: str) -> dict[str, object]:
    return {
        "session_id": "session",
        "turn_id": "turn",
        "transcript_path": None,
        "cwd": str(workspace),
        "hook_event_name": "PreToolUse",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_use_id": "call-1",
    }


def _read_payload(path: Path, *, key: str = "file_path") -> dict[str, object]:
    return {
        "session_id": "session",
        "turn_id": "turn",
        "transcript_path": None,
        "cwd": str(path.parent),
        "hook_event_name": "PreToolUse",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": "Read",
        "tool_input": {key: str(path)},
        "tool_use_id": "call-1",
    }


def _write_payload(path: Path, *, tool_name: str = "Write", key: str = "file_path") -> dict[str, object]:
    return {
        "session_id": "session",
        "turn_id": "turn",
        "transcript_path": None,
        "cwd": str(path.parent),
        "hook_event_name": "PreToolUse",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": tool_name,
        "tool_input": {key: str(path), "content": "updated\n"},
        "tool_use_id": "call-1",
    }


def _write_workflow_state(
    workspace: Path,
    *,
    phase: str,
    baseline_status: str,
    current_round: Optional[int] = None,
    rounds: Optional[dict[str, object]] = None,
) -> None:
    state_path = workspace / ".triton-agent" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": "optimize-20260623-guard",
        "phase": phase,
        "current_round": current_round,
        "baseline": {
            "status": baseline_status,
            "submitted_at": None if baseline_status == "pending" else "2026-06-23T07:55:00Z",
        },
        "rounds": rounds or {},
    }
    state_path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
