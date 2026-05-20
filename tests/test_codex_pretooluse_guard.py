import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_guard_module():
    guard_path = Path(__file__).resolve().parents[1] / "hooks" / "codex" / "pretooluse_guard.py"
    spec = importlib.util.spec_from_file_location("codex_pretooluse_guard", guard_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load guard script: {guard_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CodexPreToolUseGuardTests(unittest.TestCase):
    def test_allows_in_workspace_non_protected_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("hello\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
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

            reason = guard.evaluate_payload(
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

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, "sed -n '1,20p' ../outside.txt"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_staged_skill_script_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, f"sed -n '1,80p' {script}"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_python_one_liner_opening_protected_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, f"python3 -c \"print(open('{script}').read())\""),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_python_entrypoint_for_staged_helper_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, f"python3 {script} run-test --test-file differential_test_file.py"),
            )

            self.assertIsNone(reason)

    def test_allows_relative_python_entrypoint_for_staged_helper_script_with_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / ".codex" / "skills" / "triton-npu-run-eval" / "scripts" / "run-command.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            bench_file = workspace / "bench_triton_5_MoeInitRouting.py"
            bench_file.write_text("pass\n", encoding="utf-8")
            operator_dir = workspace / "baseline"
            operator_dir.mkdir()
            operator_file = operator_dir / "opt_triton_5_MoeInitRouting.py"
            operator_file.write_text("pass\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(
                    workspace,
                    "python3 .codex/skills/triton-npu-run-eval/scripts/run-command.py "
                    "run-bench --bench-file bench_triton_5_MoeInitRouting.py "
                    "--operator-file baseline/opt_triton_5_MoeInitRouting.py "
                    "--bench-mode msprof 2>&1",
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

            reason = guard.evaluate_payload(
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

            reason = guard.evaluate_payload(
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

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, "cat triton-agent-logs/gen-test.show-output.log"),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_triton_agent_los_nested_script_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            script = workspace / "triton-agent-logs" / "triton-agent" / "opt_kernel.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('opt')\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, f"python3 -c \"print(open('{script}').read())\""),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_python_one_liner_opening_relative_triton_agent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "gen-test.show-output.log"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("log output\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(
                    workspace,
                    'python3 -c "print(open(\'triton-agent-logs/gen-test.show-output.log\').read())"',
                ),
            )

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_read_outside_triton_agent_logs_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "triton-agent-readme.md"
            readme.write_text("not a log\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
                _policy(workspace),
                _payload(workspace, f"cat {readme}"),
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

            reason = guard.evaluate_payload(_policy(workspace), payload)

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_blocks_read_tool_payload_outside_workspace_with_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(_policy(workspace), _read_payload(outside))

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_allows_in_workspace_read_tool_payload_with_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("hello\n", encoding="utf-8")
            guard = _load_guard_module()

            reason = guard.evaluate_payload(_policy(workspace), _read_payload(readme))

            self.assertIsNone(reason)

    def test_blocks_protected_read_tool_payload_with_file_path_camel_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            guard = _load_guard_module()
            script = workspace / ".codex" / "skills" / "skill-a" / "scripts" / "helper.py"
            script.parent.mkdir(parents=True)
            script.write_text("print('helper')\n", encoding="utf-8")
            payload = _read_payload(script, key="filePath")

            reason = guard.evaluate_payload(_policy(workspace), payload)

            self.assertEqual(reason, _DENY_MESSAGE)

    def test_malformed_payload_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            guard = _load_guard_module()

            reason = guard.evaluate_payload(_policy(workspace), {"tool_name": "Bash"})

            self.assertIsNone(reason)

    def test_build_denial_output_uses_pretooluse_permission_decision_shape(self) -> None:
        guard = _load_guard_module()

        output = guard.build_denial_output(_DENY_MESSAGE)

        self.assertEqual(
            output,
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": _DENY_MESSAGE,
                }
            },
        )


_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected files (staged skill implementation files under "
    ".codex/skills/*/scripts/ or triton-agent-logs/ output). "
    "Use the skill instructions and documented command interface instead."
)


def _policy(workspace: Path) -> dict[str, object]:
    root = workspace.resolve()
    return {
        "workspace_root": str(root),
        "allow_read_roots": [str(root)],
        "deny_read_globs": [
            str(root / "triton-agent-logs" / "**"),
            str(root / ".codex" / "skills" / "*" / "scripts" / "**"),
        ],
        "deny_message": _DENY_MESSAGE,
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


if __name__ == "__main__":
    unittest.main()
