import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_allows_python_one_liner_opening_protected_script(self) -> None:
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

            self.assertIsNone(reason)

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

    def test_allows_triton_agent_logs_nested_script_read(self) -> None:
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

            self.assertIsNone(reason)

    def test_allows_python_one_liner_opening_relative_triton_agent_log(self) -> None:
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

            self.assertIsNone(reason)

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

    def test_allows_read_from_extra_allow_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            compiler_source = Path(tmp) / "compiler-sources" / "AscendNPU-IR"
            source_file = compiler_source / "passes" / "lowering.cc"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("pass\n", encoding="utf-8")
            workspace.mkdir()
            guard = _load_guard_module()

            reason = guard.evaluate_payload(
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

    def test_guard_disabled_allows_protected_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            log_file = workspace / "triton-agent-logs" / "trace.jsonl"
            log_file.parent.mkdir(parents=True)
            log_file.write_text("{}\n", encoding="utf-8")
            guard = _load_guard_module()
            policy = _policy(workspace)
            policy["guard"] = {
                "enabled": False,
                "allow_read_roots": policy["allow_read_roots"],
                "deny_read_globs": policy["deny_read_globs"],
                "deny_message": policy["deny_message"],
            }

            reason = guard.evaluate_payload(
                policy,
                _payload(workspace, f"cat {log_file}"),
            )

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

    def test_append_trace_events_records_bash_command_and_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("hello\n", encoding="utf-8")
            trace_path = workspace / "triton-agent-logs" / "otel" / "run-001" / "trace.jsonl"
            guard = _load_guard_module()

            with patch.dict(
                "os.environ",
                {
                    "TRITON_AGENT_OTEL_TRACE_PATH": str(trace_path),
                    "TRITON_AGENT_OTEL_RUN_ID": "run-001",
                    "TRITON_AGENT_OTEL_ROLE": "worker",
                    "TRITON_AGENT_WORKSPACE_ROOT": str(workspace.resolve()),
                },
                clear=False,
            ):
                guard.append_trace_events(
                    _policy(workspace),
                    _payload(workspace, "cat README.md"),
                    blocked=False,
                )

            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["type"] for event in events], ["tool_call", "command", "file_access"])
            self.assertEqual(events[1]["command_kind"], "local_command")
            self.assertEqual(events[2]["path"], "README.md")
            self.assertEqual(events[2]["status"], "started")

    def test_append_trace_events_uses_policy_trace_without_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            trace_path = workspace / "triton-agent-logs" / "otel" / "run-001" / "trace.jsonl"
            guard = _load_guard_module()
            policy = _policy(workspace)
            policy["trace"] = {
                "enabled": True,
                "path": str(trace_path),
                "run_id": "run-001",
                "role": "worker",
            }

            guard.append_trace_events(
                policy,
                _payload(workspace, "cat README.md"),
                blocked=False,
            )

            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["source"], "codex_hook")
            self.assertEqual(events[0]["phase"], "start")
            self.assertEqual(events[0]["run_id"], "run-001")

    def test_append_trace_events_records_read_tool_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            trace_path = workspace / "triton-agent-logs" / "otel" / "run-001" / "trace.jsonl"
            guard = _load_guard_module()
            policy = _policy(workspace)
            policy["trace"] = {
                "enabled": True,
                "path": str(trace_path),
                "run_id": "run-001",
                "role": "worker",
            }

            guard.append_trace_events(
                policy,
                _tool_payload(workspace, "Read", {"file_path": "kernel.py"}),
                blocked=False,
            )

            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["type"] for event in events], ["tool_call", "file_access"])
            self.assertEqual(events[1]["action"], "read")
            self.assertEqual(events[1]["path"], "kernel.py")
            self.assertEqual(events[1]["bytes"], (workspace / "kernel.py").stat().st_size)

    def test_append_trace_events_records_edit_tool_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")
            trace_path = workspace / "triton-agent-logs" / "otel" / "run-001" / "trace.jsonl"
            guard = _load_guard_module()
            policy = _policy(workspace)
            policy["trace"] = {
                "enabled": True,
                "path": str(trace_path),
                "run_id": "run-001",
                "role": "worker",
            }

            guard.append_trace_events(
                policy,
                _tool_payload(
                    workspace,
                    "Edit",
                    {
                        "file_path": "kernel.py",
                        "old_string": "print('x')\n",
                        "new_string": "print('y')\nprint('z')\n",
                    },
                ),
                blocked=False,
            )

            events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["type"] for event in events], ["tool_call", "edit"])
            self.assertEqual(events[1]["path"], "kernel.py")
            self.assertEqual(events[1]["edit_kind"], "operator")
            self.assertEqual(events[1]["removed_lines"], 1)
            self.assertEqual(events[1]["added_lines"], 2)
            self.assertTrue(events[1]["diff_digest"].startswith("sha256:"))


_DENY_MESSAGE = (
    "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
    "and do not inspect protected files (staged skill implementation files under "
    ".codex/skills/*/scripts/ or triton-agent-logs/ output). "
    "Use the skill instructions and documented command interface instead."
)


def _policy(workspace: Path, extra_allow_roots: Optional[list[Path]] = None) -> dict[str, object]:
    root = workspace.resolve()
    allow_read_roots = [str(root)]
    if extra_allow_roots is not None:
        allow_read_roots.extend(str(path.resolve()) for path in extra_allow_roots)
    return {
        "workspace_root": str(root),
        "allow_read_roots": allow_read_roots,
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


def _tool_payload(workspace: Path, tool_name: str, tool_input: dict[str, object]) -> dict[str, object]:
    return {
        "session_id": "session",
        "turn_id": "turn",
        "transcript_path": None,
        "cwd": str(workspace),
        "hook_event_name": "PreToolUse",
        "model": "gpt-5.5",
        "permission_mode": "default",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_use_id": "call-1",
    }


if __name__ == "__main__":
    unittest.main()
