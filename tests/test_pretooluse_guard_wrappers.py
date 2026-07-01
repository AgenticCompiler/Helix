import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PreToolUseGuardWrapperTests(unittest.TestCase):
    def test_codex_wrapper_emits_pretooluse_deny_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")

            result = _run_wrapper(
                backend="codex",
                workspace=workspace,
                payload=_bash_payload(workspace, f"cat {outside}"),
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": _deny_message(".codex"),
                    }
                },
            )

    def test_claude_wrapper_emits_pretooluse_deny_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")

            result = _run_wrapper(
                backend="claude",
                workspace=workspace,
                payload=_bash_payload(workspace, f"cat {outside}"),
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": _deny_message(".claude"),
                    }
                },
            )

    def test_claude_wrapper_allows_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            readme = workspace / "README.md"
            workspace.mkdir()
            readme.write_text("hello\n", encoding="utf-8")

            result = _run_wrapper(
                backend="claude",
                workspace=workspace,
                payload=_bash_payload(workspace, f"sed -n '1,20p' {readme}"),
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_claude_wrapper_allows_heredoc_write_when_body_mentions_protected_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()

            result = _run_wrapper(
                backend="claude",
                workspace=workspace,
                payload=_bash_payload(
                    workspace,
                    "cat > learned_lessons.md << 'ENDOFFILE'\n"
                    "reference .triton-agent/state.json in prose\n"
                    "ENDOFFILE",
                ),
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_claude_wrapper_still_denies_redirected_read_from_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            outside = Path(tmp) / "outside.txt"
            workspace.mkdir()
            outside.write_text("secret\n", encoding="utf-8")

            result = _run_wrapper(
                backend="claude",
                workspace=workspace,
                payload=_bash_payload(workspace, f"cat {outside} > learned_lessons.md"),
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(
                json.loads(result.stdout),
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": _deny_message(".claude"),
                    }
                },
            )


def _run_wrapper(
    *,
    backend: str,
    workspace: Path,
    payload: dict[str, object],
) -> subprocess.CompletedProcess[str]:
    templates_root = Path(__file__).resolve().parents[1] / "hooks"
    runtime_source = Path(__file__).resolve().parents[1] / "src" / "hook_runtime"
    with tempfile.TemporaryDirectory() as staged_tmp:
        staged_dir = Path(staged_tmp)
        wrapper_path = staged_dir / "pretooluse_guard.py"
        runtime_path = staged_dir / "hook_runtime"
        shutil.copy2(templates_root / backend / "pretooluse_guard.py", wrapper_path)
        shutil.copytree(runtime_source, runtime_path)

        policy_path = staged_dir / "policy.json"
        policy_path.write_text(json.dumps(_policy(workspace, backend_root=f".{backend}")) + "\n", encoding="utf-8")

        return subprocess.run(
            [sys.executable, str(wrapper_path), "--policy", str(policy_path)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )


def _policy(workspace: Path, *, backend_root: str) -> dict[str, object]:
    root = workspace.resolve()
    return {
        "workspace_root": str(root),
        "allow_read_roots": [str(root)],
        "deny_read_globs": [
            str(root / ".triton-agent"),
            str(root / ".triton-agent" / "**"),
            str(root / backend_root / "triton-agent-hooks"),
            str(root / backend_root / "triton-agent-hooks" / "**"),
            str(root / "triton-agent-logs" / "**"),
            str(root / backend_root / "skills" / "*" / "scripts" / "**"),
        ],
        "deny_message": _deny_message(backend_root),
    }


def _deny_message(backend_root: str) -> str:
    return (
        "This read is blocked by triton-agent workspace policy. Stay within the current workspace "
        "and do not inspect protected runner-managed files (temporary optimize runtime files, "
        f"staged skill implementation files under {backend_root}/skills/*/scripts/, or "
        "triton-agent-logs/ output). "
        "Use the skill instructions and documented command interface instead."
    )


def _bash_payload(workspace: Path, command: str) -> dict[str, object]:
    return {
        "session_id": "session",
        "transcript_path": None,
        "cwd": str(workspace),
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
        "tool_use_id": "call-1",
    }
