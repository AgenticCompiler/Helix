# Codex PreToolUse Hook Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `optimize --enable-agent-hooks` path that stages a workspace-local Codex `PreToolUse` hook to block shell reads outside the workspace and shell reads of staged skill implementation files under `.codex/skills/*/scripts/`.

**Architecture:** Keep hook behavior as repository templates under `hooks/codex/`, then add a small `AgentHookManager` that copies the Codex hook config and guard script into each target workspace before the Codex runner launches. Integrate this manager at `AgentRunner.run()` behind an explicit `AgentRequest.enable_agent_hooks` flag so retry paths share one lifecycle, while hooks stay disabled by default and only `optimize` exposes the initial CLI switch.

**Tech Stack:** Python 3.9, Codex `hooks.json`, JSON stdin/stdout hook protocol, `unittest`, existing process runner and backend runner abstractions

---

## File Map

- Create: `hooks/codex/hooks.json`
- Create: `hooks/codex/pretooluse_guard.py`
- Create: `src/triton_agent/agent_hooks.py`
- Create: `tests/test_agent_hooks.py`
- Create: `tests/test_codex_pretooluse_guard.py`
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/backends/base.py`
- Modify: `tests/test_backends_base.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_commands.py`
- Modify: `tests/test_optimize_runtime.py`
- Modify: `README.md`

The hook guard is deliberately not added to `skills/`. It protects the staged skill boundary but is backend launch infrastructure, not an agent workflow guide.

## Task 1: Add Failing Hook Staging Tests

**Files:**
- Create: `tests/test_agent_hooks.py`

- [ ] **Step 1: Write tests for Codex hook staging, cleanup, conflict handling, and non-Codex no-op behavior**

Create `tests/test_agent_hooks.py`:

```python
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

            state = manager.prepare_hooks("codex", workspace)

            hooks_json = workspace / ".codex" / "hooks.json"
            hook_dir = workspace / ".codex" / "triton-agent-hooks"
            policy_json = hook_dir / "policy.json"
            guard_script = hook_dir / "pretooluse_guard.py"
            self.assertTrue(hooks_json.exists())
            self.assertTrue(policy_json.exists())
            self.assertTrue(guard_script.exists())
            self.assertEqual(state.created_paths, [hooks_json, hook_dir])

            policy = json.loads(policy_json.read_text(encoding="utf-8"))
            self.assertEqual(policy["workspace_root"], str(workspace.resolve()))
            self.assertEqual(policy["allow_read_roots"], [str(workspace.resolve())])
            self.assertEqual(
                policy["deny_read_globs"],
                [str(workspace.resolve() / ".codex" / "skills" / "*" / "scripts" / "**")],
            )
            self.assertIn("triton-agent workspace policy", policy["deny_message"])

            warnings = manager.cleanup(state)

            self.assertEqual(warnings, [])
            self.assertFalse(hooks_json.exists())
            self.assertFalse(hook_dir.exists())
            self.assertTrue((workspace / ".codex").exists())

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

    def test_prepare_hooks_for_non_codex_backend_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            manager = AgentHookManager(Path(__file__).resolve().parents[1] / "hooks")

            state = manager.prepare_hooks("claude", workspace)

            self.assertEqual(state.created_paths, [])
            self.assertFalse((workspace / ".codex").exists())
            self.assertEqual(manager.cleanup(state), [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new staging tests and confirm they fail**

Run: `uv run python -m unittest tests.test_agent_hooks -v`

Expected: `ERROR` with `ModuleNotFoundError: No module named 'triton_agent.agent_hooks'`.

## Task 2: Add Failing Guard Script Tests

**Files:**
- Create: `tests/test_codex_pretooluse_guard.py`

- [ ] **Step 1: Write tests for allow, deny, stdin payload semantics, and fail-open behavior**

Create `tests/test_codex_pretooluse_guard.py`:

```python
import importlib.util
import json
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

    def test_ignores_non_bash_tool_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            guard = _load_guard_module()
            payload = _payload(workspace, "cat /etc/passwd")
            payload["tool_name"] = "Read"

            reason = guard.evaluate_payload(_policy(workspace), payload)

            self.assertIsNone(reason)

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
    "and do not inspect staged skill implementation files under .codex/skills/*/scripts/. "
    "Use the skill instructions and documented command interface instead."
)


def _policy(workspace: Path) -> dict[str, object]:
    root = workspace.resolve()
    return {
        "workspace_root": str(root),
        "allow_read_roots": [str(root)],
        "deny_read_globs": [str(root / ".codex" / "skills" / "*" / "scripts" / "**")],
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the guard tests and confirm they fail**

Run: `uv run python -m unittest tests.test_codex_pretooluse_guard -v`

Expected: `ERROR` because `hooks/codex/pretooluse_guard.py` does not exist yet.

## Task 3: Add Codex Hook Templates And Guard Script

**Files:**
- Create: `hooks/codex/hooks.json`
- Create: `hooks/codex/pretooluse_guard.py`

- [ ] **Step 1: Add the Codex hook configuration template**

Create `hooks/codex/hooks.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 .codex/triton-agent-hooks/pretooluse_guard.py --policy .codex/triton-agent-hooks/policy.json"
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 2: Generate the workspace policy during staging**

- [ ] **Step 3: Add the guard script implementation**

Create `hooks/codex/pretooluse_guard.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


READ_COMMANDS = {
    "awk",
    "cat",
    "head",
    "less",
    "more",
    "python",
    "python3",
    "rg",
    "sed",
    "tail",
}

PATH_FRAGMENT_RE = re.compile(r"(?P<path>(?:/|\.\.?/|\.codex/)[A-Za-z0-9_./*?{}+@%:,=-]+)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    args = parser.parse_args(argv)

    try:
        policy = load_policy(Path(args.policy))
        payload = json.load(sys.stdin)
    except Exception as exc:
        print(f"triton-agent hook guard failed open: {exc}", file=sys.stderr)
        return 0

    reason = evaluate_payload(policy, payload)
    if reason is None:
        return 0

    json.dump(build_denial_output(reason), sys.stdout)
    sys.stdout.write("\n")
    return 0


def load_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("policy must be a JSON object")
    return payload


def evaluate_payload(policy: dict[str, Any], payload: dict[str, Any]) -> str | None:
    try:
        tool_name = str(payload["tool_name"])
        tool_input = payload["tool_input"]
        cwd = Path(str(payload["cwd"]))
    except Exception:
        return None

    if tool_name != "Bash" or not isinstance(tool_input, dict):
        return None

    command = tool_input.get("command")
    if not isinstance(command, str):
        return None

    tokens = _split_command(command)
    if not _has_read_command(tokens):
        return None

    workspace_root = Path(str(policy["workspace_root"])).resolve()
    allow_roots = [Path(str(path)).resolve() for path in policy.get("allow_read_roots", [])]
    deny_globs = [str(pattern) for pattern in policy.get("deny_read_globs", [])]
    deny_message = str(policy["deny_message"])

    for candidate in _candidate_paths(command, tokens):
        resolved = _resolve_candidate(candidate, cwd, workspace_root)
        if resolved is None:
            continue
        if not any(_is_relative_to(resolved, root) for root in allow_roots):
            return deny_message
        if any(fnmatch.fnmatch(str(resolved), pattern) for pattern in deny_globs):
            return deny_message
    return None


def build_denial_output(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _has_read_command(tokens: list[str]) -> bool:
    return any(Path(token).name in READ_COMMANDS for token in tokens)


def _candidate_paths(command: str, tokens: list[str]) -> list[str]:
    candidates: list[str] = []
    for token in tokens:
        if token.startswith("-") or _is_read_command_token(token):
            continue
        if _looks_like_path(token):
            candidates.append(token)
    for match in PATH_FRAGMENT_RE.finditer(command):
        path = match.group("path")
        if not _is_read_command_token(path):
            candidates.append(path)
    return candidates


def _is_read_command_token(token: str) -> bool:
    return Path(token).name in READ_COMMANDS


def _looks_like_path(token: str) -> bool:
    return token.startswith(("/", "./", "../", ".codex/")) or "/" in token


def _resolve_candidate(candidate: str, cwd: Path, workspace_root: Path) -> Path | None:
    if not candidate or candidate in {".", ".."}:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve(strict=False)
    except OSError:
        return (workspace_root / candidate).resolve(strict=False)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run guard tests and confirm they pass**

Run: `uv run python -m unittest tests.test_codex_pretooluse_guard -v`

Expected: `OK`.

## Task 4: Implement Hook Staging Manager

**Files:**
- Create: `src/triton_agent/agent_hooks.py`

- [ ] **Step 1: Add the hook staging manager**

Create `src/triton_agent/agent_hooks.py`:

```python
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


_CODEX_HOOK_DIR = Path(".codex") / "triton-agent-hooks"
_DENY_READ_GLOBS = (Path(".codex") / "skills" / "*" / "scripts" / "**",)


@dataclass
class AgentHookState:
    created_paths: list[Path]


class AgentHookManager:
    def __init__(self, hooks_root: Path) -> None:
        self.hooks_root = hooks_root.resolve()

    def prepare_hooks(self, backend: str, workdir: Path) -> AgentHookState:
        if backend != "codex":
            return AgentHookState([])
        return self._prepare_codex_hooks(workdir)

    def _prepare_codex_hooks(self, workdir: Path) -> AgentHookState:
        workspace = workdir.resolve()
        codex_dir = workspace / ".codex"
        hooks_json = codex_dir / "hooks.json"
        target_hook_dir = workspace / _CODEX_HOOK_DIR
        template_dir = self.hooks_root / "codex"

        if hooks_json.exists():
            raise RuntimeError(f"Existing Codex hooks config must not be overwritten: {hooks_json}")
        if target_hook_dir.exists():
            raise RuntimeError(f"Existing Codex hook directory must not be overwritten: {target_hook_dir}")

        codex_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(template_dir / "hooks.json", hooks_json)
        target_hook_dir.mkdir(parents=True)
        shutil.copy2(template_dir / "pretooluse_guard.py", target_hook_dir / "pretooluse_guard.py")
        self._render_policy(target_hook_dir / "policy.json", workspace)
        return AgentHookState([hooks_json, target_hook_dir])

    def _render_policy(self, target_path: Path, workspace: Path) -> None:
        policy = {
            "workspace_root": str(workspace),
            "allow_read_roots": [str(workspace)],
            "deny_read_globs": [str(workspace / path) for path in _DENY_READ_GLOBS],
            "deny_message": "This read is blocked by triton-agent workspace policy. Stay within the current workspace and do not inspect staged skill implementation files under .codex/skills/*/scripts/. Use the skill instructions and documented command interface instead.",
        }
        target_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    def cleanup(self, state: AgentHookState) -> list[str]:
        warnings: list[str] = []
        for path in reversed(state.created_paths):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                elif path.exists() or path.is_symlink():
                    path.unlink()
            except OSError as exc:
                warnings.append(f"Failed to remove hook staging path {path}: {exc}")
        return warnings

    def describe_prepare(self, state: AgentHookState) -> list[str]:
        if not state.created_paths:
            return ["No agent hooks were staged."]
        return [f"staged agent hook path {path}" for path in state.created_paths]

    def describe_cleanup(self, state: AgentHookState) -> list[str]:
        if not state.created_paths:
            return ["No agent hook paths needed cleanup."]
        return [f"removed agent hook path {path}" for path in reversed(state.created_paths)]
```

- [ ] **Step 2: Run hook staging tests and confirm they pass**

Run: `uv run python -m unittest tests.test_agent_hooks -v`

Expected: `OK`.

## Task 5: Integrate Hook Staging Behind The Optimize Flag

**Files:**
- Modify: `src/triton_agent/models.py`
- Modify: `src/triton_agent/optimize/models.py`
- Modify: `src/triton_agent/optimize/orchestration.py`
- Modify: `src/triton_agent/commands/optimize.py`
- Modify: `src/triton_agent/cli.py`
- Modify: `src/triton_agent/backends/base.py`
- Modify: `tests/test_backends_base.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_optimize_commands.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add failing opt-in lifecycle and CLI tests**

Add focused tests for the public opt-in behavior:

- `AgentRunner.run()` skips hook staging when `AgentRequest.enable_agent_hooks` is false.
- `AgentRunner.run()` prepares and cleans hooks when `AgentRequest.enable_agent_hooks` is true, including failure cleanup.
- `optimize --enable-agent-hooks` is accepted by the parser and maps into `OptimizeRunOptions`.
- Default optimize requests keep `AgentRequest.enable_agent_hooks` false.
- Explicit optimize hook requests set `AgentRequest.enable_agent_hooks` true.

- [ ] **Step 2: Run the opt-in lifecycle tests and confirm they fail**

Run the focused tests in `tests/test_backends_base.py`, `tests/test_cli.py`, `tests/test_optimize_commands.py`, and `tests/test_optimize_runtime.py`.

Expected: failures because the request model, optimize options, parser, and runner do not yet expose or honor the flag.

- [ ] **Step 3: Integrate hook staging behind the explicit flag**

Modify `src/triton_agent/backends/base.py`:

```python
from pathlib import Path

from triton_agent.agent_hooks import AgentHookManager
```

Add a helper near `_log_launch_command()`:

```python
    def _hook_manager(self) -> AgentHookManager:
        repo_root = Path(__file__).resolve().parents[3]
        return AgentHookManager(repo_root / "hooks")
```

Update `run()` so hook staging wraps the whole launch and retry sequence:

```python
    def run(
        self,
        request: AgentRequest,
        stdout: Optional[TextIO] = None,
        stderr: Optional[TextIO] = None,
    ) -> AgentResult:
        command = self.build_command(request)
        if request.verbose:
            self._log_launch_command(command, stderr or sys.stderr)

        if not request.enable_agent_hooks:
            return self._run_with_retry(command, request, stdout=stdout)

        hook_manager = self._hook_manager()
        hook_state = hook_manager.prepare_hooks(request.agent_name, request.workdir)
        if request.verbose:
            emit_verbose_lines(stderr or sys.stderr, "hooks", hook_manager.describe_prepare(hook_state))
        try:
            return self._run_with_retry(command, request, stdout=stdout)
        finally:
            if request.verbose:
                emit_verbose_lines(stderr or sys.stderr, "hooks", hook_manager.describe_cleanup(hook_state))
            for warning in hook_manager.cleanup(hook_state):
                emit_verbose_lines(stderr or sys.stderr, "hooks", [warning])
```

- Add `enable_agent_hooks: bool = False` to `AgentRequest`.
- Add `enable_agent_hooks: bool = False` to `OptimizeRunOptions`.
- Add `--enable-agent-hooks` to the `optimize` parser only.
- Map the parsed CLI value through `OptimizeRunOptions` into `build_optimize_request()`.

- [ ] **Step 4: Run the opt-in lifecycle tests and confirm they pass**

Run the same focused tests from Step 2.

Expected: `OK`.

## Task 6: Document The Hook Behavior

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short Codex hook behavior note**

Add this paragraph near the shared agent option or environment-variable section in `README.md`:

```markdown
### Optional Codex Hook Guard

When `optimize --enable-agent-hooks --agent codex` is used, `triton-agent` stages a temporary workspace-local `.codex/hooks.json` before launching Codex. The hook blocks shell reads outside the current workspace and shell reads of staged skill implementation files under `.codex/skills/*/scripts/`, returning a short policy message to the agent instead. Existing workspace `.codex/hooks.json` files are not merged or overwritten; `triton-agent` fails fast so user-owned hook configuration stays explicit. Without `--enable-agent-hooks`, no hook files are staged.
```

- [ ] **Step 2: Run a documentation diff check**

Run: `git diff -- README.md docs/specs/2026-05-09-codex-pretooluse-hook-guard-design.md docs/plans/2026-05-09-codex-pretooluse-hook-guard.md`

Expected: The README text matches the implemented behavior and the spec remains aligned with the plan.

## Task 7: Final Verification

**Files:**
- All files changed in Tasks 1-6

- [ ] **Step 1: Run focused tests**

Run: `uv run python -m unittest tests.test_agent_hooks tests.test_codex_pretooluse_guard tests.test_backends_base tests.test_cli tests.test_optimize_commands tests.test_optimize_runtime -v`

Expected: `OK`.

- [ ] **Step 2: Run static checks**

Run: `uv run pyright`

Expected: `0 errors`.

- [ ] **Step 3: Run lint**

Run: `uv run --group dev ruff check`

Expected: `All checks passed!`.

- [ ] **Step 4: Run the full test suite**

Run: `uv run python -m unittest discover -s tests -v`

Expected: `OK`.

- [ ] **Step 5: Review the final diff**

Run: `git diff -- docs/specs/2026-05-09-codex-pretooluse-hook-guard-design.md docs/plans/2026-05-09-codex-pretooluse-hook-guard.md hooks/codex src/triton_agent tests README.md`

Expected: The diff only contains the Codex hook guard design, plan, templates, staging manager, opt-in optimize integration, tests, and README note.
