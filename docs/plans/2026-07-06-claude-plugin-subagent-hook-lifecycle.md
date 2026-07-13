# Claude Plugin Subagent Hook Lifecycle Implementation Plan

> **Execution mode:** Implement this plan inline in the current session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support both session and subagent lifecycle bootstrap/cleanup for the Claude optimize plugin, and stop missing `.helix/state.json` from blocking ordinary workspace edits.

**Architecture:** Keep the existing session lifecycle wrappers for direct optimize-agent startup, add plugin-local `SubagentStart` / `SubagentStop` wrappers for subagent startup, make bootstrap/cleanup idempotent across both entrypoints, record an owner marker keyed by Claude `agent_id` so subagent cleanup is precise even when later stop payloads are sparse, and relax missing-state edit handling so `PreToolUse` and the shared Python guard skip workflow-phase gating when state is absent while still denying protected runtime paths.

**Tech Stack:** Python 3.12, Claude plugin hook JSON, shared `src/hook_runtime`, `unittest`, repository `pytest`, plugin packaging tests

---

### Task 1: Add SubagentStart bootstrap and keep SessionStart idempotent

**Files:**
- Create: `hooks/claude_plugin/subagent_start.py`
- Modify: `hooks/claude_plugin/state_bootstrap.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`
- Test: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Write the failing hook tests**

```python
class ClaudeOptimizePluginHookTests(unittest.TestCase):
    def test_session_start_keeps_reusing_existing_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            first = _run_hook(
                "session_start.py",
                {
                    "agent_type": "helix-optimizer:helix-optimizer",
                    "cwd": str(workspace),
                },
            )
            second = _run_hook(
                "session_start.py",
                {
                    "agent_type": "helix-optimizer:helix-optimizer",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(first.returncode, 0)
            self.assertEqual(second.returncode, 0)
            payload = json.loads((workspace / ".helix" / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "baseline")

    def test_subagent_start_bootstraps_baseline_state_for_optimize_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)

            result = _run_hook(
                "subagent_start.py",
                {
                    "hook_event_name": "SubagentStart",
                    "subagent_type": "helix-optimizer",
                    "agent_id": "agent-opt-1",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            state_payload = json.loads(
                (workspace / ".helix" / "state.json").read_text(encoding="utf-8")
            )
            owner_payload = json.loads(
                (workspace / ".helix" / "plugin-owner.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(owner_payload, {"agent_id": "agent-opt-1", "agent_type": "helix-optimizer"})

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
            self.assertFalse((workspace / ".helix").exists())
```

```python
class ClaudeOptimizePluginBuilderTests(unittest.TestCase):
    def test_built_plugin_subagent_start_bootstraps_baseline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-optimizer")
            workspace = tmpdir / "workspace"
            workspace.mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "subagent_start.py")],
                input=json.dumps(
                    {
                        "hook_event_name": "SubagentStart",
                        "subagent_type": "helix-optimizer",
                        "agent_id": "agent-opt-1",
                        "cwd": str(workspace),
                    }
                ),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((workspace / ".helix" / "state.json").exists())
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks tests.test_claude_optimize_plugin -v`

Expected: FAIL because `subagent_start.py` does not exist yet and no owner marker is written.

- [ ] **Step 3: Extend the shared bootstrap helper with subagent payload parsing and owner-marker helpers**

```python
# hooks/claude_plugin/state_bootstrap.py
PLUGIN_OWNER_FILENAME = "plugin-owner.json"
_AGENT_TYPE_KEYS = ("subagent_type", "subagentType", "agent_type")


def resolve_agent_type(payload: dict[str, object]) -> str | None:
    for key in _AGENT_TYPE_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def should_manage_subagent_payload(payload: dict[str, object]) -> bool:
    agent_type = resolve_agent_type(payload)
    return agent_type == PLUGIN_AGENT_NAME


def record_runtime_owner(runtime_dir: Path, *, agent_id: str, agent_type: str) -> None:
    payload = {"agent_id": agent_id, "agent_type": agent_type}
    (runtime_dir / PLUGIN_OWNER_FILENAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def runtime_owner(runtime_dir: Path) -> dict[str, str] | None:
    owner_path = runtime_dir / PLUGIN_OWNER_FILENAME
    try:
        raw = json.loads(owner_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    agent_id = raw.get("agent_id")
    agent_type = raw.get("agent_type")
    if not isinstance(agent_id, str) or not agent_id:
        return None
    if not isinstance(agent_type, str) or not agent_type:
        return None
    return {"agent_id": agent_id, "agent_type": agent_type}
```

```python
def bootstrap_runtime_state(
    workspace: Path,
    *,
    compiler_source_enabled: bool | None = None,
    compiler_source_cache_dir: Path | None = None,
    run_git: RunGit | None = None,
) -> BootstrapResult:
    contexts: list[str] = []
    runtime_dir = workspace / ".helix"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    state_path = runtime_dir / "state.json"
    try:
        prepare_or_restore_optimize_workflow_state(
            None,
            workspace,
            state_path=state_path,
            run_id=_plugin_run_id(),
        )
    except ValueError as exc:
        contexts.append(_workflow_repair_guidance(str(exc)))
    ...
```

- [ ] **Step 4: Add the new `SubagentStart` wrapper**

```python
# hooks/claude_plugin/subagent_start.py
#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import (
    bootstrap_runtime_state,
    record_runtime_owner,
    resolve_agent_type,
    resolve_workspace,
    should_manage_subagent_payload,
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        print(f"helix claude plugin SubagentStart failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict) or not should_manage_subagent_payload(payload):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    agent_id = payload.get("agent_id")
    agent_type = resolve_agent_type(payload)
    if not isinstance(agent_id, str) or not agent_id or agent_type is None:
        return 0
    try:
        result = bootstrap_runtime_state(workspace)
        record_runtime_owner(workspace / ".helix", agent_id=agent_id, agent_type=agent_type)
    except Exception as exc:  # noqa: BLE001
        print(f"helix claude plugin SubagentStart failed open: {exc}", file=sys.stderr)
        return 0
    if not result.additional_context:
        return 0
    json.dump(
        {"hookSpecificOutput": {"hookEventName": "SubagentStart", "additionalContext": result.additional_context}},
        sys.stdout,
    )
    return 0
```

- [ ] **Step 5: Run the focused tests and verify they pass**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks tests.test_claude_optimize_plugin -v`

Expected: PASS for the new `SubagentStart` tests.

- [ ] **Step 6: Commit**

```bash
git add hooks/claude_plugin/state_bootstrap.py hooks/claude_plugin/subagent_start.py tests/test_claude_optimize_plugin_hooks.py tests/test_claude_optimize_plugin.py
git commit -m "feat: bootstrap claude optimize state on subagent start"
```

### Task 2: Add SubagentStop cleanup and extend hook registration to both lifecycles

**Files:**
- Create: `hooks/claude_plugin/subagent_stop.py`
- Modify: `hooks/claude_plugin/hooks.json`
- Modify: `hooks/claude_plugin/state_bootstrap.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`
- Test: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Write the failing cleanup and manifest tests**

```python
class ClaudeOptimizePluginHookTests(unittest.TestCase):
    def test_subagent_stop_removes_runtime_dir_only_for_matching_owner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runtime_dir = workspace / ".helix"
            runtime_dir.mkdir()
            (runtime_dir / "state.json").write_text("{}", encoding="utf-8")
            (runtime_dir / "plugin-owner.json").write_text(
                json.dumps({"agent_id": "agent-opt-1", "agent_type": "helix-optimizer"}),
                encoding="utf-8",
            )
            (workspace / "baseline").mkdir()

            result = _run_hook(
                "subagent_stop.py",
                {
                    "hook_event_name": "SubagentStop",
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
            runtime_dir = workspace / ".helix"
            runtime_dir.mkdir()
            (runtime_dir / "state.json").write_text("{}", encoding="utf-8")
            (runtime_dir / "plugin-owner.json").write_text(
                json.dumps({"agent_id": "agent-opt-1", "agent_type": "helix-optimizer"}),
                encoding="utf-8",
            )

            result = _run_hook(
                "subagent_stop.py",
                {
                    "hook_event_name": "SubagentStop",
                    "agent_id": "agent-diag-1",
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertTrue(runtime_dir.exists())
```

```python
class ClaudeOptimizePluginBuilderTests(unittest.TestCase):
    def test_built_plugin_hooks_manifest_registers_subagent_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_dir = build_claude_optimize_plugin(Path(tmp) / "triton-optimizer")
            hooks = json.loads((plugin_dir / "hooks" / "hooks.json").read_text(encoding="utf-8"))

            self.assertIn("SessionStart", hooks["hooks"])
            self.assertIn("SessionEnd", hooks["hooks"])
            self.assertIn("SubagentStart", hooks["hooks"])
            self.assertIn("SubagentStop", hooks["hooks"])
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks tests.test_claude_optimize_plugin -v`

Expected: FAIL because `subagent_stop.py` is absent and `hooks.json` still only registers session lifecycle hooks.

- [ ] **Step 3: Add a conservative owner-match helper and the stop wrapper**

```python
# hooks/claude_plugin/state_bootstrap.py
def payload_agent_id(payload: dict[str, object]) -> str | None:
    value = payload.get("agent_id")
    if isinstance(value, str) and value:
        return value
    return None


def should_cleanup_for_subagent(payload: dict[str, object], runtime_dir: Path) -> bool:
    owner = runtime_owner(runtime_dir)
    if owner is None:
        return False
    agent_id = payload_agent_id(payload)
    if agent_id is None:
        return False
    return owner["agent_id"] == agent_id and owner["agent_type"] == PLUGIN_AGENT_NAME
```

```python
# hooks/claude_plugin/subagent_stop.py
#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import cleanup_runtime_tree, resolve_workspace, should_cleanup_for_subagent


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        print(f"helix claude plugin SubagentStop failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    runtime_dir = workspace / ".helix"
    if not should_cleanup_for_subagent(payload, runtime_dir):
        return 0
    try:
        cleanup_runtime_tree(runtime_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"helix claude plugin SubagentStop failed open: {exc}", file=sys.stderr)
    return 0
```

- [ ] **Step 4: Extend `hooks.json` to register both session and subagent lifecycle hooks**

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup|clear|compact",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/session_start.py\""
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/session_end.py\""
          }
        ]
      }
    ],
    "SubagentStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/subagent_start.py\""
          }
        ]
      }
    ],
    "SubagentStop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/subagent_stop.py\""
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash|Read|Grep|Glob|Edit|MultiEdit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "python3 \"${CLAUDE_PLUGIN_ROOT}/hooks/pretooluse_guard.py\""
          }
        ]
      }
    ]
  }
}
```

- [ ] **Step 5: Run the focused tests and verify they pass**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks tests.test_claude_optimize_plugin -v`

Expected: PASS for `SubagentStop` cleanup and manifest registration tests, while `SessionStart` / `SessionEnd` remain present.

- [ ] **Step 6: Commit**

```bash
git add hooks/claude_plugin/hooks.json hooks/claude_plugin/state_bootstrap.py hooks/claude_plugin/subagent_stop.py tests/test_claude_optimize_plugin_hooks.py tests/test_claude_optimize_plugin.py
git commit -m "feat: wire claude optimize plugin to subagent lifecycle hooks"
```

### Task 3: Stop the Claude plugin wrapper from denying ordinary edits only because state is missing

**Files:**
- Modify: `hooks/claude_plugin/pretooluse_guard.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`

- [ ] **Step 1: Write the failing plugin-wrapper tests**

```python
class ClaudeOptimizePluginHookTests(unittest.TestCase):
    def test_pretooluse_guard_allows_edit_when_workflow_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "kernel.py").write_text("print('x')\n", encoding="utf-8")

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "helix-optimizer:helix-optimizer",
                    "cwd": str(workspace),
                    "tool_name": "Edit",
                    "tool_input": {"file_path": str(workspace / "kernel.py")},
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")

    def test_pretooluse_guard_still_denies_protected_runtime_edit_when_state_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            protected_path = workspace / ".helix" / "state.json"

            result = _run_hook(
                "pretooluse_guard.py",
                {
                    "agent_type": "helix-optimizer:helix-optimizer",
                    "cwd": str(workspace),
                    "tool_name": "Edit",
                    "tool_input": {"file_path": str(protected_path)},
                },
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["hookSpecificOutput"]["permissionDecision"], "deny")
            self.assertIn("protected internal runtime path", payload["hookSpecificOutput"]["permissionDecisionReason"])
```

- [ ] **Step 2: Run the focused test file and verify it fails**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks -v`

Expected: FAIL because the plugin wrapper still denies ordinary edits before the shared guard runs.

- [ ] **Step 3: Replace the missing-state hard deny with allow-to-fall-through behavior**

```python
# hooks/claude_plugin/pretooluse_guard.py
def main() -> int:
    ...
    tool_name = payload.get("tool_name")
    if tool_name in {"Edit", "MultiEdit", "Write"}:
        # Do not deny ordinary edits solely because workflow state is absent.
        # Shared protected-path and phase checks still run below.
        pass

    return run_with_policy(
        policy=_policy(workspace),
        payload=payload,
        failure_prefix="helix claude plugin PreToolUse",
    )
```

- [ ] **Step 4: Run the focused test file and verify it passes**

Run: `uv run python -m unittest tests.test_claude_optimize_plugin_hooks -v`

Expected: PASS for ordinary workspace edits with missing state, while protected runtime edits remain denied.

- [ ] **Step 5: Commit**

```bash
git add hooks/claude_plugin/pretooluse_guard.py tests/test_claude_optimize_plugin_hooks.py
git commit -m "fix: stop claude plugin from blocking edits on missing state"
```

### Task 4: Make the shared Python guard skip missing-state workflow gating but keep protected-path denial

**Files:**
- Modify: `src/hook_runtime/tool_use_decision.py`
- Test: `tests/test_codex_pretooluse_guard.py`

- [ ] **Step 1: Write the failing shared-guard tests**

```python
class CodexPreToolUseGuardTests(unittest.TestCase):
    def test_missing_workflow_state_allows_native_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            round_file = workspace / "opt-round-1" / "attempts.md"
            round_file.parent.mkdir(parents=True)
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(round_file))

            self.assertIsNone(reason)

    def test_missing_workflow_state_still_blocks_runtime_state_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            protected_path = workspace / ".helix" / "state.json"
            guard = _load_guard_module()

            reason = guard.deny_reason_for_tool_use(_policy(workspace), _write_payload(protected_path))

            assert reason is not None
            self.assertIn("protected internal runtime path", reason)
```

- [ ] **Step 2: Run the focused guard test file and verify it fails**

Run: `uv run python -m unittest tests.test_codex_pretooluse_guard -v`

Expected: FAIL because missing workflow state still returns the restart-hint denial.

- [ ] **Step 3: Change built-in edit handling to skip only the missing-state branch**

```python
# src/hook_runtime/tool_use_decision.py
def _deny_reason_for_built_in_edit_path(path_text: str, context: PathAccessContext) -> str | None:
    resolved_path = _resolve_path_text(path_text, context.cwd, context.workspace_root)
    ...
    workspace_relative_path = resolved_path.relative_to(context.workspace_root).as_posix()
    if _is_protected_runtime_edit_path(workspace_relative_path):
        return _protected_runtime_edit_denial(workspace_relative_path)

    workflow_state = _workflow_state_or_none(context.workspace_root)
    if workflow_state is None:
        return None

    phase = _require_state_string(workflow_state, "phase")
    if phase == "baseline":
        return None
    if phase == "awaiting_round_start":
        return _awaiting_round_start_built_in_edit_denial()
    if phase == "round_active":
        active_round_dir = _active_round_dir(workflow_state)
        if active_round_dir is None:
            return _round_active_state_invalid_denial()
        if _is_allowed_round_active_edit_path(workspace_relative_path, active_round_dir):
            return None
        return _round_active_built_in_edit_denial(active_round_dir)
    return _workflow_phase_invalid_denial(phase)
```

```python
def _round_active_state_invalid_denial() -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        "Current workflow state says a round is active, but the active round entry is invalid. "
        "Repair or restart the optimize session before continuing round edits."
    )


def _workflow_phase_invalid_denial(phase: str) -> str:
    return (
        "Built-in edit tool blocked by optimize workflow policy. "
        f"Workflow phase `{phase}` is not recognized by the edit guard. "
        "Repair or restart the optimize session before continuing."
    )
```

- [ ] **Step 4: Run the focused guard test file and verify it passes**

Run: `uv run python -m unittest tests.test_codex_pretooluse_guard -v`

Expected: PASS for missing-state ordinary writes and protected runtime-path denial, with existing `awaiting_round_start` and `round_active` tests still green.

- [ ] **Step 5: Commit**

```bash
git add src/hook_runtime/tool_use_decision.py tests/test_codex_pretooluse_guard.py
git commit -m "fix: skip optimize edit gating when workflow state is absent"
```

### Task 5: Run final verification and packaging checks

**Files:**
- Modify: `tests/test_claude_optimize_plugin.py`
- Modify: `tests/test_claude_optimize_plugin_hooks.py`
- Modify: `tests/test_codex_pretooluse_guard.py`

- [ ] **Step 1: Run the focused regression suite**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin_hooks.py tests/test_claude_optimize_plugin.py tests/test_codex_pretooluse_guard.py`

Expected: PASS with 0 failures.

- [ ] **Step 2: Run repository-standard verification**

Run: `uv run --group dev ruff check`
Expected: PASS with 0 errors.

Run: `uv run pyright`
Expected: PASS with 0 errors.

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
Expected: PASS with 0 failures.

- [ ] **Step 3: Commit the verification-complete state**

```bash
git add hooks/claude_plugin/hooks.json hooks/claude_plugin/pretooluse_guard.py hooks/claude_plugin/state_bootstrap.py hooks/claude_plugin/subagent_start.py hooks/claude_plugin/subagent_stop.py src/hook_runtime/tool_use_decision.py tests/test_claude_optimize_plugin.py tests/test_claude_optimize_plugin_hooks.py tests/test_codex_pretooluse_guard.py
git commit -m "fix: align claude optimize plugin state with subagent lifecycle"
```
