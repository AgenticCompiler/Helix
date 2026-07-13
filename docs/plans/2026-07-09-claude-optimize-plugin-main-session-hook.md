# Claude Optimize Plugin Main-Session Hook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the standalone Claude optimize plugin enforce optimize workflow hooks for direct main-session usage even when hook payloads omit `agent_type`.

**Architecture:** Treat the built Claude plugin as optimize-only for main-session lifecycle hooks. `SessionStart`, `SessionEnd`, and `PreToolUse` become unconditional once they receive a valid payload and workspace, while `SubagentStart` and `SubagentStop` keep their current selective ownership behavior. The change is driven by regression tests in both the source-tree hook tests and the built-plugin packaging tests.

**Tech Stack:** Python 3.12, Claude plugin hook wrappers, `unittest`, repository-wide `ruff`, `pyright`, and `pytest`

---

### Task 1: Add Red Tests For Direct Main-Session Hook Behavior

**Files:**
- Modify: `tests/test_claude_optimize_plugin_hooks.py`
- Modify: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Add failing source-tree hook tests for `cwd`-only direct sessions**

```python
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
                (workspace / ".helix" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(state_payload["baseline"], {"status": "pending", "submitted_at": None})

    def test_session_end_removes_runtime_dir_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("{}", encoding="utf-8")
            (workspace / "baseline").mkdir()

            result = _run_hook(
                "session_end.py",
                {
                    "cwd": str(workspace),
                },
            )

            self.assertEqual(result.returncode, 0)
            self.assertFalse((workspace / ".helix").exists())
            self.assertTrue((workspace / "baseline").exists())

    def test_pretooluse_guard_denies_protected_runtime_read_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            protected_path = workspace / ".helix" / "state.json"

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
            self.assertEqual(payload["hookSpecificOutput"]["permissionDecision"], "deny")
            self.assertIn(
                "blocked by helix workspace policy",
                payload["hookSpecificOutput"]["permissionDecisionReason"],
            )
```

- [ ] **Step 2: Add failing built-plugin tests proving the packaged `SessionStart` and `SessionEnd` hooks also work without `agent_type`**

```python
    def test_built_plugin_session_start_bootstraps_baseline_state_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-optimizer")
            workspace = tmpdir / "workspace"
            workspace.mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "session_start.py")],
                input=json.dumps({"cwd": str(workspace)}),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            state_payload = json.loads(
                (workspace / ".helix" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state_payload["phase"], "baseline")
            self.assertEqual(state_payload["baseline"], {"status": "pending", "submitted_at": None})

    def test_built_plugin_session_end_removes_runtime_dir_without_agent_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            plugin_dir = build_claude_optimize_plugin(tmpdir / "triton-optimizer")
            workspace = tmpdir / "workspace"
            workspace.mkdir()
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("{}", encoding="utf-8")
            (workspace / "baseline").mkdir()

            completed = subprocess.run(
                [sys.executable, str(plugin_dir / "hooks" / "session_end.py")],
                input=json.dumps({"cwd": str(workspace)}),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((workspace / ".helix").exists())
            self.assertTrue((workspace / "baseline").exists())
```

- [ ] **Step 3: Run the focused hook test targets and verify they fail for the new `cwd`-only cases**

Run:

```bash
uv run python -m unittest \
  tests.test_claude_optimize_plugin_hooks \
  tests.test_claude_optimize_plugin -v
```

Expected: FAIL in the new `cwd`-only tests because `session_start.py`, `session_end.py`, and `pretooluse_guard.py` currently return early when the shared optimize payload gate sees no `agent_type`.

### Task 2: Implement Unconditional Main-Session Hooks And Preserve Subagent Ownership Logic

**Files:**
- Modify: `hooks/claude_plugin/session_start.py`
- Modify: `hooks/claude_plugin/session_end.py`
- Modify: `hooks/claude_plugin/pretooluse_guard.py`
- Modify: `hooks/claude_plugin/state_bootstrap.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`
- Test: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Remove `should_manage_payload(...)` from `SessionStart` so any valid direct plugin session bootstraps runtime state**

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from state_bootstrap import bootstrap_runtime_state, resolve_workspace


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SessionStart failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    try:
        result = bootstrap_runtime_state(workspace)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SessionStart failed open: {exc}", file=sys.stderr)
        return 0
    if not result.additional_context:
        return 0
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": result.additional_context,
            }
        },
        sys.stdout,
    )
    return 0
```

- [ ] **Step 2: Remove `should_manage_payload(...)` from `SessionEnd` and `PreToolUse`, but keep `SubagentStart` selective**

```python
# hooks/claude_plugin/session_end.py
from state_bootstrap import cleanup_runtime_tree, resolve_workspace


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SessionEnd failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0
    try:
        cleanup_runtime_tree(workspace / ".helix")
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin SessionEnd failed open: {exc}", file=sys.stderr)
    return 0
```

```python
# hooks/claude_plugin/pretooluse_guard.py
from state_bootstrap import compiler_source_read_root, resolve_workspace


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001 - hook must fail open
        print(f"helix claude plugin PreToolUse failed open: {exc}", file=sys.stderr)
        return 0
    if not isinstance(payload, dict):
        return 0
    workspace = resolve_workspace(payload)
    if workspace is None:
        return 0

    return run_with_policy(
        policy=_policy(workspace),
        payload=payload,
        failure_prefix="helix claude plugin PreToolUse",
    )
```

Keep `hooks/claude_plugin/subagent_start.py` using:

```python
    if not isinstance(payload, dict) or not is_optimize_subagent_payload(payload):
        return 0
```

so optimize subagent ownership recording stays precise.

- [ ] **Step 3: Narrow the helper comment/export surface in `state_bootstrap.py` so its remaining role is clearly subagent-focused**

```python
def is_optimize_subagent_payload(payload: dict[str, object]) -> bool:
    """Return True only for optimize subagent or typed-agent payloads."""
    agent_type = resolve_agent_type(payload)
    if agent_type is None:
        return False
    return agent_type == PLUGIN_AGENT_NAME or agent_type.endswith(f":{PLUGIN_AGENT_NAME}")
```

Do not change `resolve_agent_type(...)`, `record_runtime_owner(...)`, or `should_cleanup_for_subagent(...)`.

- [ ] **Step 4: Re-run the focused hook test targets and verify they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_claude_optimize_plugin_hooks \
  tests.test_claude_optimize_plugin -v
```

Expected: PASS for the new `cwd`-only direct-session tests and the unchanged subagent ownership tests.

### Task 3: Run Standard Repository Verification

**Files:**
- Verify: `hooks/claude_plugin/session_start.py`
- Verify: `hooks/claude_plugin/session_end.py`
- Verify: `hooks/claude_plugin/pretooluse_guard.py`
- Verify: `hooks/claude_plugin/state_bootstrap.py`
- Verify: `tests/test_claude_optimize_plugin_hooks.py`
- Verify: `tests/test_claude_optimize_plugin.py`

- [ ] **Step 1: Run `ruff` for the repository**

Run:

```bash
uv run --group dev ruff check
```

Expected: PASS with no new lint errors.

- [ ] **Step 2: Run `pyright` for the repository**

Run:

```bash
uv run pyright
```

Expected: PASS with no new type errors.

- [ ] **Step 3: Run the standard pytest command for the repository**

Run:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS, including the Claude plugin hook and plugin-builder regressions.
