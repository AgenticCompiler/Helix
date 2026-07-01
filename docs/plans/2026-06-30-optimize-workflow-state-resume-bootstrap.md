# Optimize Workflow State Resume Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify optimize workflow-state startup so runner-managed optimize and Claude plugin optimize sessions both reuse valid runtime state, rebuild awaiting-round-start state from durable optimize artifacts, or bootstrap a fresh baseline state without duplicating durable metadata into `.triton-agent/state.json`.

**Architecture:** Keep one shared startup helper under `src/triton_agent/optimize/` as the single source of truth for runtime-state reuse, rebuild, and fresh bootstrap. Remove `source_operator` from runtime workflow state, relax baseline-phase built-in edit gating so it no longer depends on source-operator matching, and let the Claude plugin reuse the same bootstrap helper by packaging the minimal shared Python support it needs.

**Tech Stack:** Python, JavaScript, `pytest`

---

## File Structure

- `src/triton_agent/optimize/workflow_state.py`: shared runtime-state bootstrap/rebuild entrypoint for runner and Claude plugin callers.
- `src/triton_agent/optimize/resume.py`: reusable durable-artifact inspection and source-operator resolution helpers.
- `src/triton_agent/optimize/session_artifacts.py`: runner-managed optimize startup wiring.
- `hooks/claude_plugin/state_bootstrap.py`: thin wrapper around the shared bootstrap helper.
- `hooks/claude_plugin/session_start.py`: hook entrypoint that emits repair context only when startup cannot produce valid runtime state.
- `scripts/build-claude-optimize-plugin.py`: package the shared bootstrap support into the plugin output.
- `skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`: runtime workflow-state schema and bootstrap logic.
- `skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_baseline.py`: missing-runtime-state repair bootstrap using durable baseline state.
- `hooks/shared/tool_use_guard_policy.py`: Codex/Claude shared built-in edit policy.
- `hooks/opencode/triton-agent-hook-guard.js`: OpenCode built-in edit policy parity.
- `tests/test_optimize_workflow_state.py`: shared bootstrap and runtime-state schema tests.
- `tests/test_optimize_guidance.py`: runner-managed startup integration tests.
- `tests/test_claude_optimize_plugin_hooks.py`: plugin startup and cleanup behavior tests.
- `tests/test_claude_optimize_plugin.py`: plugin packaging tests.
- `tests/test_codex_pretooluse_guard.py`: Python guard behavior tests.
- `tests/test_opencode_hook_guard.py`: JavaScript guard behavior tests.
- `tests/test_skill_command_script.py`: optimize-state CLI contract tests.

### Task 1: Update runtime workflow-state semantics

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_baseline.py`
- Test: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_skill_command_script.py`

- [ ] Remove `source_operator` from the runtime workflow-state bootstrap payload and validation rules.
- [ ] Keep `baseline/state.json` as the durable source for source-operator metadata.
- [ ] Update submit-baseline repair bootstrap so it can recreate missing runtime state without copying `source_operator` into `.triton-agent/state.json`.
- [ ] Update tests so successful runtime-state payload assertions stop expecting `source_operator`.

### Task 2: Share bootstrap semantics across runner and Claude plugin

**Files:**
- Modify: `src/triton_agent/optimize/workflow_state.py`
- Modify: `src/triton_agent/optimize/resume.py`
- Modify: `src/triton_agent/optimize/session_artifacts.py`
- Modify: `hooks/claude_plugin/state_bootstrap.py`
- Modify: `hooks/claude_plugin/session_start.py`
- Modify: `scripts/build-claude-optimize-plugin.py`
- Test: `tests/test_optimize_workflow_state.py`
- Test: `tests/test_optimize_guidance.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`
- Test: `tests/test_claude_optimize_plugin.py`

- [ ] Add a shared helper that:
  - reuses valid `.triton-agent/state.json`
  - rebuilds awaiting-round-start state from durable resumable artifacts
  - bootstraps fresh baseline state otherwise
- [ ] Support two bootstrap call shapes:
  - runner path with an explicit source-operator hint
  - Claude plugin path without an operator hint, resolving it from `baseline/state.json` when possible
- [ ] Keep partial-session and invalid-runtime-state failures explicit instead of silently deleting files.
- [ ] Update plugin packaging so hook code can import the shared helper from bundled runtime support instead of re-implementing recovery logic.

### Task 3: Relax baseline built-in edit gating without weakening protected-path rules

**Files:**
- Modify: `hooks/shared/tool_use_guard_policy.py`
- Modify: `hooks/opencode/triton-agent-hook-guard.js`
- Test: `tests/test_codex_pretooluse_guard.py`
- Test: `tests/test_opencode_hook_guard.py`

- [ ] Remove baseline-phase dependence on `state.source_operator`.
- [ ] Allow ordinary in-workspace built-in edits during `baseline` phase.
- [ ] Keep hard denials for protected internal paths in every phase:
  - `.triton-agent/`
  - `triton-agent-logs/`
  - backend-managed hook directories
  - staged skill implementation directories
- [ ] Keep `awaiting_round_start` and `round_active` semantics unchanged apart from the runtime-state schema adjustment.

### Task 4: Align tests and verification with the new contract

**Files:**
- Modify: `tests/test_optimize_workflow_state.py`
- Modify: `tests/test_optimize_guidance.py`
- Modify: `tests/test_claude_optimize_plugin_hooks.py`
- Modify: `tests/test_claude_optimize_plugin.py`
- Modify: `tests/test_codex_pretooluse_guard.py`
- Modify: `tests/test_opencode_hook_guard.py`
- Modify: `tests/test_skill_command_script.py`

- [ ] Replace plugin expectations of “create `.triton-agent/` only and return repair guidance” with “create or rebuild `state.json` when possible”.
- [ ] Add coverage for plugin rebuild from resumable durable artifacts with no source-operator hook input.
- [ ] Add coverage that baseline-phase edits no longer require source-operator matching.
- [ ] Add coverage that protected hidden runtime and log paths are still denied during baseline.

## Verification

- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_optimize_workflow_state.py`
- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_optimize_guidance.py`
- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin_hooks.py`
- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`
- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_codex_pretooluse_guard.py`
- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_opencode_hook_guard.py`
- [ ] Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_skill_command_script.py`
