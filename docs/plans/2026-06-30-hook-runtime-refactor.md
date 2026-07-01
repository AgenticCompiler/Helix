# Hook Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `python_support` and hook-local shared Python copies with a standalone `hook_runtime` package that powers staged hooks, built plugins, and CLI reuse.

**Architecture:** Create `src/hook_runtime/` as the source-of-truth runtime package, stage or build it as `hooks/hook_runtime/`, keep wrappers thin, and move hook-executed optimize bootstrap plus guard-policy logic behind that package. `hook_runtime` stays independent from `triton_agent`, while `triton_agent` delegates inward where needed.

**Tech Stack:** Python, pytest/unittest, existing hook wrappers, Claude plugin builder, optimize skill-script bridge.

---

### Task 1: Lock Packaging Expectations In Tests

**Files:**
- Modify: `tests/test_claude_optimize_plugin.py`
- Modify: `tests/test_pretooluse_guard_wrappers.py`
- Modify: `tests/test_agent_hooks.py`

- [ ] Update plugin-build tests to expect `hooks/hook_runtime/` and to reject `python_support/`.
- [ ] Update staged-wrapper tests so they stage `hook_runtime/` instead of copying flat helper files.
- [ ] Update backend hook-staging tests to assert `hook_runtime/` is present in staged hook payloads.
- [ ] Run the focused test targets and confirm they fail for the expected old-layout assumptions.

### Task 2: Add Direct Runtime Tests

**Files:**
- Modify: `tests/test_optimize_workflow_state.py`
- Modify: `tests/test_claude_optimize_plugin_hooks.py`

- [ ] Adjust optimize workflow-state tests to import the shared runtime implementation through `hook_runtime`.
- [ ] Keep coverage for:
  - existing valid state reuse
  - resumable rebuild
  - fresh baseline bootstrap
  - invalid existing state rejection
- [ ] Keep Claude plugin hook tests proving SessionStart and PreToolUse still enforce the same workflow-state behavior.
- [ ] Run the focused runtime tests and confirm failures point to missing `hook_runtime` implementation or outdated imports.

### Task 3: Introduce `src/hook_runtime/`

**Files:**
- Create: `src/hook_runtime/__init__.py`
- Create: `src/hook_runtime/pretooluse_adapter.py`
- Create: `src/hook_runtime/tool_use_decision.py`
- Create: `src/hook_runtime/skill_loader.py`
- Create: `src/hook_runtime/optimize/__init__.py`
- Create: `src/hook_runtime/optimize/baseline.py`
- Create: `src/hook_runtime/optimize/resume.py`
- Create: `src/hook_runtime/optimize/workflow_state.py`

- [ ] Move shared PreToolUse support into `hook_runtime.pretooluse_adapter`.
- [ ] Move Python tool-use decision logic into `hook_runtime.tool_use_decision`.
- [ ] Add a self-contained skill loader that resolves skills from the runtime package root without importing `triton_agent`.
- [ ] Move optimize bootstrap and durable resume helpers into `hook_runtime.optimize.*`.
- [ ] Keep runtime behavior identical to the current semantics around baseline relaxation and resume rebuild.
- [ ] Re-run the direct runtime tests until they pass.

### Task 4: Rewire Wrappers And CLI Callers

**Files:**
- Modify: `hooks/claude/pretooluse_guard.py`
- Modify: `hooks/codex/pretooluse_guard.py`
- Modify: `hooks/claude_plugin/pretooluse_guard.py`
- Modify: `hooks/claude_plugin/state_bootstrap.py`
- Modify: `src/triton_agent/optimize/workflow_state.py`
- Modify: `src/triton_agent/optimize/resume.py` only if delegation cleanup is needed
- Modify: `src/triton_agent/skill_loader.py` only if shared helpers can now delegate safely

- [ ] Simplify wrappers so they only bootstrap imports and delegate to `hook_runtime`.
- [ ] Remove flat sibling helper loading from wrappers.
- [ ] Make plugin `state_bootstrap.py` delegate to `hook_runtime.optimize.workflow_state`.
- [ ] Make CLI-side optimize workflow-state helpers delegate to `hook_runtime` instead of owning duplicate logic.
- [ ] Run the focused wrapper and optimize-state tests to verify source-tree execution still works.

### Task 5: Update Staging And Plugin Build

**Files:**
- Modify: `src/triton_agent/backends/claude_hooks.py`
- Modify: `src/triton_agent/backends/codex_hooks.py`
- Modify: `scripts/build-claude-optimize-plugin.py`
- Modify: `tests/test_claude_optimize_plugin.py`
- Modify: `tests/test_agent_hooks.py`

- [ ] Change hook staging to copy `src/hook_runtime/` into staged `hooks/hook_runtime/`.
- [ ] Change Claude plugin build to package `hooks/hook_runtime/` and stop copying `python_support/`.
- [ ] Keep plugin settings, wrapper scripts, and skill payload selection unchanged apart from runtime packaging.
- [ ] Run focused build and staging tests until they pass.

### Task 6: Remove Old Fallbacks And Dead Runtime Paths

**Files:**
- Modify: `src/triton_agent/resources.py`
- Modify: `src/triton_agent/skill_catalog.py`
- Delete: `python_support/` contents if present in the repo
- Modify: any tests that still mention `python_support` or flat shared helper copies

- [ ] Remove `python_support`-specific application-root fallback logic.
- [ ] Remove staged-flat skill fallback logic that only existed for the old plugin layout.
- [ ] Delete obsolete runtime-copy code paths and update tests accordingly.
- [ ] Run repository searches to ensure no live code still depends on `python_support`.

### Task 7: Verify Focused Behavior And Regression Surface

**Files:**
- No new files required

- [ ] Run:
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin_hooks.py tests/test_optimize_workflow_state.py tests/test_pretooluse_guard_wrappers.py tests/test_agent_hooks.py tests/test_codex_pretooluse_guard.py tests/test_opencode_hook_guard.py`
- [ ] If touched files under `skills/*/scripts/` changed again during the refactor, run the required file-scoped pyright checks.
- [ ] Do one final `rg -n "python_support|pretooluse_guard_support.py|tool_use_guard_policy.py"` pass and confirm only intentionally retained references remain.
