# Claude Plugin Compiler Source Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-session compiler-source provisioning to the standalone Claude optimize plugin.

**Architecture:** Move reusable compiler-source preparation into `hook_runtime`, keep the existing CLI import as a facade, and let Claude plugin SessionStart provision while PreToolUse only grants read access to an existing checkout.

**Tech Stack:** Python stdlib, Claude hook JSON payloads, pytest/unittest, existing `hook_runtime`.

---

### Task 1: Runtime Compiler Source Module

**Files:**
- Create: `src/hook_runtime/optimize/compiler_source.py`
- Modify: `src/helix/optimize/compiler_source.py`
- Test: `tests/test_compiler_source.py`

- [ ] Add a failing assertion that the existing CLI import still clones missing checkouts through the shared module.
- [ ] Implement `hook_runtime.optimize.compiler_source` with the current constants, dataclass, default path resolution, clone, validation, and commit inspection.
- [ ] Replace `helix.optimize.compiler_source` with a compatibility import facade.
- [ ] Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_compiler_source.py`.

### Task 2: Plugin SessionStart Provisioning

**Files:**
- Modify: `hooks/claude_plugin/state_bootstrap.py`
- Modify: `hooks/claude_plugin/session_start.py` if needed
- Test: `tests/test_claude_optimize_plugin_hooks.py`

- [ ] Add a failing unit test that `bootstrap_runtime_state(workspace, run_git=fake)` clones the missing default checkout and returns additional compiler-source context.
- [ ] Add a failing unit test that a provisioning failure returns additional context instead of raising.
- [ ] Implement compiler-source bootstrap in `state_bootstrap.py`, with an environment override for tests and emergency disable: `HELIX_CLAUDE_PLUGIN_COMPILER_SOURCE=off`.
- [ ] Keep workflow-state diagnostics intact when both workflow and compiler-source messages are present.
- [ ] Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin_hooks.py`.

### Task 3: Plugin Guard Read Root

**Files:**
- Modify: `hooks/claude_plugin/pretooluse_guard.py`
- Test: `tests/test_claude_optimize_plugin_hooks.py`

- [ ] Add a failing test that the plugin guard allows `Read` for a file under an existing compiler source checkout.
- [ ] Implement policy construction that appends the existing compiler source checkout to `allow_read_roots` without cloning.
- [ ] Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin_hooks.py`.

### Task 4: Built Plugin Packaging And Prose

**Files:**
- Modify: `scripts/build-claude-optimize-plugin.py`
- Modify: `skills/triton/triton-npu-analyze-compiler-source/SKILL.md`
- Test: `tests/test_claude_optimize_plugin.py`
- Test: `tests/test_generation_contracts.py`

- [ ] Add a failing plugin-build test that the built plugin contains `hooks/hook_runtime/optimize/compiler_source.py`.
- [ ] Update the generated agent guidance to mention plugin-prepared compiler source context.
- [ ] Update skill wording from CLI-only to CLI/plugin-provided.
- [ ] Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py tests/test_generation_contracts.py`.

### Task 5: Focused Verification

**Files:**
- No new files

- [ ] Run `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_compiler_source.py tests/test_claude_optimize_plugin_hooks.py tests/test_claude_optimize_plugin.py tests/test_generation_contracts.py`.
- [ ] Run `uv run --group dev ruff check`.
- [ ] Run `uv run pyright`.
