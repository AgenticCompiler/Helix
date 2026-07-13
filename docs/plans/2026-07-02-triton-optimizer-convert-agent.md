# Triton Optimizer Convert Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the generated Claude plugin to `triton-optimizer` and add a Triton-only convert agent with its required convert skills.

**Architecture:** Keep the existing builder script as the single packaging entrypoint. Extend its asset model to resolve optimize skills and Triton convert skills separately, render two agent files, and copy the de-duplicated union of skills into the plugin.

**Tech Stack:** Python, unittest, existing `CommandKind` staging table, existing Claude plugin builder tests.

---

### Task 1: Builder Contract Tests

**Files:**
- Modify: `tests/test_claude_optimize_plugin.py`

- [x] **Step 1: Write failing tests**

Add assertions that `build_claude_optimize_plugin_assets()` exposes both optimize and convert skill sets, renders `agents/helix-convert.md`, and that the built manifest name is `triton-optimizer`.

- [x] **Step 2: Run focused tests to verify failure**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`

Expected: FAIL because the builder still renders only the optimize agent and still names the manifest `helix-optimize`.

### Task 2: Builder Implementation

**Files:**
- Modify: `scripts/build-claude-optimize-plugin.py`

- [x] **Step 1: Implement minimal builder changes**

Resolve `CommandKind.CONVERT` with `language="triton"`, render a new `helix-convert` agent, change plugin metadata to `triton-optimizer`, and copy the de-duplicated union of optimize and convert skills.

- [x] **Step 2: Run focused tests to verify pass**

Run: `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/test_claude_optimize_plugin.py`

Expected: PASS.

### Task 3: Verification

**Files:**
- Verify: `scripts/build-claude-optimize-plugin.py`
- Verify: `tests/test_claude_optimize_plugin.py`

- [x] **Step 1: Run standard checks**

Run:

```bash
uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

Expected: PASS for all checks.
