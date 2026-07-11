# Backends Package Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the multi-backend runner code under `src/helix/backends/` without changing CLI behavior.

**Architecture:** Move the backend abstraction, factory, and concrete runner implementations into a dedicated package that mirrors the recent generation package refactor. Keep the public behavior stable by limiting the change to file layout and import updates, with tests driving the migration.

**Tech Stack:** Python, unittest, pyright, uv

---

### Task 1: Lock In The New Package Import Surface

**Files:**
- Modify: `tests/test_codex_runner.py`
- Modify: `tests/test_opencode_runner.py`
- Modify: `tests/test_pi_runner.py`
- Modify: `tests/test_claude_runner.py`
- Modify: `tests/test_process_runner.py`
- Create: `tests/test_backends_factory.py`

- [ ] **Step 1: Update runner and process-runner tests to import backend code from `helix.backends.*`**
- [ ] **Step 2: Add direct factory coverage for `create_runner()` class selection and unsupported backend errors**
- [ ] **Step 3: Run the targeted tests and confirm they fail for missing package modules before implementation**

### Task 2: Move Backend Code Into The New Package

**Files:**
- Create: `src/helix/backends/__init__.py`
- Create: `src/helix/backends/base.py`
- Create: `src/helix/backends/factory.py`
- Create: `src/helix/backends/codex.py`
- Create: `src/helix/backends/opencode.py`
- Create: `src/helix/backends/pi.py`
- Create: `src/helix/backends/claude.py`
- Delete: `src/helix/agent.py`
- Delete: `src/helix/backends/factory.py`
- Delete: `src/helix/backends/codex.py`
- Delete: `src/helix/backends/opencode.py`
- Delete: `src/helix/backends/pi.py`
- Delete: `src/helix/backends/claude.py`

- [ ] **Step 1: Create the new package modules with the current backend implementation, keeping behavior unchanged**
- [ ] **Step 2: Preserve the stable helpers that other modules need, including `_UnifiedDiffFilter` in the Codex module**
- [ ] **Step 3: Delete the old flat modules only after the package modules exist**

### Task 3: Update Repository Imports

**Files:**
- Modify: `src/helix/cli.py`
- Modify: `src/helix/generation/orchestration.py`
- Modify: `src/helix/optimize/orchestration.py`

- [ ] **Step 1: Point backend imports at `helix.backends` or the focused package modules**
- [ ] **Step 2: Keep the current CLI wrapper surface stable while switching the implementation imports underneath**
- [ ] **Step 3: Run the targeted tests and make the minimal fixes needed to restore green**

### Task 4: Verify The Refactor

**Files:**
- Modify: `docs/specs/2026-04-09-backends-package-refactor-design.md`
- Modify: `docs/plans/2026-04-09-backends-package-refactor.md`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner tests.test_process_runner tests.test_cli -v`**
- [ ] **Step 2: Run `uv run pyright`**
- [ ] **Step 3: If verification fails, fix only backend-package regressions and re-run verification**
