# Orchestration Module Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the optimize and generation package `runtime.py` modules to `orchestration.py` while preserving behavior.

**Architecture:** Keep each package's request-building and request-running functions together, but move them under `orchestration.py` to reflect their real orchestration role. Update all in-repo imports and tests to use the new names, with no compatibility shim.

**Tech Stack:** Python, unittest, ruff, pyright, uv

---

### Task 1: Lock In The New Module Boundary

**Files:**
- Modify: `tests/test_generation_commands.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add a failing test asserting `triton_agent.generation.orchestration` exists and `triton_agent.generation.orchestration` does not**
- [ ] **Step 2: Add a failing test asserting `triton_agent.optimize.orchestration` exists and `triton_agent.optimize.orchestration` does not**
- [ ] **Step 3: Run `uv run python -m unittest tests.test_generation_commands tests.test_optimize_runtime -v` and confirm the new assertions fail before implementation**

### Task 2: Rename The Modules

**Files:**
- Move: `src/triton_agent/generation/orchestration.py` to `src/triton_agent/generation/orchestration.py`
- Move: `src/triton_agent/optimize/orchestration.py` to `src/triton_agent/optimize/orchestration.py`
- Modify: imports under `src/`
- Modify: imports and patch targets under `tests/`

- [ ] **Step 1: Rename the generation and optimize modules**
- [ ] **Step 2: Update package exports, command imports, batch imports, and tests to use `orchestration`**
- [ ] **Step 3: Re-run focused tests and confirm behavior stays green**

### Task 3: Verify The Rename

**Files:**
- Modify: `docs/specs/2026-04-15-orchestration-module-rename-design.md`
- Modify: `docs/plans/2026-04-15-orchestration-module-rename.md`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_generation_commands tests.test_optimize_runtime tests.test_cli -v`**
- [ ] **Step 2: Run `uv run --group dev ruff check`**
- [ ] **Step 3: Run `uv run pyright`**
