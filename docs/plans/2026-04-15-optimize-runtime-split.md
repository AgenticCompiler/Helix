# Optimize Runtime Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `run_optimize_request()` into smaller helpers while preserving all optimize runtime behavior.

**Architecture:** Keep shared skill staging and final cleanup in `run_optimize_request()`, and move the supervised and unsupervised execution branches into dedicated internal helper functions in `src/triton_agent/optimize/orchestration.py`.

**Tech Stack:** Python, unittest, ruff, pyright, uv

---

### Task 1: Lock In The New Boundary

**Files:**
- Modify: `tests/test_optimize_runtime.py`

- [ ] **Step 1: Add a failing test asserting `run_optimize_request()` delegates supervised requests to a dedicated helper**
- [ ] **Step 2: Add a failing test asserting `run_optimize_request()` delegates unsupervised requests to a dedicated helper**
- [ ] **Step 3: Run `uv run python -m unittest tests.test_optimize_runtime -v` and confirm the new assertions fail before implementation**

### Task 2: Split The Runtime Function

**Files:**
- Modify: `src/triton_agent/optimize/orchestration.py`

- [ ] **Step 1: Add `_run_supervised_optimize_request(...)`**
- [ ] **Step 2: Add `_run_unsupervised_optimize_request(...)`**
- [ ] **Step 3: Keep `run_optimize_request()` as the shared setup and cleanup shell that dispatches to one helper**
- [ ] **Step 4: Re-run focused optimize runtime tests and confirm behavior stays green**

### Task 3: Verify The Refactor

**Files:**
- Modify: `docs/specs/2026-04-15-optimize-runtime-split-design.md`
- Modify: `docs/plans/2026-04-15-optimize-runtime-split.md`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_optimize_runtime -v`**
- [ ] **Step 2: Run `uv run --group dev ruff check`**
- [ ] **Step 3: Run `uv run pyright`**
