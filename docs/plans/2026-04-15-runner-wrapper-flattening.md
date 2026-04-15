# Runner Wrapper Flattening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the top-level `test_runner.py` and `bench_runner.py` wrappers while preserving execution behavior.

**Architecture:** Keep execution-specific normalization and metadata helpers in `src/triton_agent/execution.py`, and move optimize-only perf parsing to optimize-local code so no top-level wrapper module is needed for skill passthroughs.

**Tech Stack:** Python, unittest, ruff, pyright, uv

---

### Task 1: Lock In The New Boundary

**Files:**
- Modify: `tests/test_run_skill_loader.py`

- [x] **Step 1: Add a failing test asserting `triton_agent.test_runner` is removed**
- [x] **Step 2: Add a failing test asserting `triton_agent.bench_runner` is removed**
- [x] **Step 3: Run `uv run python -m unittest tests.test_run_skill_loader -v` and confirm the new assertions fail before implementation**

### Task 2: Remove The Wrapper Modules

**Files:**
- Modify: `src/triton_agent/execution.py`
- Modify: `src/triton_agent/optimize/status.py`
- Modify: `src/triton_agent/optimize/resume.py`
- Delete: `src/triton_agent/test_runner.py`
- Delete: `src/triton_agent/bench_runner.py`

- [x] **Step 1: Repoint optimize resume metadata parsing to `execution.py`**
- [x] **Step 2: Move optimize status perf parsing behind optimize-local helpers**
- [x] **Step 3: Delete the redundant runner wrapper modules**
- [x] **Step 4: Re-run focused tests and confirm behavior stays green**

### Task 3: Verify The Refactor

**Files:**
- Modify: `docs/specs/2026-04-15-runner-wrapper-flattening-design.md`
- Modify: `docs/plans/2026-04-15-runner-wrapper-flattening.md`

- [x] **Step 1: Run `uv run python -m unittest tests.test_run_skill_loader tests.test_optimize_status tests.test_cli -v`**
- [x] **Step 2: Run `uv run --group dev ruff check`**
- [x] **Step 3: Run `uv run pyright` and confirm only the pre-existing `tests/test_optimize_runtime.py` Python-version union syntax error remains**
