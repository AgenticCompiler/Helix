# Optimize Loop Naming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename optimize loop and adapter types, align filenames with their responsibilities, and shrink `orchestration.py` to the optimize entrypoint layer without changing behavior.

**Architecture:** Move the loop coordinator into `run_loop.py`, move adapter and execution details into `execution.py`, and leave `orchestration.py` with request building plus the public optimize entrypoint. Runtime flow and retry semantics stay unchanged.

**Tech Stack:** Python 3.9+, unittest, pyright, ruff

---

### Task 1: Lock the rename with failing tests

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_cli.py`

- [ ] Add or update assertions so tests refer to the new optimize loop and adapter names.
- [ ] Run the focused tests and confirm they fail before implementation.

### Task 2: Rename loop and adapter types

**Files:**
- Create: `src/helix/optimize/run_loop.py`
- Create: `src/helix/optimize/execution.py`
- Delete: `src/helix/optimize/run_loop.py`
- Modify: `src/helix/optimize/orchestration.py`

- [ ] Move `OptimizeRunLoop` into `run_loop.py` and remove `supervisor.py`.
- [ ] Move adapter and execution helpers into `execution.py`.
- [ ] Keep `orchestration.py` focused on request building and the public entrypoint.

### Task 3: Update references and verify behavior

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `tests/test_supervisor.py`
- Modify: `tests/test_cli.py`

- [ ] Update all remaining references to the renamed types and helpers.
- [ ] Run targeted tests for optimize runtime, supervisor, and CLI patches.
- [ ] Run `uv run --group dev ruff check`.
- [ ] Run `uv run pyright`.
- [ ] Run `uv run python -m unittest discover -s tests -v`.
