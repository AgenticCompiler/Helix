# Comparison Skill Wrapper Flattening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant comparison forwarding layers while keeping comparison behavior owned by the triton-npu-run-eval skill modules.

**Architecture:** Keep comparison behavior in `src/helix/commands/comparison.py`, where the CLI already validates arguments and reports errors, and load the direct skill implementation modules `test_runner` and `bench_runner` from there. Do not preserve a separate package bridge for comparison-only helpers in this executable app.

**Tech Stack:** Python, unittest, ruff, pyright, uv

---

### Task 1: Lock In Direct Skill Module Loading

**Files:**
- Modify: `AGENTS.md`
- Modify: `tests/test_comparison_commands.py`
- Modify: `docs/specs/2026-04-15-comparison-skill-wrapper-flattening-design.md`
- Modify: `docs/plans/2026-04-15-comparison-skill-wrapper-flattening.md`

- [ ] **Step 1: Record the durable AGENTS rule that this repo is an executable app first and should not preserve unused internal API layers**
- [ ] **Step 2: Add a failing test asserting comparison helpers live in `helix.commands.comparison` and load `test_runner` and `bench_runner` directly**
- [ ] **Step 3: Run `uv run python -m unittest tests.test_comparison_commands -v` and confirm the new assertion fails before implementation**

### Task 2: Flatten The Wrapper Chain

**Files:**
- Modify: `src/helix/commands/comparison.py`
- Delete: `src/helix/comparison.py`
- Delete: `skills/triton-npu-run-eval/scripts/compare_result.py`
- Delete: `skills/triton-npu-run-eval/scripts/compare_perf.py`

- [ ] **Step 1: Move comparison helper loading into `commands/comparison.py`**
- [ ] **Step 2: Delete the redundant package bridge and compare wrapper skill scripts**
- [ ] **Step 3: Re-run the focused comparison tests and confirm they pass**

### Task 3: Verify No Boundary Regressions

**Files:**
- Modify: `docs/specs/2026-04-15-comparison-skill-wrapper-flattening-design.md`
- Modify: `docs/plans/2026-04-15-comparison-skill-wrapper-flattening.md`

- [ ] **Step 1: Run `uv run python -m unittest tests.test_comparison_commands tests.test_run_skill_loader -v`**
- [ ] **Step 2: Run `uv run --group dev ruff check`**
- [ ] **Step 3: Run `uv run pyright`**
