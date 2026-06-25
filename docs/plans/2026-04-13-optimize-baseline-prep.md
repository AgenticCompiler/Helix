# Optimize Baseline Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a baseline-preparation phase to `optimize` so the workflow can persist one canonical baseline under `baseline/` and force all later canonical performance conclusions to compare against `baseline/perf.txt`.

**Architecture:** Keep baseline orchestration in the CLI/runtime layer and keep the work itself in the existing optimize and eval skills. Introduce a small baseline artifact contract, resolve or create baseline assets before relying on optimize rounds, then teach prompts, gates, and status logic to distinguish canonical baseline state from round parent selection.

**Tech Stack:** Python `dataclasses`, `pathlib`, JSON metadata, existing optimize runtime/supervisor flow, `compare-perf`, Python `unittest`

---

### Task 1: Add The Canonical Baseline Contract

**Files:**
- Create: `src/triton_agent/optimize/baseline.py`
- Modify: `src/triton_agent/optimize/models.py`
- Create: `tests/test_optimize_baseline.py`

- [ ] Add strict helpers for `baseline/state.json`, `baseline/perf.txt`, and the baseline operator snapshot.
- [ ] Add focused tests for baseline discovery and validation.

### Task 2: Make Optimize Recognize Baseline Session State

**Files:**
- Modify: `src/triton_agent/optimize/resume.py`
- Modify: `tests/test_cli.py`

- [ ] Let baseline-only prepared sessions count as resumable optimize state.
- [ ] Fail fast when `continue` or `auto` sees partial baseline state.

### Task 3: Teach Prompts And Guidance About `baseline/`

**Files:**
- Modify: `src/triton_agent/prompts.py`
- Modify: `src/triton_agent/optimize/guidance.py`
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/workflow.md`

- [ ] Tell workers to establish or reuse `baseline/` before creating `opt-round-1`.
- [ ] Tell workers to use `baseline/perf.txt` for canonical optimize-session comparisons.

### Task 4: Enforce Canonical Baseline Usage

**Files:**
- Modify: `src/triton_agent/optimize/round_contract.py`
- Modify: `src/triton_agent/optimize/gate.py`
- Modify: `src/triton_agent/optimize/status.py`
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_gate.py`
- Modify: `tests/test_optimize_status.py`

- [ ] Require round metadata to record `canonical_baseline` and `comparison_target`.
- [ ] Block benchmark-passing rounds when baseline artifacts are missing or the comparison target is not `baseline/perf.txt`.
- [ ] Prefer `baseline/perf.txt` in optimize-status while keeping legacy fallback behavior.

### Task 5: Verify And Document

**Files:**
- Modify: `README.md`
- Add: `docs/specs/2026-04-13-optimize-baseline-prep-design.md`
- Add: `docs/plans/2026-04-13-optimize-baseline-prep.md`

- [ ] Update user-facing docs to describe `baseline/`.
- [ ] Run focused optimize tests, then full lint/type/test verification.
