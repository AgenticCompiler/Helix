# Optimize Profile Artifact Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make optimize cleanup retain only profiler CSV artifacts after round submission and at optimize session teardown.

**Architecture:** Reuse optimize-state round helper logic for both triggers. The skill-side round checker remains the source of truth for cleanup semantics, while runtime code calls the same helper through a thin bridge during optimize session teardown.

**Tech Stack:** Python, `unittest`, optimize-state skill scripts

---

### Task 1: Lock Down The Cleanup Contract

**Files:**
- Create: `docs/specs/2026-07-06-optimize-profile-artifact-cleanup-design.md`
- Create: `docs/plans/2026-07-06-optimize-profile-artifact-cleanup.md`

- [ ] **Step 1: Write the design doc**

Capture the dual-trigger cleanup semantics, the CSV-only profile retention
rule, and the workspace-root `PROF_*` / `OPPROF_*` deletion rule.

- [ ] **Step 2: Write the implementation plan**

Record the test-first sequence, touched modules, and focused verification.

### Task 2: Add Failing Regression Coverage

**Files:**
- Create: `tests/test_optimize_profile_cleanup.py`

- [ ] **Step 1: Add a failing `submit-round` cleanup regression**

Create a valid round with a declared `profile/` tree containing both `.csv` and
non-CSV files plus workspace-root `PROF_*` and `OPPROF_*` artifacts. Assert
that `optimize_checks.check_round()` leaves only `.csv` files under `profile/`
and removes the workspace-root profiler artifacts.

- [ ] **Step 2: Add a failing optimize-session teardown regression**

Create an optimize session workspace with an `opt-round-*` directory containing
conventional `profile/` artifacts and workspace-root `PROF_*` / `OPPROF_*`
artifacts. Assert that `OptimizeSessionArtifactsManager.cleanup_checked_session()`
performs the same pruning even without `submit-round`.

- [ ] **Step 3: Run the focused tests to confirm failure first**

Run:
`uv run python -m unittest tests.test_optimize_profile_cleanup -v`

Expected: FAIL on the new cleanup assertions before implementation.

### Task 3: Implement Shared Cleanup Logic

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
- Create: `src/helix/optimize/profile_cleanup.py`
- Modify: `src/helix/optimize/session_artifacts.py`

- [ ] **Step 1: Add profile-pruning helpers to the round skill script**

Implement recursive CSV-only pruning for round-local profile directories,
prefix-based workspace-root profiler artifact deletion, and a workspace-level
fallback sweep that reuses those helpers.

- [ ] **Step 2: Keep `submit-round` on the shared cleanup path**

Update `check_round()` so successful round validation prunes the declared
profile directory and removes root-level profiler artifacts before returning
pass.

- [ ] **Step 3: Add a thin runtime bridge for session teardown**

Expose the shared cleanup helpers through `src/helix/optimize/` using
the existing skill-loader bridge pattern.

- [ ] **Step 4: Run the same cleanup during optimize session teardown**

Invoke the shared workspace cleanup in both checked and supervised session
cleanup flows, while converting unexpected cleanup failures into warnings.

- [ ] **Step 5: Re-run the focused tests**

Run:
`uv run python -m unittest tests.test_optimize_profile_cleanup -v`

Expected: PASS

- [ ] **Step 6: Run the required strict Pyright check**

Run:
`bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/check.py`
