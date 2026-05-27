# Standalone Operator Details Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant `Count` handling and zero-duration standalone operator rows while preserving real latency semantics.

**Architecture:** Tighten `_read_profiler_metrics()` in the standalone runtime so it consumes the real `operator_details.csv` schema, filters non-contributing zero-duration rows, and retains a zero-valued row only when it corresponds to a resolved kernel. Lock the behavior in with focused standalone runtime tests before changing the implementation.

**Tech Stack:** Python, `unittest`, standalone run-eval skill scripts

---

### Task 1: Document The Behavior Change

**Files:**
- Create: `docs/specs/2026-05-27-standalone-operator-details-cleanup-design.md`
- Create: `docs/plans/2026-05-27-standalone-operator-details-cleanup.md`

- [ ] **Step 1: Write the design doc**

Capture the real standalone CSV schema, the `Count`-branch removal, the zero-duration filtering rule, and the zero-only kernel safeguard.

- [ ] **Step 2: Write the implementation plan**

Record the exact test-first sequence and required verification commands.

### Task 2: Add Failing Regression Tests

**Files:**
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] **Step 1: Add a failing test for zero-duration non-kernel filtering**

Build a temporary `operator_details.csv` fixture with mixed zero and non-zero rows, call `_read_profiler_metrics()`, and assert the resulting `ops` omits zero-duration wrapper rows while preserving positive rows and aggregate totals.

- [ ] **Step 2: Run the focused test to confirm it fails**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime.StandaloneBenchRuntimeTests.test_read_profiler_metrics_filters_zero_duration_non_kernel_rows -v`

Expected: FAIL because `_read_profiler_metrics()` currently keeps zero-duration rows in `ops`.

- [ ] **Step 3: Add a failing test for zero-duration resolved-kernel preservation**

Build a temporary `operator_details.csv` fixture whose resolved kernel rows are all zero-duration and assert `_read_profiler_metrics()` still returns that kernel row with `avg_time_us == 0.0`.

- [ ] **Step 4: Run the focused preservation test to confirm current behavior or tighten expectations**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime.StandaloneBenchRuntimeTests.test_read_profiler_metrics_preserves_zero_duration_resolved_kernel_rows -v`

Expected: PASS or targeted adjustment if the fixture reveals a gap while writing the implementation.

### Task 3: Implement The Parser Cleanup

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Test: `tests/test_standalone_bench_runtime.py`

- [ ] **Step 1: Remove `Count`-column handling from `_read_profiler_metrics()`**

Delete the optional fieldname branch and keep `active_count` only for average normalization.

- [ ] **Step 2: Filter zero-duration rows with resolved-kernel fallback**

Retain positive-duration rows by default, then re-introduce zero-valued rows only for resolved kernel names that would otherwise disappear completely.

- [ ] **Step 3: Run focused standalone runtime coverage**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime -v`

- [ ] **Step 4: Run required strict pyright**

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
