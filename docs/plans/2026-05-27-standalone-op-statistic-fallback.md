# Standalone Op Statistic Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared profiler CSV parser module and make standalone benchmark metrics fall back through `kernel_details.csv` and `op_statistic.csv` when `operator_details.csv` is missing or totals to zero.

**Architecture:** Move profile-CSV parsing into a new feature-local helper module that returns reusable parsed results. Refactor msprof to consume the shared `op_statistic` parser, and refactor standalone to parse all optional standalone sources then choose one with a strict fallback priority: `operator_details`, `kernel_details`, `op_statistic`.

**Tech Stack:** Python, `unittest`, run-eval skill scripts

---

### Task 1: Document The Refactor

**Files:**
- Create: `docs/specs/2026-05-27-standalone-op-statistic-fallback-design.md`
- Create: `docs/plans/2026-05-27-standalone-op-statistic-fallback.md`

- [ ] **Step 1: Write the design doc**

Capture the new shared parser module, strict standalone fallback rule, and verification scope.

- [ ] **Step 2: Write the implementation plan**

Record the test-first sequence and touched files.

### Task 2: Add Failing Regression Coverage

**Files:**
- Modify: `tests/run_skill_test_utils.py`
- Create: `tests/test_profile_csv_parser.py`
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] **Step 1: Add a loader helper for the new parser module**

Expose a small test helper that loads the new `run-eval` parser module directly.

- [ ] **Step 2: Add a failing shared-parser test for plain `op_statistic.csv`**

Create a fixture with a non-timestamped `op_statistic.csv` and assert the parser returns normalized `ops` rows and total time.

- [ ] **Step 3: Add a failing standalone fallback test for missing `operator_details.csv`**

Create a profile directory containing only `op_statistic.csv` and assert standalone metrics are loaded from it.

- [ ] **Step 4: Add a failing standalone fallback test for zero-total `operator_details.csv` plus usable `kernel_details.csv`**

Create all relevant files, make `operator_details.csv` total to zero, and assert standalone chooses `kernel_details.csv` before `op_statistic.csv`.

- [ ] **Step 5: Add a failing shared-parser test for plain `kernel_details.csv`**

Create a fixture with non-timestamped `kernel_details.csv` and assert the parser returns normalized `ops` rows and total duration.

- [ ] **Step 6: Run the focused tests to confirm they fail for the right reason**

Run:
`uv run python -m unittest tests.test_profile_csv_parser tests.test_standalone_bench_runtime -v`

Expected: FAIL because the shared parser module and standalone fallback path do not exist yet.

### Task 3: Implement The Shared Parser And Refactor Callers

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/profile_csv_parser.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`

- [ ] **Step 1: Create the shared parser module**

Add reusable helpers for optional CSV discovery, `op_statistic` parsing, standalone `operator_details` parsing, standalone `kernel_details` parsing, and `PerfMetrics` resolution.

- [ ] **Step 2: Refactor msprof to reuse the shared `op_statistic` parser**

Replace the inline parser with the shared helper while preserving existing error semantics.

- [ ] **Step 3: Refactor standalone to use strict source fallback**

Parse `operator_details.csv` first, then `kernel_details.csv`, then `op_statistic.csv`, and keep current zero-duration filtering behavior inside `operator_details` parsing.

- [ ] **Step 4: Run focused regression tests**

Run:
`uv run python -m unittest tests.test_profile_csv_parser tests.test_standalone_bench_runtime -v`

- [ ] **Step 5: Run broader benchmark coverage**

Run:
`uv run python -m unittest tests.test_bench_runner tests.test_standalone_bench_runtime -v`

- [ ] **Step 6: Run strict pyright on touched skill scripts**

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/profile_csv_parser.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner_msprof.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
