# Bench Case Support Flattening And Verbose Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flatten temporary benchmark support files into the case workspace root and surface clear verbose staging logs.

**Architecture:** Extend the case-workspace staging helpers to accept two explicit path groups: layout-preserving inputs and flattened support files. Then update standalone parallel callers to pass runtime helpers through the flattened path group and emit verbose log lines during local and remote staging.

**Tech Stack:** Python, `unittest`, bench runner skill scripts

---

### Task 1: Document And Lock In Staging Expectations

**Files:**
- Create: `docs/specs/2026-05-26-bench-case-support-flattening-and-verbose-staging-design.md`
- Create: `docs/plans/2026-05-26-bench-case-support-flattening-and-verbose-staging.md`

- [ ] **Step 1: Write the design doc**

Capture the two-bucket staging model, the non-goals, and the verification scope.

- [ ] **Step 2: Write the implementation plan**

Record the staging-helper signature change, verbose logging scope, and required tests.

### Task 2: Add Failing Regression Tests

**Files:**
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_remote_execution.py`

- [ ] **Step 1: Write failing tests for local flattened support staging**

Cover a local temporary case workspace where benchmark inputs preserve layout but support files flatten into the workspace root.

- [ ] **Step 2: Run the focused local test to confirm it fails**

Run a targeted `unittest` selector for the new bench runner test.

- [ ] **Step 3: Write failing tests for remote flattened support staging and verbose logs**

Cover remote case workspace targets plus explicit verbose messages for local and remote staging.

- [ ] **Step 4: Run the focused remote/logging tests to confirm they fail**

Run targeted `unittest` selectors for the new remote execution assertions.

### Task 3: Implement Helper Changes

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_deps.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`

- [ ] **Step 1: Extend case-workspace staging helpers**

Add separate flattened support-file parameters and verbose logging support for local and remote staging helpers.

- [ ] **Step 2: Update standalone parallel callers**

Pass runtime support files through the flattened support-file path, keep benchmark/operator/json inputs layout-preserving, and keep subprocess imports rooted at the case workspace.

- [ ] **Step 3: Verify focused tests pass**

Run targeted `unittest` selectors for the new and updated tests.

- [ ] **Step 4: Verify broader coverage**

Run:
`uv run python -m unittest tests.test_bench_runner tests.test_remote_execution -v`

- [ ] **Step 5: Verify touched skill scripts with strict pyright**

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner_standalone.py`
