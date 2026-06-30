# Torch NPU Profiler CSV Semantics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `torch_npu_profiler` perf metrics use one consistent kernel-view source so TileLang kernel time is preserved and `ops` stays consistent with `total_op_avg_time_us`.

**Architecture:** Keep CSV parsing feature-local in `profile_csv_parser.py`, but split operator-view parsing from kernel-view metric resolution. Standalone runtime will prefer `kernel_details.csv`, fall back to `op_statistic.csv`, and keep `operator_details.csv` only as diagnostic context. JSONL rendering will consume an explicit `total_op_avg_time_us` from the resolved metrics instead of recomputing it from mismatched sources.

**Tech Stack:** Python, `unittest`, run-eval skill scripts

---

### Task 1: Lock Down The Contract

**Files:**
- Create: `docs/specs/2026-06-29-torch-npu-profiler-csv-semantics-design.md`
- Create: `docs/plans/2026-06-29-torch-npu-profiler-csv-semantics.md`

- [ ] **Step 1: Write the design doc**

Capture the CSV roles, the kernel-view authority rule, the `total_op_avg_time_us`
definition, and the controlled `_kernel` alias behavior.

- [ ] **Step 2: Write the implementation plan**

Record the test-first sequence, touched files, and required verification
commands.

### Task 2: Add Failing Regression Coverage

**Files:**
- Modify: `tests/test_standalone_bench_runtime.py`
- Modify: `tests/test_bench_runner.py`

- [ ] **Step 1: Add a failing runtime test for kernel-view precedence**

Create a profile root where `operator_details.csv` contains only framework rows
while `kernel_details.csv` contains the real kernel, and assert profiler metrics
use the kernel rows.

- [ ] **Step 2: Add a failing runtime test for per-step total-op aggregation**

Create `kernel_details.csv` rows with multiple `Step Id` values and assert
`total_op_avg_time_us` equals the mean of per-step kernel sums.

- [ ] **Step 3: Add a failing runtime test for `_kernel` alias matching**

Declare `KernelA` in bench metadata, emit `KernelA_kernel` in the profiler CSV,
and assert `kernel_avg_time_us` resolves correctly.

- [ ] **Step 4: Add a failing JSONL rendering test for explicit total-op values**

Assert profiler-mode JSONL preserves an explicit `total_op_avg_time_us` instead
of blindly summing an unrelated or empty `ops` list.

- [ ] **Step 5: Run focused tests to confirm the new expectations fail first**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime tests.test_bench_runner -v`

Expected: FAIL on the new profiler-kernel-view assertions before implementation.

### Task 3: Implement Unified Kernel-View Metrics

**Files:**
- Modify: `skills/common/ascend-npu-run-eval/scripts/profile_csv_parser.py`
- Modify: `skills/common/ascend-npu-run-eval/scripts/bench_runtime.py`
- Modify: `skills/common/ascend-npu-run-eval/scripts/perf_artifacts.py`

- [ ] **Step 1: Extend `PerfMetrics` with explicit total-op data**

Add a `total_op_avg_time_us` field so JSONL rendering no longer recomputes
profiler totals from a potentially different source.

- [ ] **Step 2: Refactor parser helpers around kernel-view resolution**

Add helpers that aggregate kernel rows, compute per-step totals from
`kernel_details.csv`, compute aggregate totals from `op_statistic.csv`, and
apply the controlled `_kernel` alias rule.

- [ ] **Step 3: Update standalone runtime source selection**

Make `_read_profiler_metrics` prefer kernel details, then op statistics, while
retaining operator details only for logging and error context.

- [ ] **Step 4: Update perf JSONL rendering**

Render `total_op_avg_time_us` from the explicit metric field when present and
preserve existing `perf-counter` behavior.

- [ ] **Step 5: Run focused regression tests**

Run:
`uv run python -m unittest tests.test_standalone_bench_runtime tests.test_bench_runner -v`

- [ ] **Step 6: Run required strict Pyright checks**

Run:
`bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/profile_csv_parser.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/bench_runtime.py`

Run:
`bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-run-eval/scripts/perf_artifacts.py`
