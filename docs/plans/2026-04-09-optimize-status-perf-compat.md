# Optimize Status Perf Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `optimize-status` and `compare-perf` compare against code-agent perf artifacts that include extra non-latency fields, while keeping baseline perf files strict and standard.

**Architecture:** Preserve the strict baseline parser, add a baseline-driven extraction helper for compare-side files, wire it into `compare-perf` and optimize round inspection, and lock behavior with focused unit tests before implementation.

**Tech Stack:** Python 3.11, `pathlib`, existing bench-runner helpers, optimize-status helpers, Python `unittest`

---

### Task 1: Lock Compare-Side Parsing Semantics

**Files:**
- Modify: `tests/test_bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`

- [ ] **Step 1: Write the failing test**

Add a bench-runner test where:
- baseline is standard `latency-*`
- compare file includes required latency ids plus extra fields such as `mean_ms`
- comparison succeeds and ignores the extra field

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_compare_perf_files_ignores_extra_compare_fields -v`
Expected: FAIL because the parser currently rejects non-`latency-` lines

- [ ] **Step 3: Write minimal implementation**

Add a helper that extracts only required latency ids from the compare file and ignores unrelated entries.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_bench_runner.LocalBenchRunnerTests.test_compare_perf_files_ignores_extra_compare_fields -v`
Expected: PASS

### Task 2: Lock Optimize-Status Round Compatibility

**Files:**
- Modify: `tests/test_optimize_status.py`
- Modify: `src/triton_agent/optimize/status.py`

- [ ] **Step 1: Write the failing test**

Add an optimize-status test where:
- baseline perf stays standard
- a round `perf.txt` includes required latency ids plus extra metrics
- the round still parses and can win numeric best-round selection

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_optimize_status.OptimizeStatusTests.test_inspect_optimize_status_workspace_ignores_extra_round_perf_fields -v`
Expected: FAIL because round parsing still uses the strict parser

- [ ] **Step 3: Write minimal implementation**

Switch round parsing to the baseline-driven extraction helper while preserving existing warnings for missing or invalid required latency ids.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m unittest tests.test_optimize_status.OptimizeStatusTests.test_inspect_optimize_status_workspace_ignores_extra_round_perf_fields -v`
Expected: PASS

### Task 3: Update Docs And Verify

**Files:**
- Modify: `README.md`
- Modify: `docs/2026-04-01-compare-perf-subcommand.md`
- Modify: `docs/2026-04-07-optimize-status-subcommand.md`

- [ ] **Step 1: Update user-facing wording**

Document that baseline perf stays strict while compare-side and round-side files may contain extra ignored metrics.

- [ ] **Step 2: Run focused verification**

Run: `uv run python -m unittest tests.test_bench_runner tests.test_optimize_status -v`
Expected: PASS

- [ ] **Step 3: Run repo verification**

Run: `uv run --group dev ruff check`
Expected: PASS

Run: `uv run pyright`
Expected: PASS

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
