# `compare-perf` Aggregate Metrics Implementation Plan

**Goal:** Add aggregate speedup metrics to `compare-perf` and document when the triton-npu-run-eval skill should use the command.

**Architecture:** Keep parsing and per-case comparison in `bench_runner.py`, then add a small aggregate summary step that reuses the same formulas already used by `optimize-status`. Update the skill and command docs to explain the new summary output without changing command shape.

**Tech Stack:** Python, unittest, Markdown docs

## Files

- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Modify: `README.md`
- Modify: `docs/2026-04-01-compare-perf-subcommand.md`

## Steps

1. Add a failing unit test that asserts `compare_perf_files()` prints `Avg improvement`, `Geomean speedup`, and `Total speedup` alongside the existing per-case output.
2. Run the focused `test_bench_runner` suite to confirm the new assertion fails for the expected reason.
3. Implement the aggregate metric calculation and summary rendering in `bench_runner.py`.
4. Re-run the focused `test_bench_runner` suite and keep iterating until it passes.
5. Update the skill and command docs so they describe when to use `compare-perf` and what the aggregate metrics mean.
6. Run focused CLI/bench tests, then run lint, type checks, and the full unittest suite.
