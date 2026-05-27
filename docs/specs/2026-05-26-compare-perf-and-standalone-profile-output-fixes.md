# Compare Perf And Standalone Profile Output Fixes

## Summary

Fix two behavior gaps in local benchmark tooling:

- `compare-perf --skip-latency-errors` should treat non-positive numeric timings as skippable comparison errors instead of silently collapsing aggregate metrics to `unknown`.
- Local standalone `run-bench` paths should consistently honor `TRITON_AGENT_BENCH_OUTPUT_DIR`, including parallel standalone case execution.

## Goals

- Keep `compare-perf` aggregate metrics meaningful when only a subset of cases is invalid.
- Surface invalid timing cases as explicit skipped errors when skip mode is enabled.
- Make local standalone profiler artifact retention consistent across serial `run-bench` and parallel `run-bench`.

## Non-Goals

- Do not change remote benchmark artifact handling.
- Do not redesign compare-perf case-id matching or metric-source selection.
- Do not change successful aggregate formulas for valid positive timings.

## Decision

### Compare-perf invalid numeric timings

- Treat any compared case with `baseline <= 0` or `compare <= 0` as an invalid comparison case.
- Without `--skip-latency-errors`, fail immediately with a clear per-case error.
- With `--skip-latency-errors`, move that case into the skipped-error summary and continue aggregating over the remaining valid cases.
- If all cases are skipped, keep aggregate metrics as `unknown` and return failure with the skipped summary.

### Standalone local run-bench output root

- `TRITON_AGENT_BENCH_OUTPUT_DIR` remains the single local profiler retention control for benchmark profiling.
- Resolve configured local profile output roots to absolute paths before creating preserved run directories or passing them to standalone case subprocesses.
- Local parallel standalone `run-bench` case subprocesses should inherit the configured environment variable so each isolated runtime keeps artifacts under the same preserved root.
- When the variable is unset, preserve existing temporary-directory cleanup behavior.

## Verification

- Add `compare-perf` coverage for non-positive numeric cases with and without skip mode.
- Add parallel standalone `run-bench` coverage that confirms subprocess environment propagation.
