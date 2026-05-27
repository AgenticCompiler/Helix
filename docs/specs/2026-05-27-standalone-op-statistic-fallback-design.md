# Standalone Op Statistic Fallback Design

## Summary

Refactor local profiler CSV parsing into a new shared `run-eval` helper module, then make standalone benchmark metric loading prefer `operator_details.csv`, fall back to `kernel_details.csv`, and finally fall back to `op_statistic.csv` when earlier sources are missing or aggregate to zero.

## Goals

- Introduce one dedicated Python module for profiler CSV parsing under `skills/triton-npu-run-eval/scripts/`.
- Reuse the same `op_statistic` schema parser for both msprof and standalone benchmark flows.
- Add shared parsing for standalone `kernel_details.csv`.
- Keep standalone fallback behavior strict:
  - prefer `operator_details.csv`
  - fall back to `kernel_details.csv`
  - fall back to `op_statistic.csv`
  - only move to the next source when the current source is missing or has total time `0`
- Keep parsed intermediate data reusable instead of recomputing directly inside each runner.

## Non-Goals

- Do not attempt row-by-row name reconciliation between `operator_details.csv` and `op_statistic.csv`.
- Do not fall back merely because resolved kernel names do not match `operator_details.csv`.
- Do not change the perf JSONL schema.
- Do not move parsing into `src/triton_agent`; the helper should stay feature-local to `triton-npu-run-eval`.

## Decision

### New shared parser module

Create a new helper module dedicated to profiler CSV parsing. It should own:

- locating optional profiler CSV files under a profile root
- parsing `op_statistic.csv` / `op_statistic_*.csv`
- parsing standalone `operator_details.csv`
- parsing standalone `kernel_details.csv`
- returning reusable parsed results that include:
  - normalized `ops` rows
  - aggregated total time for source-selection decisions

### Shared `op_statistic` parsing

- Parse `op_statistic` with the existing schema:
  - `OP Type`
  - `Count`
  - `Total Time(us)`
  - `Avg Time(us)`
- Reuse that parser for:
  - msprof benchmark metric loading
  - standalone fallback loading

### Standalone source selection

Standalone should evaluate sources in this order:

1. Parse `operator_details.csv` if present.
2. If it exists and aggregated device-self total is greater than `0`, use it.
3. Otherwise, parse `kernel_details.csv` if present.
4. If it exists and aggregated kernel duration total is greater than `0`, use it.
5. Otherwise, parse `op_statistic.csv` if present and use it.
6. If all parsed sources have total `0`, keep the highest-priority parsed source that exists.
7. If none of the three files exists, fail clearly.

This keeps fallback strict and source-oriented, favors the more specific standalone kernel-level artifact before the coarser `op_statistic` aggregate, and avoids mixing mismatched naming granularities across sources.

## Verification

- Add standalone regression coverage for:
  - fallback when `operator_details.csv` is missing
  - fallback when `operator_details.csv` total device-self time is `0`
  - `kernel_details.csv` fallback when `operator_details.csv` is unusable
  - no fallback when `operator_details.csv` has positive total time
- Add shared parser coverage for plain `kernel_details.csv` parsing.
- Add shared parser coverage for plain `op_statistic.csv` parsing.
- Re-run msprof benchmark tests that already validate `op_statistic` consumption.
- Run strict pyright checks for the new parser module, `bench_runner_msprof.py`, and `standalone_bench_runtime.py`.
