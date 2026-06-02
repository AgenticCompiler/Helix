# Standalone Operator Details Cleanup

## Summary

Clean up standalone `operator_details.csv` parsing so the local benchmark perf JSONL keeps only device-active operator rows and no longer carries a dead `Count`-column compatibility branch.

## Goals

- Remove redundant zero-device-duration operator rows from standalone perf JSONL `ops`.
- Preserve standalone kernel-latency and total-op aggregates for real device-active rows.
- Remove the unused `Count`-based filtering branch from standalone `operator_details.csv` parsing.
- Keep standalone parsing behavior aligned with real `torch_npu.profiler` `operator_details.csv` output.

## Non-Goals

- Do not change `active_count` normalization; repeated-profile totals must still divide by benchmark repeats.
- Do not change msprof parsing or JSONL schema.
- Do not change the standalone profile-report parser in this fix.

## Decision

### Remove `Count` filtering

- Treat standalone `operator_details.csv` as requiring only the columns that real profiler output provides today: `Name` and `Device Self Duration(us)`.
- Delete the optional `Count`-column filtering branch from `_read_profiler_metrics()`.
- Keep `active_count` only as the divisor used to normalize accumulated device self duration into per-repeat average latency.

### Drop zero-device-duration rows from standalone perf metrics

- Ignore rows whose parsed `Device Self Duration(us)` is exactly `0`.
- Aggregate only positive device self durations into the emitted `ops` list and derived totals.
- Preserve the original first-seen operator ordering among retained rows.

### Zero-only kernel safeguard

- If a resolved kernel name appears in the CSV but every matched row has zero device self duration, still emit that kernel row with `avg_time_us = 0.0` instead of dropping it completely.
- This keeps explicit zero-latency benchmark fixtures comparable and avoids turning a real zero-valued kernel into a false "kernel not matched" error.

## Verification

- Add standalone runtime coverage proving zero-duration non-kernel rows are filtered from emitted perf JSONL.
- Add standalone runtime coverage proving zero-duration resolved kernel rows remain representable as `0.0`.
- Run the focused standalone runtime unit suite.
- Run the required strict pyright check for `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`.
