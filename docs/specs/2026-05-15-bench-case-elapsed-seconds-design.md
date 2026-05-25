# Bench Case Elapsed Seconds Perf Metadata Design

## Summary

`run-bench` perf artifacts currently report per-case latency and supporting comments, but they do not show how long each benchmark case took to execute end-to-end. Add a per-case elapsed-seconds metadata comment so both `msprof` and `standalone` runs expose wall-clock case duration without changing perf comparison semantics.

## Goals

- Record one wall-clock elapsed-time value for every benchmark case written to a perf artifact.
- Use the same metadata shape for `msprof` and `standalone` benchmark modes.
- Preserve the existing `latency-*` perf contract and `compare-perf` behavior.
- Record elapsed time for failed cases as well as successful cases whenever case execution started.

## Non-Goals

- Do not use elapsed-time metadata as a comparison metric in `compare-perf` or optimize status.
- Do not change the unit or meaning of existing `latency-*` values.
- Do not add new CLI flags or benchmark metadata requirements.

## Decision

- Add one comment line per case:
  - `# elapsed-seconds-<case-id>: <float>`
- Render the elapsed value with fixed 6-decimal precision in seconds so it is easy to read and clearly distinct from microsecond-scale `latency-*` values.
- Keep the new line in the per-case comment block immediately after the primary `latency-*` line and before other diagnostic comments.
- Treat the line as informational metadata only. Existing perf parsers and comparison helpers should continue to ignore it unless they are explicitly extended in the future.

## Capture Semantics

### `msprof`

- Measure elapsed wall-clock time from immediately before launching the per-case `msprof ... --bench <N>` command until that command returns.
- Record the elapsed value whether the case succeeds, the command fails, or CSV parsing fails after the command has run.

### `standalone`

- Measure elapsed wall-clock time for each standalone case attempt around the per-case profiling execution path.
- Record the elapsed value whether the case succeeds or fails after the case profiling attempt begins.

## Implementation Shape

- Extend `PerfCaseRecord` in `skills/triton-npu-run-eval/scripts/perf_artifacts.py` with an optional `elapsed_seconds` field.
- Centralize elapsed-seconds rendering in the shared perf-artifact renderer so `msprof` and `standalone` emit the same comment format.
- In `skills/triton-npu-run-eval/scripts/bench_runner.py`, capture elapsed time inside the local and remote `msprof` per-case loops and attach it to each `PerfCaseRecord`.
- In `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`, capture per-case elapsed time in the standalone bench loop and attach it to each `PerfCaseRecord`.

## Verification

- Add unit coverage proving local `msprof` perf output includes `# elapsed-seconds-case-*` for successful cases.
- Add unit coverage proving local `msprof` perf output includes `# elapsed-seconds-case-*` for failed cases.
- Add unit coverage proving standalone perf output includes `# elapsed-seconds-<case-id>` lines.
- Re-run existing `compare-perf` tests to confirm the new comment metadata remains ignored by comparison parsing.
