# Optimize Verify Design

## Summary

Add an `optimize-verify` subcommand that validates the current best optimize round without mutating existing optimize artifacts.

The command selects the numeric best round using the same performance ranking as `optimize-status`, copies both the best-round operator and the baseline operator plus the reusable harness files into a fresh verification directory, and runs correctness plus benchmark validation from that directory.

## Goals

- Verify the final agent optimization result with explicit test and benchmark reruns.
- Reuse the existing optimize baseline contract for harness paths and modes.
- Reuse existing `run-test`, `run-bench`, and `compare-perf` execution helpers.
- Keep all new verification outputs under a new run directory.
- Never overwrite `baseline/`, `opt-round-*`, top-level result files, or earlier verification runs.

## Non-Goals

- Do not launch an agent.
- Do not create a new optimization round.
- Do not rewrite `opt-note.md`.
- Do not add batch verification in the first version.
- Do not use logged best as the default selector when it disagrees with numeric best.

## User-Visible Behavior

Run:

```bash
uv run triton-agent optimize-verify --input .
uv run triton-agent optimize-verify --input . --phase test
uv run triton-agent optimize-verify --input . --phase bench
```

`--input` must point to one operator workspace with optimize artifacts.

`--phase` accepts:

- `all`: run correctness, benchmark, and benchmark comparison.
- `test`: run only correctness.
- `bench`: run benchmark and benchmark comparison.

The command creates a fresh directory:

```text
<workspace>/opt-verify/verify-YYYYMMDD-HHMMSS/
  <operator>.py
  baseline_<operator>.py
  <test-harness>.py
  <bench-harness>.py
  test.log
  rerun-baseline-bench.log
  rerun-best-bench.log
  <operator>_result.pt
  baseline_<operator>_perf.txt
  <operator>_perf.txt
  compare-perf.txt
  verify-state.json
```

If the timestamp directory already exists, the command appends a numeric suffix. Existing verification runs are never reused or overwritten.

## Selection And Inputs

The command selects the best round by reusing the `optimize-status` numeric ranking:

1. highest geomean speedup
2. highest total speedup
3. lowest mean latency

The command reads `baseline/state.json` for:

- `baseline_operator`
- `test_file`
- `test_mode`
- `bench_file`
- `bench_mode`
- `perf_artifact`

The command reads the selected round contract to find the round-local operator. It copies that operator, the declared baseline operator snapshot, and the declared test and benchmark harnesses into the verification directory before executing anything.

## Execution

For correctness verification, call the existing test runner with:

- test file: copied test harness in the verification directory
- operator file: copied operator in the verification directory
- test mode: `baseline/state.json`, unless overridden by `--test-mode`

For benchmark verification, call the existing benchmark runner with:

- benchmark file: copied benchmark harness in the verification directory
- operator file: copied baseline operator in the verification directory
- benchmark mode: `baseline/state.json`, unless overridden by `--bench-mode`

Then rerun benchmark verification with:

- benchmark file: copied benchmark harness in the verification directory
- operator file: copied best-round operator in the verification directory
- benchmark mode: `baseline/state.json`, unless overridden by `--bench-mode`

After both rerun benchmarks succeed, run `compare-perf` using:

- baseline: perf artifact produced beside the copied baseline operator
- compare: perf artifact produced beside the copied best-round operator

Save the comparison output to `compare-perf.txt`.

## State File

Write `verify-state.json` in the verification directory as a compact verification record:

- `selection`: selected numeric best round, selected round directory, source operator path, and `numeric_best_source`.
- `workspace`: verification run directory, copied best-round operator path, and copied baseline operator path.
- `inputs`: source and copied harness paths, effective test and benchmark modes, and the historical baseline perf path from `baseline/state.json`.
- `verify-result`: execution records, fresh speedup metrics, and consistency deltas for this verification run.

`selection.optimize_status` records the speedup metrics that `optimize-status` computed when selecting the best round:

- `state`
- `baseline_mean`
- `best_mean`
- `avg_improvement`
- `geomean_speedup`
- `total_speedup`
- `warnings`

`verify-result` records:

- `test`: status, return code, log path, and result artifact path.
- `rerun_baseline_bench`: status, return code, log path, perf artifact path, and latency ids.
- `rerun_best_bench`: status, return code, log path, perf artifact path, and latency ids.
- `compare_perf`: status, return code, and comparison log path.
- `speedup`: fresh `avg_improvement`, `geomean_speedup`, `total_speedup`, and warnings computed from the rerun baseline and rerun best benchmark artifacts.
- `consistency`: status plus deltas between `verify-result.speedup` and `selection.optimize_status` for average improvement, geomean speedup, and total speedup.

`verify-result.speedup` uses `null` metric values and warnings when rerun baseline or rerun best benchmark artifacts are unavailable, such as a test-only run or a failed benchmark phase.

`verify-result.consistency.status` is `matched` when geomean speedup and total speedup deltas are both within `0.2`; `avg_improvement_delta` is recorded for diagnostics but does not decide matched versus mismatched.

## Error Handling

Fail with a concise error when:

- no numeric best round is available
- `baseline/state.json` is missing or invalid
- the selected round is missing its operator
- the declared test, benchmark, or baseline perf path is missing
- a runner fails

When a runner fails after the verification directory is created, keep the directory and write `verify-state.json` with the observed return code and available artifacts.

## Testing

- Unit test target resolution from baseline and round contracts.
- Unit test fresh verification directory creation and operator copy behavior.
- Unit test phase-specific execution with mocked runners.
- CLI parser tests for `optimize-verify`, `optimize_verify`, `--phase`, local and remote execution flags.
- CLI handler tests proving execution uses the copied operator path, not the original round operator.
