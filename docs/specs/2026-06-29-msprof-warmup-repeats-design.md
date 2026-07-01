# msprof Mode Honors Warmup And Repeats

## User-Visible Semantics

- In `--bench-mode msprof`, each benchmark case must execute its kernel
  `warmup + repeats` times, not once.
- This matches the existing intent of the per-case `warmup` and `repeats`
  fields, which already drive the `torch-npu-profiler` and `perf-counter`
  modes.
- The reported metric is unchanged in shape: msprof captures the whole
  profiled subprocess and CANN's `op_statistic.csv` reports per-op
  `Avg Time(us)`, which averages across every kernel execution in that run.
- Warmup iterations are intentionally included in that average for this first
  version. Their purpose here is to amortize one-time costs (JIT compilation
  under `TRITON_ALWAYS_COMPILE=1`, first-run allocation) across the run rather
  than to be excluded from measurement. msprof has no per-iteration profiling
  window, so a single-process loop folds warmup into the CANN average by
  design.

## Problem

`run-one` is the per-case entrypoint that the msprof builders invoke through
`msprof --output=... python bench_runtime.py run-one ...`. Today
`execute_bench_case` calls `case.fn()` exactly once. So msprof measures a
single cold invocation: no warmup to settle compile/first-run cost, and no
repeat to stabilize the average. The `warmup` and `repeats` fields a bench
case declares are silently ignored in msprof mode, unlike every other mode.

## Implementation Direction

- Add an optional `--iterations N` argument to the `run-one` subcommand only.
  - Default is `1`, so every existing non-benchmark caller of `run-one`
    (IR capture, simulator, profile helpers) keeps single-shot behavior.
  - The argument is added directly on the `run-one` subparser, not on the
    shared `_add_common_case_arguments`, so `list-cases`, `profile-one`, and
    `run-all` are unaffected.
- `execute_bench_case` gains `iterations: int = 1` and runs `case.fn()` that
  many times, synchronizing after each call the same way the perf-counter and
  profiler loops do.
- Only the four msprof builders pass an explicit count, computed per case as
  `case.warmup + case.repeats`:
  - `_run_local_bench_msprof`
  - `_run_local_bench_msprof_parallel`
  - `_run_remote_bench_msprof`
  - `_run_remote_bench_msprof_parallel`
- The parallel paths resolve the per-case count by `case_id` and thread the
  resolved integer into their per-case worker helpers.
- Canonical `run-bench` semantics for `torch-npu-profiler` and `perf-counter`
  modes are unchanged. This change only affects msprof execution and the
  `run-one` default behavior (which stays single-shot).

## Verification

- Add a regression test that `execute_bench_case` invokes `case.fn()` exactly
  `iterations` times, and defaults to one call when `iterations` is omitted.
- Update msprof builder command assertions to expect
  `--iterations <warmup+repeats>` on the constructed `run-one` command.
- Run the file-scoped strict Pyright check for `bench_runtime.py` since it is a
  skill script.
