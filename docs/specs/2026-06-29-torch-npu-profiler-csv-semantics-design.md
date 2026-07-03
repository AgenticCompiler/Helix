# Torch NPU Profiler CSV Semantics

## User-Visible Semantics

- In `run-bench --bench-mode torch-npu-profiler`, the perf JSONL metrics must
  describe one consistent view of the run.
- `ops` must be the detailed rows that back `total_op_avg_time_us`.
- `kernel_avg_time_us` must be resolved from that same detailed view by matching
  the declared kernel names.
- For `torch_npu_profiler`, the authoritative detailed view is the kernel view:
  prefer `kernel_details.csv`, then fall back to `op_statistic.csv`.

## Problem

The current parser mixes incompatible CSV semantics:

- `operator_details.csv` is a framework/operator view and can omit the real
  TileLang kernel name entirely.
- `kernel_details.csv` is the per-kernel execution view and contains the actual
  TileLang kernel rows.
- `op_statistic.csv` is the aggregated kernel view.

When standalone profiling treats `operator_details.csv` as the main source, it
can lose kernel time even though the profiler captured the kernel correctly.
The issue is amplified when the bench metadata declares `foo_kernel` but the
profiler emits `foo_kernel_kernel`.

## Decision

- `torch_npu_profiler` main metrics are unified on the kernel view.
- `_read_profiler_metrics` must resolve metrics in this order:
  - parse `kernel_details.csv` when present and usable
  - otherwise parse `op_statistic.csv`
  - only use `operator_details.csv` for diagnostics or final error context, not
    for perf JSONL `ops`
- `ops` must always represent kernel-aggregated rows for profiler mode.
- `total_op_avg_time_us` must equal the average total kernel time per active
  step:
  - preferred source: group `kernel_details.csv` rows by `Step Id`, sum each
    step, then average those step totals
  - fallback source: `sum(Total Time(us)) / active_count` from
    `op_statistic.csv`
- Kernel matching stays strict except for one controlled alias rule:
  - try exact matches first
  - if still unmatched, allow `declared_name + "_kernel"` to match one profiler
    op name

## Verification

- Add regression tests showing `torch_npu_profiler` ignores
  `operator_details.csv` as a metrics source when `kernel_details.csv` is
  available.
- Add regression tests proving `ops` and `total_op_avg_time_us` remain
  self-consistent in rendered perf JSONL.
- Add regression tests for the `_kernel` alias behavior.
- Run focused benchmark runtime tests, broader bench-runner tests, and the
  required strict Pyright checks for modified skill scripts.
