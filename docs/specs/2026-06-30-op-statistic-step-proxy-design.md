# Op Statistic Step Proxy Semantics

Superseded by `2026-06-30-op-statistic-active-count-priority-design.md` for the
current fallback policy. This note records the earlier Count-GCD-first design
that motivated the parser refactor.

## Summary

Tighten the `torch_npu_profiler` fallback semantics for `op_statistic.csv` so
they stay consistent with the kernel-view design without trusting the benchmark
`active_count` blindly.

## Problem

`kernel_details.csv` is authoritative because it carries `Step Id`, so the
parser can compute per-step totals directly.

`op_statistic.csv` does not have `Step Id`. The current fallback recomputes
per-step averages by dividing each row's `Total Time(us)` by the benchmark
`active_count`. Real profiler output proves that this is not reliable:

- benchmark `active_count` can differ from the exported active step count
- `op_statistic.csv` already contains both native per-launch `Avg Time(us)` and
  per-op `Count`

That makes `Total / active_count` silently wrong when the exported trace has a
different number of active steps.

## Decision

- Keep `kernel_details.csv` as the primary source for `torch_npu_profiler`.
- When `op_statistic.csv` is the only aggregation view:
  - treat `Count` as the best available step proxy input
  - infer an effective step count from the positive-row-count GCD
  - fall back to benchmark `active_count` only when the CSV counts do not yield
    a useful proxy
  - normalize each row to a per-step average so `ops` and
    `total_op_avg_time_us` stay in the same unit
- Remove the now-unused `operator_details.csv` parser path so the code matches
  the current design.

## Verification

- Add a regression test where benchmark `active_count=50` but
  `op_statistic.csv` counts imply `45` real active steps, and assert the
  fallback resolves the correct per-step kernel and total-op values.
- Remove the dead `operator_details.csv` parser and the runtime's unused
  discovery log.
- Re-run focused runtime tests, remote tests, `pyright`, and the full test
  suite.
