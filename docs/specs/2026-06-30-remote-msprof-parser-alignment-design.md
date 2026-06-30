# Remote msprof Parser Alignment

## Summary

Align the remote `msprof` metric parser with the shared local parser so remote
benchmark results do not drift from local `msprof` semantics when the shared
parser evolves.

## Problem

`bench_runner.py` still embeds a standalone remote Python script for reading
remote `op_statistic.csv`. That script duplicates parsing logic instead of
reusing `profile_csv_parser.py`.

This creates two concrete risks:

- local `msprof` now gets controlled `_kernel` alias matching through
  `resolve_perf_metrics`, but remote `msprof` still uses exact matching
- future parser changes (field normalization, error handling, explicit
  `total_op_avg_time_us`) can land locally while remote keeps stale semantics

This path is specific to remote `msprof`, not remote `torch_npu_profiler`.

## Decision

- keep remote `msprof` parsing remote-side so we do not need to copy profiler
  CSVs back to the local host
- replace the embedded ad-hoc parser with a tiny remote script that imports the
  staged shared helpers:
  - `find_latest_op_statistic_csv`
  - `parse_op_statistic_csv`
  - `resolve_perf_metrics`
- return the same `PerfMetrics` shape that local parsing now uses, including an
  explicit `total_op_avg_time_us` when available

## Verification

- add a regression test that executes the embedded remote parser script against
  a temporary `op_statistic.csv`
- prove that declared kernel `KernelA` matches profiler op
  `KernelA_kernel`
- prove that the remote parser returns explicit `total_op_avg_time_us`
