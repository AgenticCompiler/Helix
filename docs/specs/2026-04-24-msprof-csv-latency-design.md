# Msprof CSV Latency Design

## Goal

- Change `bench-mode=msprof` benchmarking to measure each benchmark case from `msprof` CSV output instead of parsing terminal text.
- Keep `compare-perf` keyed on the existing `latency-case-<N>` contract while allowing extra comment-only timing context in the same perf file.

## User-Visible Behavior

- For each benchmark case, the runner executes:
  - local: `msprof --output=<tmp-dir> python ... --bench <N>`
  - remote: `msprof --output=<tmp-dir> python3 ... --bench <N>`
- After the command succeeds, the runner finds `op_statistic_<timestamp>.csv` under `<tmp-dir>`.
- The runner reads the benchmark header `# kernel: <name>` and finds the `op_statistic` row whose `OP Type` exactly matches that kernel name.
- The runner uses that matched row's `Avg Time(us)` as the benchmark latency for the case.
- The runner also sums every row's `Avg Time(us)` value as the total profiled runtime for that case.
- The runner writes two lines per case to the perf artifact:
  - `latency-case-<N>: <kernel_avg_time>`
  - `# raw-op-statistic-case-<N>: {"ops":[{"op_type":"...", "avg_time_us":...}, ...]}`
- By default, the runner deletes the temporary `msprof` output directory after parsing, in both local and remote mode.
- Local `msprof` benchmarking also supports an opt-in artifact retention environment variable:
  - when `TRITON_AGENT_MSPROF_OUTPUT_DIR` is unset, behavior stays unchanged and temporary local profiler directories are deleted
  - when `TRITON_AGENT_MSPROF_OUTPUT_DIR` points to a directory, the local runner creates one preserved run directory under that location and stores each case under `case-<N>/`
  - preserved local directories are not deleted after success or failure so the raw `msprof` artifacts remain available for inspection
- Remote mode does not support artifact retention through this environment variable.
- `compare-perf` continues to compare only `latency-case-*` entries and ignores `# ...` comment lines in both baseline and compare files.

## Error Handling

- If the profiler output directory or `op_statistic_*.csv` is missing, fail explicitly.
- If the CSV is missing `Avg Time(us)` or contains no data rows, fail explicitly.
- If benchmark metadata is missing `# kernel:`, fail explicitly.
- If `op_statistic` does not contain an `OP Type` exactly matching the declared kernel name, fail explicitly.
- If `TRITON_AGENT_MSPROF_OUTPUT_DIR` points to a non-directory path, fail explicitly.
- Temporary profiler directories must still be cleaned up when command execution or CSV parsing fails.
  Preserved local artifact directories are the exception and remain on disk intentionally.

## Verification

- Add unit coverage for local `msprof` benchmarking to verify:
  - command shape uses `msprof --output=<tmp-dir>`
  - `latency-case-*` uses the matched kernel row `Avg Time(us)`
  - the perf file also includes the `# raw-op-statistic-case-*` JSON comment line
  - local temporary directories are removed
  - setting `TRITON_AGENT_MSPROF_OUTPUT_DIR` preserves local per-case output directories under the configured root
- Add remote unit coverage to verify:
  - remote commands create and clean a temporary profiler directory
  - `latency-case-*` uses the matched kernel row `Avg Time(us)`
  - the perf file also includes the `# raw-op-statistic-case-*` JSON comment line
- Add `compare-perf` coverage to verify:
  - baseline comment lines are ignored
  - compare-side comment lines are ignored
