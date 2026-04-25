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
  If no `OP Type` exactly matches the declared kernel name, the runner records `NA` for the latency instead of failing.
- The runner also sums every row's `Avg Time(us)` value as the total profiled runtime for that case.
- The runner writes two lines per case to the perf artifact:
  - `latency-case-<N>: <kernel_avg_time>`
  - `# raw-op-statistic-case-<N>: {"ops":[{"op_type":"...", "avg_time_us":...}, ...]}`
  When the kernel row is missing, the first line becomes `latency-case-<N>: NA`.
- By default, the runner deletes the temporary `msprof` output directory after parsing, in both local and remote mode.
- Local `msprof` benchmarking also supports an opt-in artifact retention environment variable:
  - when `TRITON_AGENT_MSPROF_OUTPUT_DIR` is unset, behavior stays unchanged and temporary local profiler directories are deleted
  - when `TRITON_AGENT_MSPROF_OUTPUT_DIR` points to a directory, the local runner creates one preserved run directory under that location and stores each case under `case-<N>/`
  - preserved local directories are not deleted after success or failure so the raw `msprof` artifacts remain available for inspection
  - preserved run and case directories must be created with owner-only permissions so `msprof` does not reject them under permissive user `umask` settings
- Remote mode does not support artifact retention through this environment variable.
- `compare-perf` continues to compare only `latency-case-*` entries and ignores unrelated `# ...` comment lines in both baseline and compare files.
- When a baseline `latency-case-*` value is `NA`, comparison falls back to the matching case's total profiled runtime.
  The fallback total is derived from the `# raw-op-statistic-case-*` JSON comment in both baseline and compare files so the metric stays aligned.

## Error Handling

- If the profiler output directory or `op_statistic_*.csv` is missing, fail explicitly.
- If the CSV is missing `Avg Time(us)` or contains no data rows, fail explicitly.
- If benchmark metadata is missing `# kernel:`, fail explicitly.
- If `TRITON_AGENT_MSPROF_OUTPUT_DIR` points to a non-directory path, fail explicitly.
- If a perf file contains `latency-case-<N>: NA` but the matching `# raw-op-statistic-case-<N>` comment is missing or malformed, fail explicitly.
- If comparison needs total-op fallback for a case but the compare-side raw-op statistics are missing or malformed, fail explicitly.
- Temporary profiler directories must still be cleaned up when command execution or CSV parsing fails.
  Preserved local artifact directories are the exception and remain on disk intentionally.

## Verification

- Add unit coverage for local `msprof` benchmarking to verify:
  - command shape uses `msprof --output=<tmp-dir>`
  - `latency-case-*` uses the matched kernel row `Avg Time(us)`
  - missing kernel rows produce `latency-case-*: NA` while still recording raw-op statistics
  - the perf file also includes the `# raw-op-statistic-case-*` JSON comment line
  - local temporary directories are removed
  - setting `TRITON_AGENT_MSPROF_OUTPUT_DIR` preserves local per-case output directories under the configured root
- Add remote unit coverage to verify:
  - remote commands create and clean a temporary profiler directory
  - `latency-case-*` uses the matched kernel row `Avg Time(us)`
  - missing kernel rows produce `latency-case-*: NA` while still recording raw-op statistics
  - the perf file also includes the `# raw-op-statistic-case-*` JSON comment line
- Add `compare-perf` coverage to verify:
  - baseline comment lines are ignored
  - compare-side comment lines are ignored
  - baseline cases with `NA` fall back to total-op comparison using raw-op statistics from both sides
