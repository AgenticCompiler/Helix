# Msprof CSV Latency Design

## Goal

- Change `bench-mode=msprof` benchmarking to measure each benchmark case from `msprof` CSV output instead of parsing terminal text.
- Keep the existing perf-file contract: each case still produces one `latency-case-<N>: <value>` line.

## User-Visible Behavior

- For each benchmark case, the runner executes:
  - local: `msprof --output=<tmp-dir> python ... --bench <N>`
  - remote: `msprof --output=<tmp-dir> python3 ... --bench <N>`
- After the command succeeds, the runner finds `op_statistic_<timestamp>.csv` under `<tmp-dir>`.
- The runner sums every row's `Avg Time(us)` value and uses that sum as the latency for that benchmark case.
- The runner deletes the temporary `msprof` output directory after parsing, in both local and remote mode.

## Error Handling

- `bench-mode=msprof` benchmark execution no longer depends on `# kernel:` metadata because it no longer uses `msprof op --kernel-name=...`.
- If the profiler output directory or `op_statistic_*.csv` is missing, fail explicitly.
- If the CSV is missing `Avg Time(us)` or contains no data rows, fail explicitly.
- Temporary profiler directories must still be cleaned up when command execution or CSV parsing fails.

## Verification

  - Add unit coverage for local `msprof` benchmarking to verify:
  - command shape uses `msprof --output=<tmp-dir>`
  - latency comes from summing `Avg Time(us)`
  - local temporary directories are removed
- Add remote unit coverage to verify:
  - remote commands create and clean a temporary profiler directory
  - latency comes from the parsed remote CSV
