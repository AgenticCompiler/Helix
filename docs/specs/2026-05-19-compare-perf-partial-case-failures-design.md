# Compare Perf Partial Case Failures

## Summary

Adjust `compare-perf` so per-case latency errors can be skipped explicitly instead of always aborting or always skipping.

## User-Visible Behavior

- Keep the existing `latency-<id>: NA` to total-op fallback behavior unchanged.
- By default, a non-recoverable `# latency-error-<id>:` marker still fails the comparison immediately.
- When `--skip-latency-errors` is passed, skip those cases from comparison output and aggregate metrics instead of aborting immediately.
- Continue comparing all remaining valid latency ids.
- If `--skip-latency-errors` was used and any case was skipped, print a final failure summary that lists the skipped cases and return exit code `1`.
- Aggregate metrics (`Avg improvement`, `Geomean speedup`, `Total speedup`, and `Metric source`) must be computed only from successfully compared cases when skipping is enabled.

## Non-Goals

- Do not change case-id matching rules.
- Do not change how missing kernel matches are treated today.
- Do not change the `latency-<id>: NA` total-op fallback rule.
