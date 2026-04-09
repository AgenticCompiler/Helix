# `compare-perf` Aggregate Metrics Design

## Summary

- Extend `compare-perf` so it reports the same aggregate performance metrics as `optimize-status`.
- Keep the existing per-latency comparison lines and compatibility rules unchanged.
- Clarify in the `operator-eval` skill when `compare-perf` should be used and how to read its output.

## Behavior

- Continue parsing the baseline perf file as the strict source of required `latency-*` ids.
- Continue parsing the compare-side perf file by extracting only the baseline-required latency ids and ignoring extra fields.
- After printing the per-latency comparison lines, also print:
  - `Avg improvement`
  - `Geomean speedup`
  - `Total speedup`
- Keep the command exit behavior unchanged: return `0` on success and `1` on malformed input or missing required ids.

## Metric Definitions

- `Avg improvement = mean((baseline_i - compare_i) / baseline_i)`
- `Geomean speedup = geomean(baseline_i / compare_i)`
- `Total speedup = sum(baseline_i) / sum(compare_i)`

These definitions intentionally match the metrics used by `optimize-status` so CLI commands and optimize records use the same language.

## Documentation

- Update the `operator-eval` skill with a dedicated `compare-perf` section that explains when to use it:
  - comparing baseline vs candidate perf artifacts after a benchmark run
  - checking whether an optimization changed individual cases or overall speed
- Update the public compare-perf docs so the aggregate metrics are documented alongside the per-case delta output.
