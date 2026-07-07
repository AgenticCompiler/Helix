# Metric Source Short Alias Design

## Summary

Add `-m` as a short option alias for the existing `--metric-source` flag.

## User-Visible Behavior

- Commands that already accept `--metric-source` also accept `-m`.
- `-m` and `--metric-source` are exact aliases for the same parsed value.
- Existing defaults and validation stay unchanged:
  - `run-bench` and `compare-perf` still accept `auto|kernel|total-op|all`
  - `probe-bench` still accepts `auto|kernel|total-op` and still rejects `all`

## Scope

In scope:

- the repository CLI parser entries for `run-bench`, `probe-bench`, and `compare-perf`
- the staged `ascend-npu-run-eval` script parser entries for `run-bench` and `compare-perf`
- parser regression tests for the new short alias

Out of scope:

- changes to runtime comparison behavior
- README or skill prompt wording updates
- adding short aliases for unrelated flags

## Design

Register `-m` alongside `--metric-source` anywhere the parser already exposes
that option. Keep `dest`, defaults, and choice validation unchanged so all
downstream code continues to read `args.metric_source` without modification.

Do not mention the alias in skill guidance. This is a parser-surface
compatibility improvement only.
