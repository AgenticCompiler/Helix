# Op Statistic Active Count Priority

## Summary

Align the `op_statistic.csv` fallback with the current profiler exports used by
the team: when benchmark `active_count` is available, treat it as the primary
per-step divisor and use Count-derived GCD only as a diagnostic cross-check.

## Problem

`op_statistic.csv` does not carry `Step Id`, so the fallback path needs an
external step divisor.

The previous implementation inferred that divisor from the positive-row Count
GCD and only used `active_count` as a last resort. That was useful for older
exports, but the current datasets under investigation now show:

- benchmark `active_count` matches the exported active-step semantics
- Count-derived GCD matches `active_count` on the current profiler outputs
- the team wants the benchmark contract to remain authoritative for this path

## Decision

- Keep `kernel_details.csv` plus `Step Id` as the authoritative primary source.
- When falling back to `op_statistic.csv`:
  - prefer benchmark `active_count` when it is available
  - still compute the Count-derived GCD as a consistency check
  - emit a warning when inferred GCD and `active_count` differ
  - use GCD only when `active_count` is unavailable

## Verification

- Update parser-level tests so a mismatch now keeps `active_count` and emits a
  warning.
- Update runtime-level fallback tests to assert the same behavior.
- Re-run targeted unit tests, strict script pyright, and the affected runtime
  suites.
