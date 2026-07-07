# Status Valid Round Gating Design

## Summary

Change `triton-agent status` so round performance is counted only when the
round explicitly passed both correctness and benchmark validation.

## Goals

- Prevent stale perf artifacts from failed rounds from affecting best-round and
  trend calculations.
- Make `status` follow the same round validity rule as the optimize round
  contract.
- Keep the behavior read-only: `status` reports invalid round artifacts as
  warnings instead of mutating workspace files.

## Non-Goals

- Do not preserve legacy status compatibility for rounds that only contain
  perf artifacts.
- Do not change baseline perf selection behavior.
- Do not change verified-result handling under `opt-verify/`.

## User-Visible Behavior

- A round participates in `status` calculations only when:
  - `opt-round-N/round-state.json` exists
  - `correctness_status` is `passed`
  - `benchmark_status` is `passed`
- If a round directory exists but has no valid passed round state, `status`
  skips that round and emits a warning.
- A workspace with only invalid or legacy rounds reports no comparable round
  perf data instead of claiming a best round from leftover perf files.

## Architecture

`src/triton_agent/status/core.py` should load round state before reading any
round perf artifact. The round is comparable only when the state loads
successfully and both gate fields are `passed`. Metric-source selection should
continue to come from the same round state object once the round is accepted.

## Testing

Add focused status tests that cover:

- invalid rounds with perf artifacts are skipped
- legacy perf-only rounds are skipped
- existing best-round and trend calculations still work when valid round state
  is present
