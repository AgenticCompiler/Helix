# Compare Perf Auto Metric Source Consistency

## Summary

Fix `compare-perf --metric-source auto` so each compared latency case uses one shared metric basis across the baseline and compare artifacts.

## Problem

Today `auto` mode decides the baseline-side metric source per case, but the required compare-side parse can still resolve the same case through a different source. That allows an invalid comparison where:

- baseline uses kernel latency
- compare uses total-op fallback
- the command still reports `Metric source: kernel` or `PASS`

This is incorrect because one delta is being computed from mismatched semantics.

## User-Visible Behavior

- In `--metric-source auto`, a compared latency case must resolve to exactly one shared metric basis:
  - `kernel` when both sides have comparable kernel latency
  - `total-op` when either side lacks kernel latency but both sides provide total-op timing
- `mixed` remains valid only across different cases in the same command, not within a single case.
- If a case cannot resolve to one shared basis under `auto`, that case must fail or be skipped under `--skip-latency-errors`, just like other metric-source-specific comparison errors.

## Implementation Notes

- Keep `kernel` and `total-op` explicit modes unchanged.
- Preserve the existing baseline-first required-id matching and comparison-mode metadata flow where possible.
- Add one shared normalization step for auto-mode comparison so paired entries are reinterpreted consistently before delta computation and metric-source reporting.

## Verification

- Add a regression test covering text perf files where the baseline has kernel latency but the compare file only has total-op fallback for the same latency id.
- Keep the existing JSONL regression coverage that preserves total-op fallback when the baseline already resolved to total-op.
- Run targeted compare-perf tests plus the repository verification commands relevant to the touched files.
