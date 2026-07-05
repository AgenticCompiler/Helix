# Submit-Baseline Perf Fallback Design

## Summary

Keep `submit-baseline` validation strict, but align its fallback perf-artifact discovery with the current `run-bench` naming convention.

## Problem

Today `submit-baseline` validates the perf artifact declared in `baseline/state.json` when that file is present and valid. That path already supports both legacy `baseline/perf.txt` and current `baseline/<operator>_perf.txt`.

The mismatch appears when `baseline/state.json` is missing or invalid. In that fallback path, the baseline checker only looks for `baseline/perf.txt`. A workspace that already contains a valid `run-bench` output such as `baseline/kernel_perf.txt` still fails baseline submission even though the current optimize contract and baseline-preparation skill both allow operator-named baseline perf artifacts.

## User-Visible Behavior

- If `baseline/state.json` is present and valid, `submit-baseline` must keep treating the declared `perf_artifact` path as authoritative.
- If `baseline/state.json` is missing or invalid, fallback baseline inspection must accept either:
  - `baseline/perf.txt`
  - exactly one `baseline/*_perf.txt`
- If neither fallback artifact exists, the failure message should no longer imply that only `baseline/perf.txt` is valid.

## Non-Goals

- Do not weaken baseline gating by skipping perf-artifact validation.
- Do not change round artifact naming or comparison-target rules.
- Do not rewrite valid `baseline/state.json` contents automatically.

## Implementation Notes

- Keep the authoritative-path behavior unchanged for valid `baseline/state.json`.
- Add one shared fallback helper in the baseline checker for perf-artifact discovery.
- Prefer the explicit legacy path `baseline/perf.txt` when it exists; otherwise accept a single operator-named `*_perf.txt`.
- Update the default missing-artifact wording so the fallback contract is clear when state is missing or invalid.

## Verification

- Add a regression test covering invalid `baseline/state.json` plus a present `baseline/<operator>_perf.txt`.
- Keep the existing missing-artifact regression, but update it to match the broader failure wording.
- Run targeted optimize baseline and optimize check tests, then the repository verification commands relevant to touched files.
