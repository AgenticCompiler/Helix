# Status Baseline State Perf Selection Design

## Summary

Change `helix status` so baseline perf selection treats
`baseline/state.json` `perf_artifact` as the authoritative baseline artifact
whenever that state entry is available.

## Goals

- Stop warning `found multiple baseline perf files` when `baseline/state.json`
  already points to the canonical baseline perf artifact.
- Make `status` follow the same baseline artifact authority that the optimize
  baseline contract already documents.
- Keep the change read-only: `status` should report invalid baseline state as a
  warning instead of rewriting files or guessing a replacement path.

## Non-Goals

- Do not change round perf selection or round validity gating.
- Do not preserve or add any special handling for `baseline/perf.txt`.
- Do not redesign top-level workspace fallback behavior beyond the baseline
  state precedence change.

## Problem

The current `select_baseline_perf_file()` logic scans `baseline/*_perf.txt`
before it checks whether `baseline/state.json` already names the canonical perf
artifact. In real workspaces, that can produce a false ambiguity warning when
multiple perf files exist in `baseline/` but only one of them is recorded in
state.

That behavior is inconsistent with the optimize artifact contract, which says
the checker should trust the paths declared in `baseline/state.json` before
guessing from directory contents.

## User-Visible Semantics

- If `baseline/state.json` loads successfully and its `perf_artifact` resolves
  to an existing file, `status` must use that file as the baseline perf input.
- In that case, `status` must not emit `found multiple baseline perf files`
  merely because other `baseline/*_perf.txt` files are present.
- If `baseline/state.json` loads successfully but the declared `perf_artifact`
  does not resolve to a file, `status` should warn about the missing declared
  baseline perf artifact and should not guess another file from `baseline/`.
- Only when `baseline/state.json` is missing or cannot provide a usable
  `perf_artifact` declaration may `status` fall back to the existing legacy
  directory scan and top-level workspace heuristics.

## Proposed Implementation

- Update `src/helix/status/core.py` baseline selection to check
  `baseline/state.json` before scanning `baseline/*_perf.txt`.
- Reuse the existing baseline-state loader so `status` stays aligned with the
  optimize baseline contract.
- Resolve the declared `perf_artifact` with the same state-relative-then-
  workspace-relative path behavior used elsewhere in the baseline checks.
- Keep the existing legacy fallback logic for workspaces that do not have a
  usable baseline state file.

## Testing

Add focused status regressions that cover:

- multiple baseline perf files where `baseline/state.json` points to the
  canonical one
- a state-declared baseline perf path that is missing while sibling
  `baseline/*_perf.txt` files still exist

These tests should prove both the authority preference and the strict
"do not guess when state declared a missing file" behavior.
