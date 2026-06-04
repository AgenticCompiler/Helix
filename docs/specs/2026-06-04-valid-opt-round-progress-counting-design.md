---
title: Valid Opt Round Progress Counting
created: 2026-06-04
summary: Count only completed round artifacts toward optimize round progress so precreated opt-round directories do not satisfy min-round requirements.
---

# Valid Opt Round Progress Counting

## Summary

- Treat `opt-round-N/` as a completed round only when it satisfies the minimum round artifact contract.
- Do not count precreated or partial `opt-round-*` directories toward `check-round --min-rounds`, checked/supervised round gating, or continuous-mode resume progress.
- Reuse the existing round-contract skill logic as the source of truth instead of introducing a second independent notion of round validity.

## Problem

The optimize workflow currently treats the presence of `opt-round-*` directories as round progress. That is too loose for real sessions because the worker may create the next round directory before it has produced a valid optimized operator, perf artifact, or `round-state.json`.

This creates two user-visible errors:

- `check-round --min-rounds` can report that more rounds are complete than actually exist.
- CLI round loops can think progress happened when only an empty or partial directory was created.

## Design

### Valid completed round

Count a round only when the directory has the minimum completed-round package:

- numeric `opt-round-N` directory name
- `attempts.md`
- `summary.md`
- `round-state.json`
- resolved round operator artifact
- resolved round perf artifact
- `round-state.json` records:
  - `correctness_status == "passed"`
  - `benchmark_status == "passed"`

This is intentionally narrower than "directory exists" and intentionally lighter than rerunning the full `check-round` pass logic. The goal is to exclude precreated and obviously incomplete rounds without changing pass-time warnings or other advisory behavior.

### Shared helper

- Add one shared helper in the round submit contract script to answer:
  - whether one round directory counts as a completed round for progress accounting
  - which round directories in a workspace count
- Make both the skill-side `check-round` min-round summary and the CLI runtime round counters consume that helper.

## Non-Goals

- Do not change the main `check-round` decision semantics.
- Do not require `perf-analysis.md`, profiler artifacts, IR artifacts, or `opt-note.md` for round counting.
- Do not redefine best-round selection or optimize-status reporting in this change.

## Verification

- Add focused tests showing that a precreated `opt-round-*` directory without completed artifacts is ignored by:
  - `check_round(..., min_rounds=...)`
  - runtime round count / latest-round helpers
