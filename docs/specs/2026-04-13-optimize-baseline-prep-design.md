# Optimize Baseline Preparation Design

## Summary

- Add an explicit baseline-preparation phase at the start of `optimize`.
- Let the optimize agent generate missing validation harnesses and do minimal repair work until the workspace has one usable baseline.
- Persist the established baseline as durable workspace artifacts so every optimize round can compare against the same canonical baseline through `compare-perf`.
- Separate session-level baseline identity from round-level parent selection.

## Problem

- The optimize workflow currently assumes the original operator can act as `round 0`, but the workspace may be missing tests, missing benchmarks, missing baseline perf data, or the original operator may require minimal repair before it can be validated.
- When that happens, the first successful optimized round may accidentally become the de facto baseline.
- That collapses two different concepts:
  - the canonical baseline for the optimize session
  - the parent candidate for one specific optimization round

## Goals

- Make baseline establishment an explicit phase of `optimize`.
- Allow optimize to generate missing test and benchmark harnesses before the first optimization round.
- Allow optimize to do the minimum necessary repair work to establish a runnable, benchmarkable baseline.
- Persist baseline artifacts in stable files so all later rounds compare against the same source of truth.
- Keep `parent_round` and canonical baseline tracking separate.

## Non-Goals

- Do not add a separate top-level CLI command just for baseline preparation.
- Do not treat baseline repair as an optimization round.
- Do not let later optimize rounds rewrite the canonical baseline.

## Baseline Artifacts

Persist the baseline in a stable workspace directory:

- `baseline/state.json`
- `baseline/perf.txt`
- `baseline/<operator-filename>`

### `baseline/state.json`

Required fields:

- `baseline_kind`
- `source_operator`
- `baseline_operator`
- `test_file`
- `test_mode`
- `bench_file`
- `bench_mode`
- `perf_artifact`
- `correctness_status`
- `benchmark_status`
- `baseline_established`

## Canonical Comparison

For any completed round, the headline metrics must come from:

```bash
compare-perf --baseline baseline/perf.txt --compare <round perf artifact>
```

Round-local parent comparison is still allowed for diagnosis, but it does not replace canonical baseline comparison.

## Gate And Contract Changes

- No optimize round may pass unless `baseline/state.json`, `baseline/perf.txt`, and the baseline operator snapshot exist and describe a passed baseline.
- Round metadata should explicitly distinguish:
  - `parent_round`
  - `canonical_baseline`
  - `comparison_target_path`

## Expected Outcome

- Optimize sessions can start from incomplete or slightly broken workspaces without corrupting round semantics.
- Every optimize session has one stable, auditable baseline that all canonical `compare-perf` results refer back to.
