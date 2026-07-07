# Optimize Round State Simplify Design

## Summary

Remove redundant fields from the optimize round state, performance status model, and
skill contracts.  All removed fields were either constant sentinels or computed
derivatives that duplicated information already available from other sources.

## Motivation

### `canonical_baseline` and `perf_summary_source` in RoundState

The round gate evaluation (`evaluate_round_gate`) checked these fields against
constant sentinel values:

- `canonical_baseline != "baseline"` always fails — there is exactly one baseline
- `perf_summary_source != "compare-perf"` always fails — `compare-perf` is the only
  authority for benchmark deltas

These fields required every `round-state.json` to include them, yet they never
carried a decision-relevant value other than their canonical constant.  Removing
them eliminates boilerplate without weakening the gate.

### `baseline_mean` and `best_mean` in OptimizeStatusWorkspace

These raw latency means were computed for the overall summary in `opt-note.md`
but never consumed by downstream tooling.  The overall summary should highlight
speedup ratios (`Geomean speedup`, `Avg improvement`), not raw latencies.

### `total_speedup` (fully removed)

`total_speedup = sum(baseline) / sum(compare)` was computed alongside
`geomean_speedup` and used as a secondary sort key in best-round selection and
as a second delta in verification consistency checks.

The May 2026 optimize-target design (`docs/specs/2026-05-20-optimize-target-design.md`)
had already called for `Total speedup` to stop being a required display field.
This change completes that direction by removing the computation entirely.

Best-round selection now uses `(geomean_speedup, -mean_latency)`.
Verification consistency relies solely on `geomean_speedup_delta`.

## Changes

### RoundState (optimize_check_contract.py)

| Removed field | Rationale |
|---|---|
| `canonical_baseline: str` | Always `"baseline"`; gate check is a no-op |
| `perf_summary_source: str` | Always `"compare-perf"`; gate check is a no-op |

- `contract.json`: removed both from `round_state_required_fields`
- `artifacts.md`: removed both from required field list
- Gate `comparison_target_path` check: changed from hardcoded string (`!= "baseline/perf.txt"`) to dynamic validation against `baseline/state.json` `perf_artifact`. This allows both the legacy `baseline/perf.txt` and the new `baseline/<operator>_perf.txt` conventions.

### OptimizeStatusWorkspace (models.py)

| Removed field | Rationale |
|---|---|
| `baseline_mean: float \| None` | Raw latency; unused downstream |
| `best_mean: float \| None` | Raw latency; unused downstream |
| `total_speedup: float \| None` | `geomean_speedup` suffices |
| `verified_total_speedup: float \| None` | `verified_geomean_speedup` suffices |

### OptimizeStatusRound (models.py)

| Removed field | Rationale |
|---|---|
| `total_speedup: float` | Redundant with `geomean_speedup` |

### Computation

- `_summarize_perf_metrics` (perf_artifacts.py): returns `(avg_improvement, geomean_speedup)` instead of 3-tuple
- `compare-perf` output: prints `Avg improvement` and `Geomean speedup` only
- `best_round` selection key: `(geomean_speedup, -mean_latency)` instead of `(geomean_speedup, total_speedup, -mean_latency)`
- Verification `_build_speedup_state`: stops computing `total_speedup`
- Verification `_build_consistency_state`: `decision_deltas = (geomean_delta,)` instead of `(geomean_delta, total_delta)`

### Skills and references

- `opt-note-format.md`: overall summary template removed `Baseline mean`, `Best mean`, and `Total speedup`; removed the "Keep Total speedup available…" guidance line
- `artifacts.md`: baseline perf path changed from `baseline/perf.txt` to `baseline/<operator>_perf.txt` to match actual `run bench` output
- `compare-perf.md`: added separate command examples for kernel-target and operator-target rounds; removed `Total speedup` from output description
- `SKILL.md`: removed `Total speedup` from the authority output list; updated baseline perf path

### baseline/perf.txt naming

`artifacts.md` previously required `baseline/perf.txt`, but `run bench` actually
generates `<operator>_perf.txt`.  All references now use
`baseline/<operator>_perf.txt`.

Runtime baseline detection updated to recognize the new naming:

- `workspace_has_optimize_artifacts` checks `baseline/*_perf.txt` instead of only `baseline/perf.txt`
- `select_baseline_perf_file` searches `baseline/*_perf.txt` before falling back to top-level files; a single match is accepted, multiple matches produce a warning
- `prepare-optimize-baseline/SKILL.md` updated to write `baseline/<operator>_perf.txt`
