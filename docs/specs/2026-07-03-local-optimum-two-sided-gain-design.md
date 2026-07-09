# Local-Optimum Detection Should Use A Two-Sided Gain Bound

## Summary

The local-optimum check in `skills/common/ascend-npu-optimize-state/scripts/round/local_optimum.py` warns the agent when recent rounds appear to have stalled and suggests resuming from an earlier round to explore a different path. Its stall test is currently one-sided: it fires whenever every adjacent baseline-relative geomean gain is at or below a small positive threshold. Because the test only bounds the upper side, a round whose speedup *collapses* (a large negative gain) also satisfies it, so a crash or a cold-sample measurement dip is mislabeled as stagnation. This change makes the bound two-sided so only genuinely flat sequences trigger the warning.

## Background

`collect_local_optimum_warnings` looks at the most recent `window` (default 3) consecutive rounds, recomputes each round's vs-baseline geomean speedup from each round's declared perf artifact (falling back to the shared round perf resolver), and forms the adjacent gains `gain[i] = speedup[i] - speedup[i-1]`. It then emits the warning when:

```python
all(gain <= config.max_geomean_gain for gain in adjacent_gains)  # max_geomean_gain default 0.02
```

"Stagnation" means the speedup is no longer moving in either direction. The current predicate only checks that it is not moving *up*, so it treats any downward move — including a sharp collapse — as part of a stall.

Observed in `workspace/NPUKernelBench` agent logs, real emitted warnings include adjacent gains such as `-26.21x, -38.86x`, `-56.83x, -79.24x`, and `-1.58x, -13.84x`. These are performance cliffs (real compiler pessimizations or cold-sample measurement dips), not plateaus, yet each was reported as "recent rounds show only marginal ... changes ... may be stuck in a local optimum," together with the suggestion to resume from a round before the sequence. Feeding a measurement/compiler collapse into a "you have plateaued, go back to an earlier parent" recommendation is misleading guidance.

## User-Visible Semantics

- A genuinely flat sequence (every adjacent gain within a small band around zero, e.g. `+0.00x, +0.01x` or `+0.01x, -0.01x`) still triggers the local-optimum warning.
- A sequence containing a large speedup drop (e.g. `-26.21x`) no longer triggers the local-optimum warning, because such a drop is not a plateau.
- The warning still uses the same window size, threshold default, and the same two tuning environment variables (`TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_WINDOW`, `TRITON_AGENT_OPTIMIZE_LOCAL_OPTIMUM_MAX_GEOMEAN_GAIN`), but the threshold now means the maximum absolute adjacent baseline-relative geomean speedup change that still counts as nearly flat, and the warning text now describes marginal speedup changes rather than gains.
- Only the emission condition narrows; nothing else about round evaluation, parent selection, or artifact contracts changes. This check is advisory only; it does not itself pick parents or alter which round is promoted.

## Approach

Change the stall predicate from a one-sided upper bound to a two-sided magnitude bound, so both "did not rise" and "did not fall" must hold:

```python
all(abs(gain) <= config.max_geomean_gain for gain in adjacent_gains)
```

The threshold's meaning becomes "how close to zero counts as no movement"; it now bounds both directions symmetrically. No change to `LocalOptimumConfig`, the env parsing, or the window logic. The warning message builder should switch from talking about marginal speedup gains to marginal speedup changes so slightly negative flat windows are still described accurately.

## Non-Goals

- This does not address the root cause of the speedup swings themselves (cold vs. warm NPU benchmark sampling writing an occasional slow-tier `perf.txt`). That is a separate benchmarking-stability concern.
- This does not add a distinct "performance regression / collapse" warning. It only stops collapses from being misclassified as stagnation. A dedicated collapse diagnostic can be considered separately if wanted.
- No change to `status --view trend`, which reads the same round perf artifacts.

## Verification

- Unit test: a flat 3-round window (gains within the band, both slightly positive and slightly negative) still produces the warning.
- Unit test: a 3-round window containing a large negative gain (e.g. `-26x`) produces no warning.
- Unit test: a boundary case at exactly `±max_geomean_gain` still warns (inclusive bound preserved).
- Existing local-optimum tests for window/metric-basis gating and env overrides continue to pass.
- `bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/local_optimum.py` passes.
- Standard repo checks: `uv run --group dev ruff check`, `uv run pyright`, `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`.
