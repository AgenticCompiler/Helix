# Optimize Status Speedup Design

## Summary

- Extend `optimize-status` to report both `Geomean speedup` and `Total speedup` in addition to the existing `Avg improvement`.
- Change numeric best-round selection to use `Geomean speedup` as the primary ranking metric.
- Synchronize optimize workflow documentation and `opt-note.md` summary guidance so human-readable records use the same metric vocabulary as the CLI.

## Goals

- Preserve the existing case-equal `Avg improvement` metric as a secondary signal.
- Add one benchmark-suite-style speedup metric and one whole-workload speedup metric.
- Make the CLI, optimize guidance, and `opt-note.md` overall summary agree on what metric determines the final best round.

## Non-Goals

- Do not change how baseline or round perf artifacts are discovered.
- Do not parse historical `opt-note.md` summary numbers as the source of truth.
- Do not add configurable metric-selection flags in this change.

## User-Visible Behavior

### Reported Metrics

- `Avg improvement`
  - keep the current formula:
    ```text
    mean((baseline_i - round_i) / baseline_i)
    ```
- `Geomean speedup`
  - compute:
    ```text
    speedup_i = baseline_i / round_i
    geomean_speedup = exp(mean(log(speedup_i)))
    ```
  - display as `1.23x`
- `Total speedup`
  - compute:
    ```text
    total_speedup = sum(baseline_i) / sum(round_i)
    ```
  - display as `1.23x`

### Best Round Selection

- Rank comparable rounds primarily by `Geomean speedup`.
- Use `Total speedup` as the first tiebreaker.
- Use lower `Best mean` as the final tiebreaker.

### Optimize Workflow Records

- `opt-note.md` overall summaries should record:
  - `Baseline mean`
  - `Best mean`
  - `Avg improvement`
  - `Geomean speedup`
  - `Total speedup`
- Optimize guidance should explain that the final best round is determined by `Geomean speedup`.

## Implementation Shape

- Extend `OptimizeStatusRound` and `OptimizeStatusWorkspace` with speedup fields.
- Add helpers in `src/helix/optimize/status.py` for:
  - average improvement
  - geomean speedup
  - total speedup
- Update `src/helix/optimize/render.py` to print the new metrics.
- Update optimize guidance and reference docs so `opt-note.md` and round summaries use the same terminology.

## Testing

- Unit tests for status calculation:
  - geomean speedup calculation
  - total speedup calculation
  - best-round selection by geomean instead of average improvement
- Render tests for the new output lines.
- CLI tests for plain-text `optimize-status` output including both speedup lines.
- Documentation tests or assertions where needed for updated `opt-note` guidance text.
