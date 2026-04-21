# `optimize-status` Verified Speedups Design

## Goal

Add verified-only speedup columns to `optimize-status --format markdown`.

The existing `Geomean speedup` and `Total speedup` columns continue to show the current optimize-status numeric best metrics. The new columns show the rerun verification metrics from the latest complete successful `opt-verify/verify-*/verify-state.json`.

## Semantics

- `optimize-status` remains read-only and never runs verification.
- A workspace is verified only when the latest verify state has passed `test`, `rerun_baseline_bench`, `rerun_best_bench`, and `compare_perf`.
- Verified speedup values come from `verify-result.speedup.geomean_speedup` and `verify-result.speedup.total_speedup`.
- If the latest verify state is missing, failed, incomplete, malformed, or does not include numeric speedup values, the verified speedup cells stay blank.

## Implementation

- Extend `OptimizeStatusWorkspace` with optional verified geomean and total speedup fields.
- Parse verified speedups alongside the existing latest-verify status inspection.
- Add two Markdown columns after `Verified`: `Verified Geomean speedup` and `Verified Total speedup`.
- Keep text output unchanged except for continuing to show the latest verify state path for diagnostics.

## Tests

- Status inspection test proving verified speedups are read from a passed verify state.
- Status inspection test proving failed/latest partial verify state does not expose verified speedups.
- Markdown render and CLI tests proving new columns render values for verified workspaces and blank cells otherwise.
