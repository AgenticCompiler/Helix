# Compare Perf Metric Source Selection

## Summary

Add an explicit `compare-perf` option to choose which timing source drives comparison metrics: automatic selection, kernel-only, or total-op-only.

## User-Visible Behavior

- Add `--metric-source auto|kernel|total-op` to `compare-perf`.
- Default to `auto` to preserve current behavior.
- `auto` means:
  - prefer kernel latency when the case has a comparable kernel timing
  - fall back to total-op timing when kernel latency is unavailable but raw op statistics exist
- `kernel` means:
  - compare only kernel latency for every case
  - if a case does not have comparable kernel latency, that case is an error
- `total-op` means:
  - compare only total-op timing for every case
  - if a case does not have raw op statistics needed to compute total-op timing, that case is an error

## Interaction With Existing Flags

- `--skip-latency-errors` keeps its current meaning.
- Without `--skip-latency-errors`, any case that is invalid under the selected metric source fails the whole comparison immediately.
- With `--skip-latency-errors`, invalid cases under the selected metric source are skipped, remaining valid cases are still compared, and the command returns failure after printing the skipped-case summary.

## Output Rules

- Per-case delta lines continue to print baseline, compare, and delta.
- Aggregate metrics (`Avg improvement`, `Geomean speedup`, `Total speedup`) are computed only from cases successfully compared under the selected metric source.
- `Metric source` output should match the selected mode:
  - `auto` may still report `kernel`, `total-op`, or `mixed`
  - `kernel` reports `kernel`
  - `total-op` reports `total-op`

## Implementation Scope

- Thread the new option through:
  - repository CLI
  - `triton-npu-run-eval` skill command script
  - comparison command wrapper
  - perf parsing and comparison logic
- Add tests for parser wiring, mode forwarding, and comparison behavior in all three modes.
- Update `README.md` and the `compare-perf` skill reference.

## Non-Goals

- Do not change the archived perf file format.
- Do not change case-id matching rules.
- Do not change correctness-result comparison behavior.
