# Optimize Status Perf Compatibility Design

## Summary

- Make `optimize-status` and `compare-perf` tolerate code-agent round perf files that include extra non-latency lines such as `mean_ms: ...`.
- Keep baseline perf files strict and standard so they remain the schema source for comparisons.
- Compare round or candidate perf files by extracting only the latency ids required by the baseline and ignoring unrelated extra fields.

## Goals

- Preserve the existing standard baseline perf contract: `latency-<id>: <float>`.
- Let `optimize-status` consume `opt-round-N/perf.txt` files even when they contain extra summary metrics.
- Let `compare-perf` compare a standard baseline against a looser candidate perf file without failing on unrelated fields.

## Non-Goals

- Do not relax baseline perf parsing.
- Do not invent partial matching rules when a required baseline latency id is missing.
- Do not change round artifact discovery priority unless needed for the new extraction flow.

## User-Visible Behavior

### Baseline Files

- Baseline perf files remain strict.
- Every non-empty line must still be a valid `latency-<id>: <float>` entry.
- Malformed, duplicate, or empty baseline files still fail exactly as before.

### Compare-Side Files

- `compare-perf` first parses the baseline file strictly.
- It then reads the compare file by extracting only the baseline latency ids.
- Extra compare-side fields such as `mean_ms`, `median_ms`, or free-form notes are ignored.
- Missing required latency ids, duplicate required ids, or invalid numeric values for required ids still fail.

### Optimize Round Files

- `optimize-status` keeps the existing round artifact source order:
  1. `opt-round-N/perf.txt`
  2. `opt-round-N/*_perf.txt`
- Once a baseline perf file is known, round perf parsing extracts only the baseline latency ids from the chosen round artifact.
- Extra round-local fields are ignored.
- Missing required latency ids or invalid required values still surface as warnings and keep the round from numeric comparison.

## Implementation Shape

- Keep the existing strict perf parser for baseline use cases.
- Add a second helper that parses a file for a required set of latency ids while ignoring unrelated fields.
- Reuse that helper in:
  - `skills/triton-npu-run-eval/scripts/bench_runner.py` for `compare_perf_files`
  - `src/helix/optimize/status.py` for round perf extraction
- Keep `select_baseline_perf_file()` and `find_round_perf_file()` behavior unchanged unless tests show a discovery gap.

## Testing

- Bench-runner tests proving `compare-perf` ignores extra compare-side fields.
- Optimize-status tests proving round `perf.txt` files with extra metrics still participate in best-round selection.
- Baseline strictness remains covered by existing parser behavior and regression tests.
