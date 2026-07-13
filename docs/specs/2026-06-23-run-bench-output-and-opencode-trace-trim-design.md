# Run Bench Output And Opencode Trace Trim

## Context

Two small behavior issues need tightening:

1. `run-bench --output <path>` fails when the target parent directory does not exist.
2. Opencode trace events still include `source`, `confidence`, and `duration_source`, even though those fields are not part of the desired stable trace payload.

## Design

### Perf Output Writes

Keep the change at the perf artifact write boundary so every caller that writes perf JSONL gets the same safety behavior. Before writing the file, create `path.parent` with `parents=True` and `exist_ok=True`.

This keeps `bench_runner.py` focused on selecting output paths while `perf_artifacts.py` remains responsible for making writes succeed.

### Opencode Trace Payload

Trim the extra metadata directly in `hooks/opencode/helix-hook-guard.js` so both start and end events share the same smaller schema. Keep the existing event types, status fields, timestamps, run IDs, command summaries, and durations, but drop:

- `source`
- `confidence`
- `duration_source`

## Verification

- Add a regression test proving `write_perf_lines()` creates a missing parent directory.
- Add a trace regression test proving emitted Opencode events no longer contain the removed fields.
- Run focused pytest coverage for both areas.
- Run the required skill-script strict pyright check for any modified file under `skills/*/scripts/`.
