# Benchmark Run Reporting Contract

The execution contract is defined in:

- [bench-standalone-run-spec.md](bench-standalone-run-spec.md)
- [bench-msprof-run-spec.md](bench-msprof-run-spec.md)

## Minimum report

- Benchmark mode
- Target operator or benchmark file
- Final status
- Primary metric if available
- Saved perf file path when data was persisted

## Persistence rule

- Persist benchmark results under `bench_results/`.
- Baseline perf files should use `old_perf-<operator>.txt`.
- Optimized perf files should use `opt_perf-<operator>-<pattern>.txt`.
- Persist one normalized `latency: <value>` line per benchmark case.

## Failure summary

Classify failures briefly:

- `environment/setup`
- `import/path`
- `compiler/runtime`
- `benchmark harness`
- `timeout`

## Comparison note

If the user compares optimized and baseline results, report both values and the implied speedup direction clearly.
