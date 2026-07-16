# `compare-perf`

Use the `compare-perf` MCP tool after you already have two perf artifacts for the same benchmark cases, typically:

- after `run-bench` on a baseline operator and an optimized operator
- during optimize workflows when you want both per-case deltas and a headline speed summary

Argument examples:

- Kernel-target optimize rounds: `compare-perf(baseline="baseline/<operator>_perf.txt", compare="opt-round-N/opt_<operator>_perf.txt", metric_source="kernel")`
- Operator-target optimize rounds: `compare-perf(baseline="baseline/<operator>_perf.txt", compare="opt-round-N/opt_<operator>_perf.txt", metric_source="all")`
- Record `effective_metric_source: total-op` for the canonical round conclusion in operator-target workflows.

Rules:

- Both files must contain JSONL performance records, one JSON object per benchmark case.
- `metric_source` selects how `compare-perf` derives each case's timing: `auto` preserves the current kernel-first fallback behavior, `kernel` requires kernel latency, `total-op` requires the record's `total_op_avg_time_us`, and `all` prints both kernel and total-op comparison sections.
- Cross-mode comparison (e.g., `perf-counter` vs `msprof`) is rejected with an error. Perf-counter results support all metric sources: `kernel`, `total-op`, and `all` produce the same value; `auto` resolves to the same result.
- By default, non-recoverable JSONL `error_message` values fail the comparison immediately. Set `skip_latency_errors=True` to keep comparing valid cases and report skipped-case errors at the end.
- The command prints per-case deltas plus `Avg improvement` and `Geomean speedup`.
- During optimize workflows, treat this command as the authority for claimed benchmark deltas and speedups.
- For kernel-target optimize rounds, prefer the kernel-oriented view, but record the resolved `effective_metric_source` when fallback changes the real basis.
- For operator-target optimize rounds, use `metric_source="all"` so both kernel and total-op views are visible, then treat the total-op section as the canonical round conclusion.
