# `compare-perf`

Use the `compare-perf` MCP tool after you already have two perf artifacts for the same benchmark cases, typically:

- after `run-bench` on a baseline operator and an optimized operator
- during optimize workflows when you want both per-case deltas and a headline speed summary

Argument examples:

- Kernel-target optimize rounds: `compare-perf(baseline="baseline/<operator>_perf.txt", compare="opt-round-N/opt_<operator>_perf.txt", metric_source="kernel")`
- Operator-target optimize rounds: `compare-perf(baseline="baseline/<operator>_perf.txt", compare="opt-round-N/opt_<operator>_perf.txt", metric_source="all")`
- Record `effective_metric_source: total-op` for the canonical round conclusion in operator-target workflows.

Rules:

- Keep the baseline file in the standard `latency-<id>: <float>` format.
- The compare-side file may include extra summary lines such as `mean_ms: ...`; the helper ignores them unless they replace a required latency entry.
- `metric_source` selects how `compare-perf` derives each case's timing: `auto` preserves the current kernel-first fallback behavior, `kernel` requires kernel latency, `total-op` requires raw op statistics for total-op aggregation, and `all` prints both kernel and total-op comparison sections.
- By default, non-recoverable `# latency-error-<id>:` markers fail the comparison immediately. Set `skip_latency_errors=True` to keep comparing valid cases and report skipped-case errors at the end.
- The command prints per-case deltas plus `Avg improvement` and `Geomean speedup`.
- During optimize workflows, treat this command as the authority for claimed benchmark deltas and speedups.
- For kernel-target optimize rounds, prefer the kernel-oriented view, but record the resolved `effective_metric_source` when fallback changes the real basis.
- For operator-target optimize rounds, use `metric_source="all"` so both kernel and total-op views are visible, then treat the total-op section as the canonical round conclusion.
