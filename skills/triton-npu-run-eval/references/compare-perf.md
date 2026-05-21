# `compare-perf`

Use `compare-perf` after you already have two perf artifacts for the same benchmark cases, typically:

- after `run-bench` on a baseline operator and an optimized operator
- during optimize workflows when you want both per-case deltas and a headline speed summary

Run:

```bash
python3 ./scripts/run-command.py compare-perf --baseline <baseline_perf.txt> --compare <candidate_perf.txt>
```

Rules:

- Keep the baseline file in the standard `latency-<id>: <float>` format.
- The compare-side file may include extra summary lines such as `mean_ms: ...`; the helper ignores them unless they replace a required latency entry.
- `--metric-source auto|kernel|total-op|all` selects how `compare-perf` derives each case's timing: `auto` preserves the current kernel-first fallback behavior, `kernel` requires kernel latency, `total-op` requires raw op statistics for total-op aggregation, and `all` prints both kernel and total-op comparison sections.
- By default, non-recoverable `# latency-error-<id>:` markers fail the comparison immediately. Add `--skip-latency-errors` to keep comparing valid cases and report skipped-case errors at the end.
- The command prints per-case deltas plus `Avg improvement`, `Geomean speedup`, and `Total speedup`.
- During optimize workflows, treat this command as the authority for claimed benchmark deltas and speedups.
- For kernel-target optimize rounds, prefer the kernel-oriented view, but record the resolved `effective_metric_source` when fallback changes the real basis.
- For operator-target optimize rounds, use `--metric-source all` so both kernel and total-op views are visible, then treat the total-op section as the canonical round conclusion.
