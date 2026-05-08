# NPUKernelBench operator workspace layout

Verbatim structure description (authoritative for this skill):

- baseline: initial version of Triton kernel (before optimization)
- opt-round-{i}: version of Triton kernel after round i
- optimize-logs: ignore
- opt-note.md: overall summary of optimization process
- NN_OperatorName.py: original PyTorch file
- NN_OperatorName.json: list of shapes tested for benchmark
- NN_OperatorName_perf.txt: benchmark result for PyTorch file, add times under raw-op-statistic-case -> "ops" for the PyTorch time.
- learned_lessons.md: short summary of learned lessons created during optimization
- triton_NN_OperatorName.py: initial Triton kernel (same as baseline)
- triton_NN_OperatorName_perf.txt: benchmark result for initial Triton version
- opt_triton_NN_OperatorName.py: optimized Triton kernel, not the same as baseline.
- opt_triton_NN_OperatorName_perf.txt: benchmark result for optimized Triton version

Inside each opt-round-{i}:

- attempts.md: description of current round attempts
- opt_triton_NN_OperatorName.py: version of Triton kernel after optimization
- opt_triton_NN_OperatorName_perf.txt: benchmark result for after optimization (name may be slightly different)
- round-state.json: brief info about status after this round.
- summary.md: more detailed summary after this round.
- logs/compare-perf.txt: comparison between performance results.

## Clarifications used by agents (does not replace the list above)

- Treat `optimize-logs/` as noisy automation output unless the user explicitly asks to inspect it.
- Top-level `triton_NN_OperatorName.py` should match the canonical snapshot under `baseline/` for a well-formed tree.
- `opt-round-{i}/` may also contain extra evidence from deeper analysis (for example `perf-analysis.md`, `profile/`, `ir/`); those are not listed above but are still valid to read when present.
- `logs/compare-perf.txt` may be absent. When it is missing, infer comparisons from `summary.md`, `attempts.md`, `round-state.json`, paired `*_perf.txt` files (baseline, round-local, and top-level `triton_*` / `opt_triton_*`), and `opt-note.md`, and state uncertainty explicitly instead of inventing a single canonical speedup table.

## PyTorch timing lines in `NN_OperatorName_perf.txt`

Benchmark exports interleave human-readable lines with machine-oriented comments. PyTorch reference timings for each shape case appear in comment lines shaped like:

`# raw-op-statistic-case-<k>: {"ops":[...]}`

Parse the JSON on that line: each op entry may include timing fields such as `avg_time_us`. Treat those values as the PyTorch-side raw op statistic evidence for case `k`, not the Triton kernel timing.
