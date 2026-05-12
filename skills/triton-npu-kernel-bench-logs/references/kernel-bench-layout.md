# NPUKernelBench operator workspace layout

**Git ignore:** In this repository, **`workspace/` is gitignored**. Repository-wide search tools that respect `.gitignore` may **hide** exports under `workspace/NPUKernelBench_level_1_2_triton/`. Use **direct path reads**, **`find workspace/…`**, or **`rg --no-ignore … workspace/…`** to list or search those files; do not infer absence from an empty git-aware search. See `skills/triton-npu-kernel-bench-logs/SKILL.md` (**Locating bench trees**).

Verbatim structure description (authoritative for this skill):

- baseline: initial version of Triton kernel (before optimization)
- opt-round-{i}: version of Triton kernel after round i
- optimize-logs: ignore
- opt-note.md: overall summary of optimization process
- NN_OperatorName.py: original PyTorch file
- NN_OperatorName.json: list of shapes tested for benchmark
- NN_OperatorName_perf.txt: benchmark result for PyTorch file, add times under raw-op-statistic-case -> "ops" for the PyTorch time.
- learned_lessons.md: short summary of learned lessons created during optimization
- **Initial vs optimized Triton naming (either convention per export):**
  - **Convention A:** `triton_NN_OperatorName.py` (initial Triton, same as baseline), `triton_NN_OperatorName_perf.txt`; `opt_triton_NN_OperatorName.py` (latest optimized snapshot at operator root), `opt_triton_NN_OperatorName_perf.txt`.
  - **Convention B:** no separate `triton_*` files; top-level **`opt_NN_OperatorName.py`** is the optimized Triton kernel (initial snapshot under **`baseline/`**). PyTorch timing export remains **`NN_OperatorName_perf.txt`** (same **`<Operator>_perf.txt`** shape as above). Triton benchmark text may appear as **`baseline/perf.txt`**, under **`opt-round-*`**, or additional `*_perf.txt` files depending on the export—use whatever exists next to **`opt_*.py`** for that round.

Inside each opt-round-{i}:

- attempts.md: description of current round attempts
- Triton after this round: **`opt_triton_NN_OperatorName.py`** and **`opt_triton_NN_OperatorName_perf.txt`**, or the same roles as **`opt_NN_OperatorName.py`** (and round-local perf if present) when the export uses convention B.
- round-state.json: brief info about status after this round.
- summary.md: more detailed summary after this round.
- logs/compare-perf.txt: comparison between performance results.

## Clarifications used by agents (does not replace the list above)

- Treat `optimize-logs/` as noisy automation output unless the user explicitly asks to inspect it.
- Under **convention A**, top-level `triton_NN_OperatorName.py` should match the canonical snapshot under `baseline/` for a well-formed tree. Under **convention B**, compare `baseline/` to **`opt_NN_OperatorName.py`** at the root and per-round **`opt_NN_OperatorName.py`** in `opt-round-*`.
- `opt-round-{i}/` may also contain extra evidence from deeper analysis (for example `perf-analysis.md`, `profile/`, `ir/`); those are not listed above but are still valid to read when present.
- `logs/compare-perf.txt` may be absent. When it is missing, infer comparisons from `summary.md`, `attempts.md`, `round-state.json`, paired perf files (baseline `perf.txt` or `*_perf.txt`, round-local exports, and top-level **`triton_*` / `opt_triton_*` or `opt_*` / `<Operator>_perf.txt`** as applicable), and `opt-note.md`, and state uncertainty explicitly instead of inventing a single canonical speedup table.

## PyTorch timing lines in `NN_OperatorName_perf.txt`

Benchmark exports interleave human-readable lines with machine-oriented comments. PyTorch reference timings for each shape case appear in comment lines shaped like:

`# raw-op-statistic-case-<k>: {"ops":[...]}`

Parse the JSON on that line: each op entry may include timing fields such as `avg_time_us`. Treat those values as the PyTorch-side raw op statistic evidence for case `k`, not the Triton kernel timing.
