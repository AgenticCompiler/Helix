# Optimization Artifact Contract

## Workspace Layout

Assume the operator workspace is the directory that contains the input operator file.

Expected long-lived artifacts:

- `test_<operator-stem>.py` or the mode-specific generated correctness artifact
- `bench_<operator-stem>.py` or the mode-specific generated benchmark artifact
- `opt-note.md`
- `learned_lessons.md`
- `opt-round-1/`
- `opt-round-2/`
- ...

## Per-Round Directory

Each completed round directory must contain:

- the optimized operator file for that round
- `attempts.md`
- `summary.md`
- performance result text copied or summarized from the benchmark run
- any profiler or IR evidence collected for that round, kept in stable round-local directories
- any small auxiliary logs needed to understand the round outcome

Recommended layout:

```text
opt-round-N/
  <optimized-operator>.py
  attempts.md
  summary.md
  perf.txt
  profile/
  ir/
  logs/
```

Keep the layout simple. Do not create unnecessary nested documentation.

Use these subdirectories consistently:

- `profile/`
  Keep profiler artifacts for the round here, for example a copied-back `PROF_*` directory or a stable local wrapper directory that contains the profiler output.
- `ir/`
  Keep archived IR capture artifacts for the round here, for example `triton_dump/`, `bishengir_stages/`, `all-ir.txt`, and `capture-manifest.json`.
- `logs/`
  Use this only for small auxiliary logs that do not justify a dedicated contract of their own.

## Summary Requirements

`opt-round-N/summary.md` should include:

- parent round or parent candidate
- optimization hypothesis
- why that hypothesis looked plausible
- what evidence motivated the round
- code changes that mattered
- correctness validation result
- benchmark comparison
- whether this round becomes the current best candidate
- follow-up ideas

The optimization points section is required. This is the part future engineers are most likely to reuse.

## Attempt Log Requirements

`opt-round-N/attempts.md` should be updated throughout the round, not only at the end.

Record at least:

- the initial round hypothesis
- why the hypothesis may help
- what evidence supports the round direction
- why profiling or IR capture was skipped, when those tools were not used
- each meaningful code change or optimization attempt
- correctness failures and how they were repaired
- benchmark outcomes, including regressions or inconclusive results
- decisions to continue, pivot, or abandon the current round idea

Keep entries chronological so another engineer can reconstruct how the round evolved.

## Bench Evidence

- Preserve the normalized benchmark result from `run-bench`.
- If `run-bench` already writes a perf file under `bench_results/`, either copy the relevant result into the round directory or summarize it in `perf.txt` with a clear reference back to the source file.
- Keep enough evidence to reconstruct the performance claim without rerunning the entire round immediately.
- Keep `perf.txt` even when profiler or IR evidence also exists; richer evidence does not replace the round's basic benchmark summary.

## Profile And IR Evidence

- When profiling is needed for a round decision, keep the resulting profiler artifacts under `opt-round-N/profile/`.
- When IR capture is needed for a round decision, keep the resulting IR directory under `opt-round-N/ir/`.
- A standard round-local IR workflow looks like:
  ```bash
  python3 ../ascend-operator-ir-analyzer/scripts/capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
  python3 ../ascend-operator-ir-analyzer/scripts/inspect_ir.py find-changes --ir-dir opt-round-N/ir --limit 20
  ```
- Prefer preserving profiler and IR evidence inside the round that motivated it instead of scattering those artifacts at the workspace top level.
- If the same evidence informs multiple later rounds, mention the reused path explicitly in `attempts.md` or `summary.md` rather than copying large directories repeatedly.

## Original Operator

- Treat the original input file as round 0.
- Never edit the original file in place during optimization rounds.
- Use copied operator files inside round directories for experimental edits.

## Learned Lessons Log

`learned_lessons.md` is a top-level reusable knowledge log for the whole optimization session.

Use it for durable notes that should survive beyond one round, such as:

- compiler error repairs that are likely to recur
- profile-to-optimization mappings
- new Triton code patterns that suggest an optimization opportunity
- validated heuristics that are not yet documented in the pattern library

Keep entries short and reusable. Prefer durable lessons over round-specific narrative.
