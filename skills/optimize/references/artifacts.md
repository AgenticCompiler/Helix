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
- any small auxiliary logs or artifacts needed to understand the round outcome

Recommended layout:

```text
opt-round-N/
  <optimized-operator>.py
  attempts.md
  summary.md
  perf.txt
  logs/
```

Keep the layout simple. Do not create unnecessary nested documentation.

## Summary Requirements

`opt-round-N/summary.md` should include:

- parent round or parent candidate
- optimization hypothesis
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
- each meaningful code change or optimization attempt
- correctness failures and how they were repaired
- benchmark outcomes, including regressions or inconclusive results
- decisions to continue, pivot, or abandon the current round idea

Keep entries chronological so another engineer can reconstruct how the round evolved.

## Bench Evidence

- Preserve the normalized benchmark result from `run-bench`.
- If `run-bench` already writes a perf file under `bench_results/`, either copy the relevant result into the round directory or summarize it in `perf.txt` with a clear reference back to the source file.
- Keep enough evidence to reconstruct the performance claim without rerunning the entire round immediately.

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
