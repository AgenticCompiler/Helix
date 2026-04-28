# Optimization Artifact Contract

## Workspace Layout

Assume the operator workspace is the directory that contains the input operator file.

Expected long-lived artifacts:

- `test_<operator-stem>.py` or the mode-specific generated correctness artifact
- `bench_<operator-stem>.py` or the mode-specific generated benchmark artifact
- `baseline/`
- `baseline/state.json`
- `baseline/perf.txt`
- `opt-note.md`
- `learned_lessons.md`
- `opt-round-1/`
- `opt-round-2/`
- ...

## Top-Level Session Note

`opt-note.md` is the top-level ledger for completed round entries and one final `## Overall Summary`.

Do not write session-start diagnosis or tentative bottleneck narrative in `opt-note.md`.

For round 1, record the starting hypothesis in `opt-round-1/attempts.md`. For later rounds, keep the initial hypothesis in that round's `attempts.md`.

## Baseline Directory

The canonical optimize baseline lives under `baseline/`.

Required baseline artifacts:

- one baseline operator snapshot under `baseline/`
- `baseline/perf.txt`
- `baseline/state.json`

`baseline/state.json` must contain these fields:

- `baseline_kind`
- `source_operator`
- `baseline_operator`
- `test_file`
- `test_mode`
- `bench_file`
- `bench_mode`
- `perf_artifact`
- `correctness_status`
- `benchmark_status`
- `baseline_established`

Set `baseline_established` to `true` only after correctness passed, benchmark passed, and the canonical baseline artifacts are in place.

Treat these state fields as the authoritative artifact references for baseline validation:

- `baseline_operator`
- `perf_artifact`

That means the checker should first verify the paths declared in `baseline/state.json` instead of guessing from default filenames. Legacy directory scanning is only a fallback when `baseline/state.json` is missing or invalid.

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
  opt_<original-operator>.py
  attempts.md
  summary.md
  opt_<original-operator>_perf.txt
  perf-analysis.md
  profile/
  ir/
  logs/
```

Keep the layout simple. Do not create unnecessary nested documentation.

Each completed round must also include `round-state.json`.

`round-state.json` must contain these fields:

- `round`
- `parent_round`
- `hypothesis`
- `evidence_sources`
- `correctness_status`
- `benchmark_status`
- `perf_artifact`
- `canonical_baseline`
- `comparison_target`
- `perf_summary_source`
- `summary_path`
- `opt_note_updated`
- `next_recommendation`

Treat these round-state fields as the authoritative artifact references for round validation:

- `summary_path`
- `perf_artifact`
- `perf_analysis_path` when present
- `profile_dir` when present
- `ir_dir` when present

That means the checker should first verify the paths declared in `round-state.json` instead of guessing from default filenames. Legacy directory scanning is only a fallback when `round-state.json` is missing or invalid.

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
- `Primary analysis level`
- `Supporting evidence`
- final analysis level
- why that hypothesis looked plausible
- what evidence motivated the round
- which evidence actually decided the round outcome
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
- `Primary analysis level`
- `Supporting evidence`
- the current analysis level
- why the hypothesis may help
- what evidence supports the round direction
- why the round stayed at that level or why it escalated deeper
- why profiling or IR capture was skipped, when those tools were not used
- any reused deeper evidence path when the round starts below pattern triage
- each meaningful code change or optimization attempt
- correctness failures and how they were repaired
- benchmark outcomes, including regressions or inconclusive results
- decisions to continue, pivot, or abandon the current round idea

Keep entries chronological so another engineer can reconstruct how the round evolved.

## Bench Evidence

- Preserve the normalized benchmark result from `run-bench`.
- For a valid round, the round-local perf artifact must be the generated `opt_<original-operator>_perf.txt`.
- That `opt_<original-operator>_perf.txt` file must be produced by the `triton-npu-run-eval` skill's `run-bench` flow for the round operator, not handwritten, post-processed, or substituted from another workflow.
- Keep enough evidence to reconstruct the performance claim without rerunning the entire round immediately.
- Keep `opt_<original-operator>_perf.txt` even when profiler or IR evidence also exists; richer evidence does not replace the round's basic benchmark summary.

## Deep Analysis Evidence

- When a round needs deeper diagnosis, prefer writing a standalone `perf-analysis.md` in the round directory.
- Treat `perf-analysis.md` as optional in the first iteration of this workflow, not a required artifact for every round.
- When `round-state.json` declares `perf_analysis_path`, that path becomes the authoritative location for the analysis file.
- Treat the content and structure of `perf-analysis.md` as owned by `triton-npu-analyze-round-performance`, not by the optimize workflow contract.

## Profile And IR Evidence

- When profiling is needed for a round decision, keep the resulting profiler artifacts under `opt-round-N/profile/`.
- When IR capture is needed for a round decision, keep the resulting IR directory under `opt-round-N/ir/`.
- A standard round-local IR workflow looks like:
  ```bash
  python3 ../triton-npu-analyze-ir/scripts/capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
  python3 ../triton-npu-analyze-ir/scripts/inspect_ir.py find-changes --ir-dir opt-round-N/ir --limit 20
  ```
- Prefer preserving profiler and IR evidence inside the round that motivated it instead of scattering those artifacts at the workspace top level.
- If the same evidence informs multiple later rounds, mention the reused path explicitly in `attempts.md` or `summary.md` rather than copying large directories repeatedly.

## Original Operator

- Treat the original input file as round 0.
- Never edit the original file in place during optimization rounds.
- Use copied operator files inside round directories for experimental edits.

## Learned Lessons Log

`learned_lessons.md` is a top-level strict reusable optimization-knowledge log for the whole optimization session.

Only add an entry when it is evidence-backed, portable to related Triton Ascend NPU operators, and written as a reusable rule, diagnostic mapping, or optimization heuristic. Each entry should state where the lesson applies or what limits it.

Use it for durable notes that should survive beyond one round, such as:

- profile-to-optimization mappings
- IR-to-code-change mappings
- compiler error repairs that reveal recurring Triton or Ascend NPU constraints
- new Triton code patterns that suggest an optimization opportunity
- validated heuristics that are not yet documented in the pattern library

Round-local command failures, failed guesses, file names, shape-specific details, temporary troubleshooting notes, and narrative summaries of what happened in one round belong in `attempts.md`, `summary.md`, or `opt-note.md`, not in `learned_lessons.md`.
