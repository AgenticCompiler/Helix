---
name: optimize
description: Iteratively optimize a Triton Ascend NPU operator with correctness and performance gates. Use for operator optimization tasks that need repeated correctness validation, benchmark validation, multi-round experiment tracking, reusable optimization notes, and profiler-backed performance analysis when benchmark results need deeper explanation.
---

# Optimize

Optimize one Triton Ascend NPU operator through repeated validated rounds.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

## Inputs

- Operator source code or an operator file path

## Outputs

- A canonical `baseline/` directory under the operator workspace before the first optimization round completes
- One baseline operator snapshot under `baseline/`
- One `baseline/state.json`
- One `baseline/perf.txt`
- A sequence of `opt-round-N/` directories under the operator workspace
- One optimized operator file inside each completed round directory
- One `attempts.md` inside each round directory, updated throughout the round
- One `summary.md` inside each completed round directory
- Round-local benchmark evidence, with optional `profile/` and `ir/` evidence directories when deeper investigation is needed
- Updated `opt-note.md` in the operator workspace
- Updated `learned_lessons.md` in the operator workspace whenever the run discovers reusable optimization knowledge
- Correctness, benchmark, and profiling evidence produced through the `operator-eval` skill

## Required Preconditions

- Establish or reuse `baseline/` before treating any `opt-round-N/` directory as a completed optimization round.
- Check whether the operator workspace already has reusable correctness tests and benchmark cases.
- Reuse existing validation artifacts when they already cover the current optimize run.
- Generate missing correctness tests with `test-gen` only when the workspace does not already contain a usable harness.
- Generate missing benchmark cases with `bench-gen` only when the workspace does not already contain a usable harness.
- Do not start optimization rounds until both validation artifacts exist.

## Required References

- Read [workflow.md](references/workflow.md) before starting the first optimization round.
- Read [artifacts.md](references/artifacts.md) before creating any round directory or updating `opt-note.md`.
- Read [opt-note-format.md](references/opt-note-format.md) before editing `opt-note.md`.
- Read [round-failure-handling.md](references/round-failure-handling.md) when correctness or benchmark validation fails.
- Read [patterns/index.md](references/patterns/index.md) before choosing any optimization pattern reference.
- Use the sibling `ascend-npu-operator-profiler` skill when benchmark numbers need operator-level performance evidence, hotspot diagnosis, bottleneck analysis, or profiler-backed comparison across runs.
- Use the sibling `ascend-operator-ir-analyzer` skill when compiler lowering details, stage-to-stage IR changes, or round-local IR evidence are needed to explain benchmark behavior. In optimize rounds, keep that evidence under `opt-round-N/ir/`, for example:
  ```bash
  python3 ../ascend-operator-ir-analyzer/scripts/capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
  python3 ../ascend-operator-ir-analyzer/scripts/inspect_ir.py list-stages --ir-dir opt-round-N/ir --sort-by interesting --limit 20
  ```
- Use the bundled helper script at [`../operator-eval/scripts/run-command.py`](../operator-eval/scripts/run-command.py) for generation, validation, profiling, and comparison commands; if the outer optimize task is remote-aware, carry the same remote flags through those commands.
- When profiling benchmark harnesses, prefer `../operator-eval/scripts/run-command.py profile-bench ...`; carry the same `--remote` and `--remote-workdir` settings through profiler runs as well.
- Treat `references/knowledge/` as optional background material for future expansion, not part of the minimum optimize workflow.

## Pattern References

Do not read all pattern references at once.

Use [patterns/index.md](references/patterns/index.md) to choose the most relevant optimization direction first. Treat that index as the single entry point for detailed pattern references under `references/patterns/`.

Pattern references are helpful guidance, not the only allowed source of ideas.

If your own Triton, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction than the current pattern library, you may use that direction directly as long as you still record the hypothesis clearly and validate it with the same correctness and benchmark gates.

You do not need an existing pattern file to justify every optimization round.

## Learned Lessons

Maintain `learned_lessons.md` in the operator workspace as a running notebook of reusable optimization knowledge.

record learned lessons whenever you discover reusable knowledge, not only at the end of a round.

Use it for concise notes such as:

- compiler error repairs for Triton or Ascend NPU compilation failures
- profile-guided optimization lessons, including how profiling symptoms mapped to concrete code changes
- new optimization points inferred from recurring Triton code patterns
- validated ideas that are not yet covered by the existing pattern library
- benchmark interpretation rules that would help future rounds start faster

## Workflow

1. Inspect the operator workspace, resolve the correctness and benchmark modes, and confirm which validation artifacts already exist.
2. Generate missing tests or benchmarks through `../operator-eval/scripts/run-command.py` before starting any optimization round.
3. Generate only the missing harness types when reusable validation artifacts already exist.
4. Establish or reuse `baseline/` before creating `opt-round-1`.
5. If the operator or harnesses need minimal repair to produce a correct, benchmarkable starting point, do that work during baseline preparation rather than inside `opt-round-1`.
6. Save the canonical baseline as `baseline/state.json`, `baseline/perf.txt`, and one baseline operator snapshot under `baseline/`.
7. Treat the canonical baseline as the session-level comparison target, not as an optimization round.
8. Record a short diagnosis before the first code-changing round. The diagnosis should name the suspected bottleneck, the current evidence, and what kind of optimization direction looks justified.
9. Create `opt-round-N/`, copy the chosen parent operator into it, and start `attempts.md` immediately so every meaningful attempt and measurement is recorded.
10. Before editing code, state the optimization hypothesis for the round, explain why it may help, and cite the supporting evidence. Evidence may come from code inspection, benchmark behavior, profiling, IR inspection, or a combination of them.
11. If you skip profiling or IR capture for a round, explain in `attempts.md` why the existing evidence is already sufficient.
12. Read `references/patterns/index.md`, pick one optimization hypothesis, and read only the one or two detailed pattern references that match that hypothesis when they are relevant; if a better hypothesis comes from your own Triton or NPU optimization knowledge, use that hypothesis directly and document it clearly.
13. Apply one coherent optimization theme for the round, then run correctness validation before trusting any performance result.
14. Whenever you discover reusable debugging or optimization knowledge, append a short note to `learned_lessons.md` immediately instead of waiting for the round summary.
15. After correctness passes, run benchmark validation; when benchmark timing alone does not explain the result well enough, use `ascend-npu-operator-profiler` to gather operator-level evidence and keep the resulting profiler artifacts under the current round directory.
16. After both baseline and round benchmark perf artifacts exist, run `compare-perf` through `../operator-eval/scripts/run-command.py` and use that output as the only source for `Avg improvement`, `Geomean speedup`, `Total speedup`, and any claimed benchmark delta.
17. Do not hand-calculate speedups or percentage improvements from raw perf files.
18. Compare round benchmark results against `baseline/perf.txt` for canonical optimize-session metrics, even when the current round also compares locally against its chosen parent.
19. When IR inspection is needed to understand compiler lowering or confirm an optimization effect, use `ascend-operator-ir-analyzer`, capture into `opt-round-N/ir/`, and inspect that same directory directly.
20. Use the benchmark, profiler, IR evidence, and `compare-perf` output to decide whether to keep iterating, abandon the direction, or finalize the round `summary.md`, update `opt-note.md`, and preserve any reusable insight in `learned_lessons.md`.

## Quality Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Treat `baseline/` as the canonical optimize baseline for the session.
- Reuse existing tests and benchmark harnesses when they are already available for the workspace; generate new ones only when required artifacts are missing.
- Always run correctness before trusting performance results.
- Always explain why a proposed optimization may help before editing code for that round.
- Always record what evidence supports the chosen optimization direction.
- Use `baseline/perf.txt` for canonical optimize-session performance comparisons.
- Use profiler evidence when benchmark timing alone does not explain the result well enough.
- Always use `compare-perf` as the authoritative source for performance deltas and speedup metrics once comparable perf artifacts exist.
- Do not hand-calculate speedups or percentage improvements from raw perf files.
- Keep round-specific profiler evidence under `opt-round-N/profile/` and IR evidence under `opt-round-N/ir/` when they are collected.
- Keep parent-child traceability explicit so later engineers can understand which idea produced each round.
- Prefer multiple diverse optimization directions over a single greedy chain from the current best version.
- Prefer selective pattern reading over bulk-loading all optimization references.
- Do not begin with blind tiling, autotune, or launch-parameter search when the available evidence does not justify that direction.
- Prefer the strongest validated optimization idea, whether it comes from the pattern library or your own Triton and Ascend NPU knowledge.
- Do not silently discard optimization intent; preserve important comments that explain why a change helps.
- Record within-round attempts continuously so long-running rounds do not lose intermediate learning.
- Record reusable compiler fixes, profile interpretations, and newly discovered optimization heuristics in `learned_lessons.md` while they are still fresh.
- Record optimization points in enough detail that another engineer could reuse them on a related operator.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
