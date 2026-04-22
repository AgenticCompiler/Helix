---
name: triton-npu-optimize
description: Iteratively optimize a Triton Ascend NPU operator with correctness and performance gates. Use for operator optimization tasks that need repeated correctness validation, benchmark validation, multi-round experiment tracking, reusable optimization notes, and profiler-backed performance analysis when benchmark results need deeper explanation.
---

# Optimize

## Goal

Optimize one Triton Ascend NPU operator through validated rounds anchored to a canonical `baseline/`.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

## Inputs

- Operator source code or an operator file path

## Outputs

- `baseline/`
- `opt-round-N/`
- completed round entries and one final `## Overall Summary` in `opt-note.md`
- updated `learned_lessons.md` only when the session discovers reusable optimization knowledge that passes the admission bar
- round-local `profile/`, `ir/`, `perf-analysis.md`, or `compiler-analysis.md` artifacts when deeper investigation is needed

## Core Loop

- establish or reuse `baseline/`
- open `opt-round-N/` and start `attempts.md`
- choose the current analysis level
- make one coherent optimization attempt
- validate correctness and benchmark performance
- record the round outcome

## Stage 0: Baseline Setup

- Reuse the existing `baseline/` only when it remains canonical for the current operator workspace.
- Otherwise use the sibling `triton-npu-prepare-optimize-baseline` skill to establish or repair the baseline before creating `opt-round-1/`.
- Establish or reuse `baseline/` before treating any `opt-round-N/` directory as a completed optimization round.
- Read [artifacts.md](references/artifacts.md) before choosing authoritative baseline or round artifact paths.
- Read [opt-note-format.md](references/opt-note-format.md) before initializing `opt-note.md`.
- Keep top-level optimize workflow references at the skill boundary: use sibling skills for baseline preparation, evaluation, profiling, IR analysis, compiler-source analysis, and round gating rather than direct helper-script paths here.

## Stage 1: Round Entry

- Create `opt-round-N/` from a validated parent candidate and keep parent-child traceability explicit.
- Start `attempts.md` immediately so every meaningful attempt and measurement is recorded.
- For round 1, record the initial round hypothesis in `opt-round-1/attempts.md` before the first code change.
- Record the current analysis level, why it may help, and what evidence supports starting there.
- If the round starts from reused deeper evidence, cite the reused evidence path and explain why the shallower level is already established or insufficient.
- Treat `opt-note.md` as the top-level round ledger plus final `## Overall Summary`.

## Stage 2: Layered Analysis

### pattern triage

- Inspect current code structure and benchmark behavior before choosing a direction.
- Read `references/patterns/index.md`.
- Read only the one or two detailed pattern references that match a real hypothesis.
- Pattern references are helpful guidance, not the only allowed source of ideas.
- If your own Triton, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction than the current pattern library, you may use that direction directly as long as you still record the hypothesis clearly and validate it with the same correctness and benchmark gates.
- You do not need an existing pattern file to justify every optimization round.
- Do not treat pattern triage as permission for blind pattern search.

### profiling diagnosis

- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.
- Use the sibling `triton-npu-profile-operator` skill when benchmark numbers need operator-level performance evidence, hotspot diagnosis, bottleneck analysis, or profiler-backed comparison across runs.
- Use the sibling `triton-npu-analyze-round-performance` skill when one round needs a deeper diagnosis that should end in `opt-round-N/perf-analysis.md`, especially for scalar/vector/cube imbalance, transfer-heavy behavior, or suspected pipeline overlap issues.
- Write `opt-round-N/perf-analysis.md` when the deeper round-analysis flow is used.

### IR attribution

- Use IR attribution only after profiler-backed symptoms still need explanation.
- Use the sibling `triton-npu-analyze-ir` skill when compiler lowering details, stage-to-stage IR changes, or round-local IR evidence are needed to explain benchmark behavior.
- Keep IR evidence under `opt-round-N/ir/`.
- In optimize rounds, keep IR capture round-local, for example:
  ```bash
  python3 ../triton-npu-analyze-ir/scripts/capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
  python3 ../triton-npu-analyze-ir/scripts/inspect_ir.py list-stages --ir-dir opt-round-N/ir --sort-by interesting --limit 20
  ```

### compiler-source escalation

- Use compiler-source escalation only when compiler source analysis is enabled and after profiler and IR evidence have narrowed a concrete compiler-side question.
- When compiler source analysis is enabled, treat the compiler source checkout as read-only and use the sibling `triton-npu-analyze-compiler-source` skill only when compiler-source evidence is genuinely needed.
- Write `opt-round-N/compiler-analysis.md`.

## Stage 3: Validate And Record

- Run correctness validation before trusting any performance result.
- After correctness passes, run benchmark validation and preserve the round-local benchmark evidence.
- Use `baseline/perf.txt` for canonical optimize-session performance comparisons, even when the current round also compares locally against its parent.
- Always use the `triton-npu-run-eval` skill to run `compare-perf` after both baseline and round perf artifacts exist.
- Use the `triton-npu-run-eval` skill to run `compare-perf` after both baseline and round perf artifacts exist.
- Always use the `triton-npu-run-eval` skill's `compare-perf` flow as the authoritative source for performance deltas and speedup metrics once comparable perf artifacts exist.
- Use `compare-perf` as the only source for `Avg improvement`, `Geomean speedup`, `Total speedup`, and any claimed benchmark delta.
- Do not hand-calculate speedups or percentage improvements from raw perf files.
- Use the sibling `triton-npu-optimize-check` skill to run `check-round` and repair the current round until it passes before continuing or stopping.

## Round Records

- `attempts.md`: chronological round log for the current round, including the current analysis level, the starting hypothesis, escalation reasons, meaningful code changes, correctness failures, and benchmark outcomes.
- `summary.md`: round conclusion, optimization points that mattered, the final analysis level, and which evidence actually decided the round.
- `opt-note.md`: top-level round ledger plus final `## Overall Summary`.
- `learned_lessons.md`: strict reusable knowledge only.

Maintain `learned_lessons.md` in the operator workspace as a strict reusable optimization-knowledge distillation log.

Append a lesson only when it passes all admission criteria:

- The lesson generalizes to a family of Triton Ascend NPU operators, not only the current operator.
- The lesson is supported by correctness, benchmark, profiler, IR, or compiler-error evidence.
- The lesson is written as a reusable rule, diagnostic mapping, or optimization heuristic.
- The lesson states where it applies or what limits it.
- The lesson could plausibly be promoted into an optimize skill, profiling analysis reference, IR analysis reference, or pattern reference.

Use `learned_lessons.md` for concise distilled rules such as:

- profile-to-optimization mappings
- IR-to-code-change mappings
- compiler error repairs that reveal recurring Triton or Ascend NPU constraints
- new optimization points inferred from recurring Triton code patterns
- validated benchmark interpretation rules that would help future rounds start faster

Do not use `learned_lessons.md` for round narrative, local command failures, failed guesses, temporary troubleshooting notes, file names, shape-specific details, or summaries of what happened in one round. Put that material in `opt-round-N/attempts.md`, `opt-round-N/summary.md`, or `opt-note.md` instead.

## Hard Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Optimize the Triton kernel path, not just the public wrapper surface.
- Do not replace the core computation with pure PyTorch just to improve benchmark numbers.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
- Do not begin with blind tiling, autotune, or launch-parameter search when the available evidence does not justify that direction.
- Do not put round narrative into `learned_lessons.md`.
- Record reusable compiler fixes, profile interpretations, and newly discovered optimization heuristics in `learned_lessons.md` while they are still fresh, but only when they pass the strict admission criteria.
