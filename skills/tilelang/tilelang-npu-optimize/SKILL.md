---
name: tilelang-npu-optimize
description: Iteratively optimize a TileLang Ascend NPU operator with correctness and performance gates. Use for operator optimization tasks that need repeated correctness validation, benchmark validation, multi-round experiment tracking, reusable optimization notes, and profiler-backed performance analysis when benchmark results need deeper explanation.
---

# Optimize (TileLang)

## Goal

Optimize one TileLang Ascend NPU operator through validated rounds anchored to a canonical `baseline/`.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

Optimize target modes:

- `kernel`: optimize the TileLang Ascend NPU kernel path itself. Compare rounds with a kernel-oriented `compare-perf` view, but if the comparison falls back to total-op for some cases, record the resolved `effective_metric_source` and surface that mismatch as a warning instead of discarding the round outright.
- `operator`: optimize end-to-end operator latency. Wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code are all valid optimization surfaces as long as the TileLang Ascend NPU computation path remains real. Show both kernel and total-op `compare-perf` views, but use the total-op view as the canonical round conclusion.

## Inputs

- Operator source code or an operator file path

## Outputs

- `baseline/`
- `opt-round-N/`
- completed round entries and one final `## Overall Summary` in `opt-note.md`
- round-local `profile/` or `perf-analysis.md` artifacts when deeper investigation is needed

## TileLang API Reference

Before any round, read the [TileLang API reference](../tilelang-npu-api-reference/SKILL.md) so you understand the APIs the kernel uses.

## Core Loop

- establish or reuse `baseline/`
- open `opt-round-N/`, initialize round strategy state through `ascend-npu-optimize-state start-round`, and start `attempts.md`
- choose the current analysis level
- make one coherent optimization attempt
- optionally screen the direction cheaply with `probe-bench` when the available run-eval surface exposes it
- validate correctness and benchmark performance
- record the round outcome

## Stage 0: Baseline Setup

- Reuse the existing `baseline/` only when it remains canonical for the current operator workspace.
- Otherwise use the sibling `npu-prepare-optimize-baseline` skill to establish or repair the baseline before creating `opt-round-1/`.
- Establish or reuse `baseline/` before treating any `opt-round-N/` directory as a completed optimization round.
- Read [artifacts.md](references/artifacts.md) before choosing authoritative baseline or round artifact paths.
- Keep top-level optimize workflow references at the skill boundary: use sibling skills for baseline preparation, evaluation, profiling, and round gating rather than direct helper-script paths here.

## Stage 1: Round Entry

- Create `opt-round-N/` from a validated parent candidate and keep parent-child traceability explicit.
- Use the sibling `ascend-npu-optimize-state` skill's `start-round` subcommand to initialize the active round's `round_strategy`, `analysis_policy`, and `reason` before the first code change in that round.
- Start `attempts.md` immediately so every meaningful attempt and measurement is recorded.
- Treat the structured `State Update` blocks in `attempts.md` as script-written workflow history; do not manually duplicate the same `round_strategy`, `analysis_policy`, and `reason` bookkeeping in both `attempts.md` and `summary.md`.
- For round 1, record the initial round hypothesis in `opt-round-1/attempts.md` before the first code change.
- When pattern triage is used, explicitly record the candidate patterns you considered, the selected pattern if one is chosen, and why that pattern looks plausible in `attempts.md`.
- When a named pattern guides the round, explicitly record the final selected pattern direction in `summary.md`.
- Record the current analysis level as `Primary analysis level`, and record `Supporting evidence` separately.
- Record why that level may help and what evidence supports starting there.
- If the round starts from reused deeper evidence, cite the reused evidence path and explain why the shallower level is already established or insufficient.
- Treat `opt-note.md` as the top-level round ledger plus final `## Overall Summary`.
- If the active round's intent or required evidence depth changes mid-round, use the sibling `ascend-npu-optimize-state` skill's `set-current-round-state` subcommand instead of silently changing the round contract in prose only.

## Stage 2: Layered Analysis

Optimize analysis is layered.

- Default escalation order: `pattern triage -> profiling diagnosis`.
- Start each round at the shallowest level that can justify the next move.
- Escalate only when the current level is insufficient.
- Keep `Primary analysis level` distinct from `Supporting evidence` in round records.
- Record the chosen level and why the round stayed there or escalated deeper.
- Show the level explicitly in `attempts.md`, for example `Primary analysis level: profiling diagnosis`.
- When a round escalates, record both `Escalation: <from> -> <to>` and `Escalation reason: <why the previous level was insufficient>`.
- Do not rely on the presence of `profile/` or `perf-analysis.md` to imply the current level; state it directly.

### pattern triage

- Inspect current code structure and benchmark behavior before choosing a direction.
- Use the sibling [`../tilelang-npu-optimize-knowledge/SKILL.md`](../tilelang-npu-optimize-knowledge/SKILL.md) as the TileLang optimize knowledge library.
- When the optimize target is `operator`, also use the sibling [`../torch-npu-optimize-knowledge/SKILL.md`](../torch-npu-optimize-knowledge/SKILL.md) for Torch NPU and whole-operator pattern routing such as framework-op fallback, wrapper-level changes, or broader operator restructuring.
- Read [`../tilelang-npu-optimize-knowledge/references/pattern_index.md`](../tilelang-npu-optimize-knowledge/references/pattern_index.md) before detailed pattern references (create this file if it does not exist yet).
- Read only the one or two most relevant detailed pattern files under [`../tilelang-npu-optimize-knowledge/references/patterns/`](../tilelang-npu-optimize-knowledge/references/patterns/) after the generated index has narrowed the candidate set.
- When the optimize target is `operator` and the bottleneck looks Torch NPU or framework-op specific, read [`../torch-npu-optimize-knowledge/references/pattern_index.md`](../torch-npu-optimize-knowledge/references/pattern_index.md) before detailed Torch NPU pattern references.
- When code structure is still unclear at pattern triage, inspect the operator file directly and narrow candidates with the generated pattern index.
- Do not leave the chosen pattern implicit in free-form prose; write it down explicitly in `attempts.md`, and carry the final named pattern direction into `summary.md` when it guided the round.
- Very strongly consider using subagents to read pattern references, scan for potentially useful optimization ideas, and report back which patterns look promising for the current kernel.
- When subagents are available, prefer using them to broaden pattern exploration before committing to the round hypothesis.
- Pattern references are helpful guidance, not the only allowed source of ideas.
- If your own TileLang, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction than the current pattern library, you may use that direction directly as long as you still record the hypothesis clearly and validate it with the same correctness and benchmark gates.
- You do not need an existing pattern file to justify every optimization round.
- Do not treat pattern triage as permission for aimless pattern search without tying the candidate patterns back to the kernel structure or observed evidence.

### profiling diagnosis

- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.
- Use the sibling `npu-profile-operator` skill when benchmark numbers need operator-level performance evidence, hotspot diagnosis, bottleneck analysis, or profiler-backed comparison across runs.
- Use the sibling `npu-analyze-round-performance` skill when one round needs a deeper diagnosis that should end in `opt-round-N/perf-analysis.md`, especially for scalar/vector/cube imbalance, transfer-heavy behavior, or suspected pipeline overlap issues.
- Use the sibling knowledge skill's symptom cards to narrow pattern candidates after structured profiler evidence exists, rather than rereading the whole pattern library.
- Write `opt-round-N/perf-analysis.md` when the deeper round-analysis flow is used.

## Stage 3: Validate And Record

- Run correctness validation before trusting any performance result.
- After correctness passes, you may use the sibling `npu-run-eval` skill's `probe-bench` flow as a fast baseline-vs-candidate screen when you need a cheap directional signal before paying canonical benchmark cost.
- Use `probe-bench` only when the active run-eval surface actually exposes it. If the current surface does not expose `probe-bench`, skip the screen and continue with canonical validation.
- Treat `probe-bench` as non-canonical screening evidence only. Do not use its output as the official round perf artifact, do not write probe artifacts into `round-state.json`, and do not claim round speedups from probe output alone.
- Use `probe-bench` to reject clearly bad candidates early or to justify keeping a clearly promising direction long enough to run canonical benchmarking.
- After correctness passes, run benchmark validation and preserve the round-local benchmark evidence.
- In each round directory, keep the optimized operator snapshot as `opt_<original-operator>.py`.
- In each round directory, keep the benchmark artifact as `opt_<original-operator>_perf.txt`, ensure that file is generated by the `npu-run-eval` skill's `run-bench` flow, and record that exact filename in `round-state.json`.
- Use `baseline/<operator>_perf.txt` for canonical optimize-session performance comparisons, even when the current round also compares locally against its parent.
- Once baseline and round perf artifacts both exist, use the `npu-run-eval` skill to run `compare-perf`.
- Even after `probe-bench` reports `likely_gain` or `likely_regression`, still run canonical `run-bench` and `compare-perf` before recording any official round conclusion.
- Treat the `npu-run-eval` skill's `compare-perf` flow as the only authority for claimed benchmark deltas and speedups, including `Avg improvement`, `Geomean speedup`, and any claimed benchmark delta.
- Record exactly one resolved comparison basis in `round-state.json` as `effective_metric_source`, using `kernel`, `total-op`, or `mixed`.
- In `kernel` target mode, prefer the kernel-oriented comparison result, but if `compare-perf` falls back to total-op for some or all cases, keep the round eligible and record that fallback as a warning.
- In `operator` target mode, show both kernel and total-op comparison results so you can diagnose whether kernel improvements translated end-to-end, then record `effective_metric_source: total-op` for the official round conclusion.
- Do not hand-calculate speedups or percentage improvements from raw perf files.
- Use the sibling `ascend-npu-optimize-state` skill's `submit-round` subcommand to submit the current round and repair the round until it passes before continuing or stopping.
- After the round submission passes, read the JSON `guideline` field for the exit signal: if minimum rounds are satisfied, the session may stop after this round.
- Before opening the next round, use the sibling `ascend-npu-optimize-state` skill's `start-round` subcommand to re-check the one-round-at-a-time and no-blind-sweep workflow constraints.

## Round Records

- `attempts.md`: chronological round log for the current round, including the script-written `State Update` history, `Primary analysis level`, `Supporting evidence`, the starting hypothesis, selected pattern candidates and pivots when pattern triage is used, escalation reasons, meaningful code changes, correctness failures, probe-screening outcomes when `probe-bench` is used, and canonical benchmark outcomes.
- `summary.md`: round conclusion, optimization points that mattered, the final selected pattern direction when one guided the round, the final analysis level, supporting evidence that decided the round, and unresolved questions if deeper analysis may still be needed. Do not duplicate the full round strategy state history here.
- `opt-note.md`: top-level round ledger plus final `## Overall Summary`.

## Learned Lessons

Maintain `learned_lessons.md` in the operator workspace as a strict reusable optimization-knowledge distillation log.

Admission criteria:

Append a lesson only when it passes all admission criteria:

- The lesson generalizes to a family of TileLang Ascend NPU operators, not only the current operator.
- The lesson is supported by correctness, benchmark, profiler, or compiler-error evidence.
- The lesson is written as a reusable rule, diagnostic mapping, or optimization heuristic.
- The lesson states where it applies or what limits it.
- The lesson could plausibly be promoted into an optimize skill, profiling analysis reference, or pattern reference.

Use `learned_lessons.md` for concise distilled rules such as:

- profile-to-optimization mappings
- compiler error repairs that reveal recurring TileLang or Ascend NPU constraints
- new optimization points inferred from recurring TileLang code patterns
- validated benchmark interpretation rules that would help future rounds start faster

Do not use `learned_lessons.md` for round narrative, local command failures, failed guesses, temporary troubleshooting notes, file names, shape-specific details, or summaries of what happened in one round.

Put round-local narrative, temporary troubleshooting notes, command failures, and shape-specific details in `opt-round-N/attempts.md`, `opt-round-N/summary.md`, or `opt-note.md` instead.

## Hard Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Optimize the TileLang kernel path, not just the public wrapper surface.
- Do not replace the core computation with pure PyTorch just to improve benchmark numbers.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
- Do not begin with blind tiling or launch-parameter search when the available evidence does not justify that direction.
- When launch or tile/block configuration tuning is justified, prefer an autotuning approach over burning multiple optimization rounds on hand-tuned parameter sweeps.
- Do not finish a round by restoring the parent snapshot alone; if edits are discarded, restart from the last validated parent as a new attempt or round instead of claiming rollback as the delivered optimization.
- Do not write forward-looking optimization plans or predict what to try next. After each round, the next optimization direction must be driven by fresh profiling or benchmark evidence from the changed code. Write no more than one round at a time.
