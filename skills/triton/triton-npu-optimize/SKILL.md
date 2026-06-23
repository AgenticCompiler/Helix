---
name: triton-npu-optimize
description: Iteratively optimize a Triton Ascend NPU operator with correctness and performance gates. Use for operator optimization tasks that need repeated correctness validation, benchmark validation, multi-round experiment tracking, reusable optimization notes, and profiler-backed performance analysis when benchmark results need deeper explanation.
---

# Optimize

## Goal

Optimize one Triton Ascend NPU operator through validated rounds anchored to a canonical `baseline/`.

Use this skill when the user wants the operator itself improved rather than only generating or running tests and benchmarks.

Optimize target modes:

- `kernel`: optimize the Triton Ascend NPU kernel path itself. Compare rounds with a kernel-oriented `compare-perf` view, but if the comparison falls back to total-op for some cases, record the resolved `effective_metric_source` and surface that mismatch as a warning instead of discarding the round outright.
- `operator`: optimize end-to-end operator latency. Wrapper logic, data movement, scheduling, pre-processing, post-processing, and kernel code are all valid optimization surfaces as long as the Triton Ascend NPU computation path remains real. Show both kernel and total-op `compare-perf` views, but use the total-op view as the canonical round conclusion.

## Inputs

- Operator source code or an operator file path

## Outputs

- `baseline/`
- `opt-round-N/`
- completed round entries and one final `## Overall Summary` in `opt-note.md`
- round-local `profile/`, `ir/`, `perf-analysis.md`, or `compiler-analysis.md` artifacts when deeper investigation is needed

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
- Otherwise use the sibling `ascend-npu-prepare-optimize-baseline` skill to establish or repair the baseline before creating `opt-round-1/`.
- Establish or reuse `baseline/` before treating any `opt-round-N/` directory as a completed optimization round.
- Read [artifacts.md](references/artifacts.md) before choosing authoritative baseline or round artifact paths.
- Read [opt-note-format.md](references/opt-note-format.md) before initializing `opt-note.md`.
- Keep top-level optimize workflow references at the skill boundary: use sibling skills for baseline preparation, evaluation, profiling, IR analysis, compiler-source analysis, and round gating rather than direct helper-script paths here.

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

## Optimization Priority Guide

Optimizations are **not equal in impact**. Prioritize based on the operator's kernel structure, not personal preference. Applying a low-impact optimization (e.g., compile hints) early may consume rounds that should go to structural changes, and structural changes may later make the early work obsolete.

### Priority 1: Structural optimizations (2-10x impact)

Eliminate Python for-loops that launch one Triton kernel per iteration by fusing the loop **inside** the kernel. The kernel maintains state in registers across loop iterations; it does not need to load all data upfront.

**Cue**: `for t in range(N): kernel[grid](...)` in the wrapper → fuse the loop inside the kernel as `for t in range(N): ...` inside `@triton.jit`.

**Cue**: `torch.exp(...)`, `torch_npu.npu_*(...)`, or other torch/npu operations in the wrapper that feed data into a Triton kernel → move them into the kernel using `tl.exp()`, `tl.load()` + inline computation, etc. Each external op remaining in the wrapper adds a separate NPU kernel launch.

**Cue**: When an operator file defines multiple Triton kernels on the same hot path → fuse them into a single kernel before optimizing further (also see Hard Rules).

### Priority 2: Wrapper overhead elimination (1.1-1.5x impact)

After the kernel structure is fixed, eliminate unnecessary wrapper-level operations:

- Dtype conversions (`tensor.to(torch.float32)`) that could be done inside the kernel with `.to(tl.float32)` — but prefer doing them once in the wrapper rather than per-load inside the kernel.
- Redundant `contiguous()` calls on already-contiguous tensors.
- Unnecessary mask + `other=` on `tl.load` when the tile exactly covers the dimension (e.g., hidden_size divisible by BLOCK_H and no tail case).

### Priority 3: Micro-architecture tuning (1.05-1.3x impact)

BLOCK_H/BLOCK_M/BLOCK_N, num_stages, autotune. Only effective after structural optimizations are applied. Autotune (preferred over manual sweeps) should be done last.

### Priority 4: Compiler hints (1.0-1.05x impact)

`tl.constexpr`, `tl.max_contiguous`, `tl.multiple_of`, `propagate_nan`. Small improvements on an already-optimized kernel. These should not consume rounds that could go to higher-priority work.

### Check before each round

Before committing to a round hypothesis, ask: *"Is there a higher-priority optimization available that I have not yet attempted?"* If yes, do that instead. The higher-priority change may make the current hypothesis obsolete or less impactful.

## Stage 2: Layered Analysis

Optimize analysis is layered.

- Default escalation order: `pattern triage -> profiling diagnosis -> IR attribution -> compiler-source escalation`.
- Start each round at the shallowest level that can justify the next move.
- Escalate only when the current level is insufficient.
- Keep `Primary analysis level` distinct from `Supporting evidence` in round records.
- Using IR as supporting evidence does not automatically change the round's primary analysis level.
- Record the chosen level and why the round stayed there or escalated deeper.
- Show the level explicitly in `attempts.md`, for example `Primary analysis level: profiling diagnosis`.
- When a round escalates, record both `Escalation: <from> -> <to>` and `Escalation reason: <why the previous level was insufficient>`.
- Do not rely on the presence of `profile/`, `ir/`, or `perf-analysis.md` to imply the current level; state it directly.

### pattern triage

- Inspect current code structure and benchmark behavior before choosing a direction.
- Use the sibling [`../triton-npu-optimize-knowledge/SKILL.md`](../triton-npu-optimize-knowledge/SKILL.md) as the generic optimize knowledge library.
- When the optimize target is `operator`, also use the sibling [`../torch-npu-optimize-knowledge/SKILL.md`](../torch-npu-optimize-knowledge/SKILL.md) for Torch NPU and whole-operator pattern routing such as framework-op fallback, wrapper-level changes, or broader operator restructuring.
- **Structural optimization priority gate (MUST evaluate before any micro-optimizations):** Before considering fp32-elision, contiguous-call removal, cache modifiers, or autotune, you **must** evaluate these three structural optimization directions in order. Record the evaluation result for each in `attempts.md` even if the answer is "not applicable":
  1. **Multi-row batching (program-multiple-rows, BLOCK_M variant):** Is the kernel row-structured and processing one row per program? Does the current launch lack a row-batching dimension (`BLOCK_M > 1`)? The **2D vectorized BLOCK_M variant** (`offs_m[:, None]` + `offs_n[None, :]` broadcasting with `BLOCK_M` as `tl.constexpr`) is the preferred form because it enables coalesced loads and parallel row processing. The looped `BLOCK_ROWS` variant is a weaker fallback. Read [`../triton-npu-optimize-knowledge/references/patterns/program-multiple-rows.md`](../triton-npu-optimize-knowledge/references/patterns/program-multiple-rows.md) and explicitly choose the 2D BLOCK_M variant if the kernel structure permits it.
  2. **Dispatch-parameter specialization:** Does the operator use dispatch parameters (dimension arguments, dtype, mode flags) that cause the same generic kernel to handle structurally different data layouts? Are there parameter values that incur avoidable wrapper-level overhead (layout transposition, contiguity materialization, reshape chains)? Evaluate whether dedicated kernels for specific parameter values can materialize the optimal layout on the host before kernel launch, eliminating in-kernel compensation work. This is the **layout-materialization-elision** approach.
  3. **Grid-decomposition optimization (tile-selection-heuristic):** After structural dispatch is addressed, evaluate whether a manual tile-selection heuristic (grid-minimization sweep over `BLOCK_M × BLOCK_SIZE` tile sizes) can beat autotune. Autotune with a limited config set may not adapt well to diverse shape regimes. If the operator spans wide shape ranges (e.g., rows varying by orders of magnitude, col widths from tens to tens of thousands), read [`../triton-npu-optimize-knowledge/references/patterns/tile-selection-heuristic.md`](../triton-npu-optimize-knowledge/references/patterns/tile-selection-heuristic.md) and consider replacing or augmenting autotune with a grid-minimization heuristic.
- After the structural optimization priority gate is satisfied (all three directions evaluated and addressed where applicable), proceed to the mandatory diagnostic steps below.
- **Simulation-signal as mandatory diagnostic step:** When `extracted_bin_data/report.txt` exists, you must read [`../triton-npu-optimize-knowledge/references/patterns/scalar-vector-simulation-signal.md`](../triton-npu-optimize-knowledge/references/patterns/scalar-vector-simulation-signal.md) and execute the signal matching flow (check each Category in priority order). If any Category fires, `scalar-vector-simulation-signal (Cat N)` could appear as a candidate pattern in the `attempts.md` candidate list. When selecting the final pattern, evaluate `scalar-vector-simulation-signal` alongside other candidates: it may be selected as the primary pattern (using its Code Manifestations and generic transforms), or it may route to a more specific domain pattern via its Related Patterns / Optimization Direction. Record the fired Categories and the reasoning for whether simulation-signal was selected or routed to another pattern in `attempts.md`.
- **Autotune as mandatory signal-driven optimization:** When `extracted_bin_data/report.txt` exists in the workspace or current `opt-round-N/` directory, you **must** read [`../triton-npu-optimize-knowledge/references/patterns/autotune.md`](../triton-npu-optimize-knowledge/references/patterns/autotune.md) and execute all three phases. Execute this check **after** the simulation-signal check (scalar-vector-simulation-signal) so that code-level signals are evaluated first. **Phase 1 (Pre-Gate A-Cat-5):** Check if memory layout is fundamentally fragmented. If A-Cat-5 fires, autotune is not suitable — route to `compile_hint` or `discrete_memory_access` and exit. **Phase 2 (Parameter Diagnostics):** Only if Pre-Gate passes — check A-Cat-6 → A-Cat-1 → A-Cat-2 → A-Cat-3 → A-Cat-4 in order. The fired Category's Optimization Direction already contains concrete `triton.Config` examples. **Phase 3 (Config Route):** Choose Route 1 (`configs=[]` auto-infer), Route 2 (`hints`), or Route 3 (hand-written, using Category examples as starting points). Record the diagnosis result and reasoning explicitly in `attempts.md`.
- **Profiling-data-driven self-directed exploration:** When `extracted_bin_data/` exists, read the instruction-level profiling methodology in [`references/optimize.md`](references/optimize.md). Read all available files: `report.txt`, `dataType_4_API_INSTR.json`, `dataType_3_API_FILE.json`, `dataType_1_SOURCE_<operator_name>.txt`, `flows.json`, and `dataType_2_TRACE.json`. Identify bottlenecks against all 6 metrics, evaluate all 7 optimization categories, and avoid the 5 prohibited optimizations. Every metric and category needs an explicit conclusion. Record bottleneck metrics, evaluated categories, and their conclusions in `attempts.md`.
- **Cross-round profiling comparison:** When multiple rounds have `extracted_bin_data/`, compare profiling characteristics across `baseline/`, the best-performing round, and the current round. Identify which profiling feature changes correlated with prior improvements and pursue directions that move those same metrics favorably.
- **Self-directed exploration still requires hypothesis and validation:** Record an explicit hypothesis in `attempts.md` before changing code, make one coherent change per attempt, and validate with correctness and benchmark before claiming success, as with any other optimization round.
- When a profiling-data-driven direction produces a concrete improvement, record the primary bottleneck category and the applied optimization category in `summary.md`.
- Read [`../triton-npu-optimize-knowledge/references/pattern_index.md`](../triton-npu-optimize-knowledge/references/pattern_index.md) before detailed pattern references.
- Read only the one or two most relevant detailed pattern files under [`../triton-npu-optimize-knowledge/references/patterns/`](../triton-npu-optimize-knowledge/references/patterns/) after the generated index has narrowed the candidate set.
- When the optimize target is `operator` and the bottleneck looks Torch NPU or framework-op specific, read [`../torch-npu-optimize-knowledge/references/pattern_index.md`](../torch-npu-optimize-knowledge/references/pattern_index.md) before detailed Torch NPU pattern references.
- When code structure is still unclear at pattern triage, inspect the operator file directly and narrow candidates with the generated pattern index.
- Do not leave the chosen pattern implicit in free-form prose; write it down explicitly in `attempts.md`, and carry the final named pattern direction into `summary.md` when it guided the round.
- Very strongly consider using subagents to read pattern references, scan for potentially useful optimization ideas, and report back which patterns look promising for the current kernel.
- When subagents are available, prefer using them to broaden pattern exploration before committing to the round hypothesis.
- Pattern references are helpful guidance, not the only allowed source of ideas.
- If your own Triton, Ascend NPU, or kernel-optimization knowledge suggests a stronger direction than the current pattern library, you may use that direction directly as long as you still record the hypothesis clearly and validate it with the same correctness and benchmark gates.
- You do not need an existing pattern file to justify every optimization round.
- When the kernel is structurally matmul-like, read the staged `triton-npu-optimize-knowledge` skill's `references/patterns/classic-matmul.md` before rewriting the hot loop so the round records the standard tiled-matmul shape, dtype, and masking rules explicitly.
- Do not treat pattern triage as permission for aimless pattern search without tying the candidate patterns back to the kernel structure or observed evidence.

### profiling diagnosis

- Use profiling diagnosis as the default deeper entrypoint when pattern triage is not enough.
- Use the sibling `ascend-npu-profile-operator` skill when benchmark numbers need operator-level performance evidence, hotspot diagnosis, bottleneck analysis, or profiler-backed comparison across runs.
- Use the sibling `ascend-npu-analyze-round-performance` skill when one round needs a deeper diagnosis that should end in `opt-round-N/perf-analysis.md`, especially for scalar/vector/cube imbalance, transfer-heavy behavior, or suspected pipeline overlap issues.
- Use the sibling knowledge skill's symptom cards to narrow pattern candidates after structured profiler or IR evidence exists, rather than rereading the whole pattern library.
- This deeper diagnosis may end as either `profile-only diagnosis` or `profile-plus-IR diagnosis`.
- Write `opt-round-N/perf-analysis.md` when the deeper round-analysis flow is used.

### IR attribution

- Use IR attribution only after profiler-backed symptoms still need explanation.
- `ascend-npu-analyze-round-performance` may still own `opt-round-N/perf-analysis.md` when the round deepens from profiler evidence into IR-backed attribution.
- In that flow, use `triton-npu-analyze-ir` as the IR evidence companion for capture, navigation, and stage-level inspection.
- Use the sibling `triton-npu-analyze-ir` skill when compiler lowering details, stage-to-stage IR changes, or round-local IR evidence are needed to explain benchmark behavior.
- Keep IR evidence under `opt-round-N/ir/`.
- In optimize rounds, keep IR capture round-local. Use the `triton-npu-analyze-ir` skill's helpers with argument shapes such as:
  ```text
  capture_ir.py --ir-dir opt-round-N/ir --bench-file bench_<operator>.py --operator-file opt-round-N/<optimized-operator>.py
  inspect_ir.py list-stages --ir-dir opt-round-N/ir --sort-by interesting --limit 20
  ```

### compiler-source escalation

- Use compiler-source escalation only when compiler source analysis is enabled and after profiler and IR evidence have narrowed a concrete compiler-side question.
- Use the sibling `triton-npu-analyze-compiler-source` skill only when the round still needs performance-focused explanation before the next operator change is clear.
- When compiler source analysis is enabled, treat the compiler source checkout as read-only and use compiler-source evidence only when it is genuinely needed.
- Write `opt-round-N/compiler-analysis.md`.

## Kernel Semantic Repairs

During optimization, you may encounter Triton kernel parameters that control numerical or behavioral semantics (NaN propagation, precision, rounding, overflow behavior, etc.). These require special handling because on Ascend NPU:

1. **They change numerical output** — always validate correctness when adding or removing them.
2. **They may also affect performance** — a parameter that appears purely semantic can influence compiler instruction selection. The performance impact cannot be predicted from first principles; it must be measured.

### General rules

- During kernel optimization work, inspect call sites of operators that expose semantic parameters (e.g., `tl.maximum`, `tl.minimum`, or any function with `propagate_nan` or similar flags) across the entire operator, not only the lines touched by the primary optimization. Inconsistent parameter usage between call sites is a red flag.
- When a semantic parameter is absent from a call site where it could apply:
  - **Try adding it and benchmark both variants.** Record the benchmark comparison explicitly in `attempts.md`.
  - If the change regresses or is neutral, restore the original and document the result.
- **Do not skip this based on reasoning alone.** The most common mistake is treating a semantic parameter as "just a correctness change, not a performance change" and dismissing it without measurement. If you find yourself thinking "this is only a correctness change," that is exactly the signal to benchmark it.
- **Test semantic changes in isolation when possible.** If you bundle a semantic-parameter change with other modifications and correctness fails, follow the isolated-testing guidance in Stage 3 to isolate the true cause. A semantic change incorrectly blamed for an unrelated failure may never be re-evaluated.
- When the semantics controlled by the parameter do not matter for the operator's use case, still benchmark both variants — on Ascend NPU the performance effect is often independent of whether the semantics are required.

### Example: `propagate_nan` on `tl.maximum` / `tl.minimum`

The most common semantic repair on Ascend NPU kernels is adding `propagate_nan=tl.PropagateNan.ALL` to `tl.maximum()` or `tl.minimum()` calls that omit it. Apply the general rules above when encountering this case. Specifically:

- `propagate_nan` changes NaN-input behavior (semantic effect) but also selects faster vector comparison instructions on Ascend NPU (performance effect).
- Benchmark both with and without `propagate_nan` even when NaN propagation is not semantically required — the performance effect is independent of whether NaN semantics matter.

## Stage 3: Validate And Record

- Run correctness validation before trusting any performance result.
- After correctness passes, you may use the sibling `ascend-npu-run-eval` skill's `probe-bench` flow as a fast baseline-vs-candidate screen when you need a cheap directional signal before paying canonical benchmark cost.
- Use `probe-bench` only when the active run-eval surface actually exposes it. If the current surface does not expose `probe-bench`, skip the screen and continue with canonical validation.
- Treat `probe-bench` as non-canonical screening evidence only. Do not use its output as the official round perf artifact, do not write probe artifacts into `round-state.json`, and do not claim round speedups from probe output alone.
- Use `probe-bench` to reject clearly bad candidates early or to justify keeping a clearly promising direction long enough to run canonical benchmarking.
- After correctness passes, run benchmark validation and preserve the round-local benchmark evidence.
- In each round directory, keep the optimized operator snapshot as `opt_<original-operator>.py`.
- In each round directory, keep the benchmark artifact as `opt_<original-operator>_perf.txt`, ensure that file is generated by the `ascend-npu-run-eval` skill's `run-bench` flow, and record that exact filename in `round-state.json`.
- Use `baseline/<operator>_perf.txt` for canonical optimize-session performance comparisons, even when the current round also compares locally against its parent.
- Once baseline and round perf artifacts both exist, use the `ascend-npu-run-eval` skill to run `compare-perf`.
- Even after `probe-bench` reports `likely_gain` or `likely_regression`, still run canonical `run-bench` and `compare-perf` before recording any official round conclusion.
- Treat the `ascend-npu-run-eval` skill's `compare-perf` flow as the only authority for claimed benchmark deltas and speedups, including `Avg improvement`, `Geomean speedup`, and any claimed benchmark delta.
- Record exactly one resolved comparison basis in `round-state.json` as `effective_metric_source`, using `kernel`, `total-op`, or `mixed`.
- In `kernel` target mode, prefer the kernel-oriented comparison result, but if `compare-perf` falls back to total-op for some or all cases, keep the round eligible and record that fallback as a warning.
- In `operator` target mode, show both kernel and total-op comparison results so you can diagnose whether kernel improvements translated end-to-end, then record `effective_metric_source: total-op` for the official round conclusion.
- Do not hand-calculate speedups or percentage improvements from raw perf files.
- Use the sibling `ascend-npu-optimize-state` skill's `submit-round` subcommand to submit the current round (with `--min-rounds <N>` when the session has a minimum round requirement) and repair the round until it passes before continuing or stopping.
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

Do not use `learned_lessons.md` for round narrative, local command failures, failed guesses, temporary troubleshooting notes, file names, shape-specific details, or summaries of what happened in one round.

Put round-local narrative, temporary troubleshooting notes, command failures, and shape-specific details in `opt-round-N/attempts.md`, `opt-round-N/summary.md`, or `opt-note.md` instead.

## Hard Rules

- Optimize the operator file itself, not the generated tests or benchmark harness.
- Optimize the Triton kernel path, not just the public wrapper surface.
- Do not replace the core computation with pure PyTorch just to improve benchmark numbers.
- Do not claim success for a round without both correctness evidence and benchmark evidence.
- Do not begin with blind tiling, autotune, or launch-parameter search when the available evidence does not justify that direction.
- When launch or tile/block configuration tuning is justified, prefer `autotune` over burning multiple optimization rounds on hand-tuned parameter sweeps.
- Do not finish a round by restoring the parent snapshot alone; if edits are discarded, restart from the last validated parent as a new attempt or round instead of claiming rollback as the delivered optimization.
- Do not spend optimization rounds tuning `num_warps` for Ascend NPU targets; it is CUDA-oriented and is not a meaningful Ascend launch knob.
- When an operator file defines multiple Triton kernels on a relevant hot path, fuse them into a single kernel first, then optimize that fused kernel rather than tuning separate kernels in isolation. The same applies when Torch/CANN auxiliary ops precede a Triton kernel on a hot path and only feed that kernel — fold the auxiliary logic into the Triton kernel rather than leaving it as separate `aclnn*` / torch ops. The fused implementation must remain a Triton kernel; do not delegate the fused logic to `torch.ops.npu.*` or `aclnn*` ops. See the [`../triton-npu-optimize-knowledge/references/patterns/auxiliary-op-fusion.md`](../triton-npu-optimize-knowledge/references/patterns/auxiliary-op-fusion.md) pattern for the fused-kernel mechanics, multi-output and multi-stage variants, and avoid conditions.
- Do not write forward-looking optimization plans or predict what to try next. After each round, the next optimization direction must be driven by fresh profiling or benchmark evidence from the changed code. Write no more than one round at a time.
