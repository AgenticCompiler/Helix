---
name: triton-npu-analyze-round-performance
description: Use when an optimize round needs deep performance diagnosis from round-local profile and optional IR evidence, especially to explain scalar/vector/cube imbalance, frequent data movement, weak pipeline overlap, or other signals that should be traced back to problems in the current operator implementation.
---

# Analyze Optimize Round Performance

Diagnose one `opt-round-N/` at a time and write the result to `opt-round-N/perf-analysis.md`.

This skill is for deep round analysis inside `triton-npu-optimize`, not for supervisor audits and not for whole-session summaries.
This skill is the owner of `opt-round-N/perf-analysis.md` for round-level performance diagnosis inside `triton-npu-optimize`.

It supports both:

- `profile-only diagnosis`
- `profile-plus-IR diagnosis`

When IR attribution is needed, use `triton-npu-analyze-ir` as the IR evidence companion for capture, navigation, and stage-level inspection. That does not transfer ownership of `opt-round-N/perf-analysis.md`.

Use two complementary analysis paths to find performance problems:

- profiling analysis to identify where time is going and which hardware-facing symptom dominates
- IR analysis to explain why that symptom appears in the current lowering and operator structure

Use profiler-first layered analysis. Start from profiler evidence, deepen into `.bin` when the CSV-level view is not enough, and use IR as explanation and attribution rather than as the default entrypoint.

When compiler source analysis is enabled by the launch prompt or workspace guidance, treat it as a later escalation after profile and IR analysis. Use `triton-npu-analyze-compiler-source` only when this skill has narrowed the problem to a concrete performance-related compiler-side question that still needs source-backed explanation before the next operator change is clear.

Read [references/ascend-npu-profiling-analysis.md](references/ascend-npu-profiling-analysis.md) when the round needs deeper interpretation of `op_summary`, `task_time`, `api_statistic`, `msprof` JSON, or `.bin` signals.
Read [references/ascend-npu-optimization-guidance.md](references/ascend-npu-optimization-guidance.md) when you need help turning profiling symptoms and IR findings into concrete potential optimization points.
Read [references/ascend-npu-architecture-notes.md](references/ascend-npu-architecture-notes.md) when the likely optimization point depends on chip differences such as A3 versus A5 buffer sizes, layout behavior, or cube/vector data handoff.
Read [`../triton-npu-optimize-knowledge/references/symptom_index.md`](../triton-npu-optimize-knowledge/references/symptom_index.md) when structured profile or IR evidence is available and you need Triton/kernel-oriented symptom cards to narrow likely pattern directions before returning to detailed pattern references.
When the optimize target is `operator` and the evidence points to Torch NPU or framework-op behavior, also read [`../torch-npu-optimize-knowledge/references/pattern_index.md`](../torch-npu-optimize-knowledge/references/pattern_index.md) for operator-level pattern routing.

Read the references in this order:

1. profiling analysis, to understand the dominant profiler signals
2. optimization guidance, to turn those signals plus IR findings into potential optimization points
3. architecture notes, to adjust those optimization points for chip-specific constraints when needed

## Default workflow

1. Resolve the current round directory and round-local operator file.
2. Confirm the round has profile evidence.
   - Prefer round-local evidence such as `opt-round-N/profile/`.
   - If profile evidence is missing, collect it first through the existing profiling flow from [`../triton-npu-profile-operator/SKILL.md`](../triton-npu-profile-operator/SKILL.md).
3. Strongly consider spawning a subagent before the deep analysis phase.
   - Use this when profile or IR artifacts are large, or when the round already has a long `attempts.md`.
   - If context is still small enough, the current agent may continue directly.
4. Extract profile signals first.
   - Prefer the bundled summary helper in JSON mode:
     ```bash
      python3 ../triton-npu-run-eval/scripts/run-command.py profile-report --profile-dir <profile-dir> --format json
     ```
   - Use this as the default structured entrypoint for profiling evidence.
5. Interpret profiling evidence through the profiling reference instead of ad hoc guesses.
   - Follow [references/ascend-npu-profiling-analysis.md](references/ascend-npu-profiling-analysis.md) for layered signal interpretation.
   - Escalate into `.bin` when CSV-level evidence is still not explanatory enough.
6. Use [`../triton-npu-optimize-knowledge/references/symptom_index.md`](../triton-npu-optimize-knowledge/references/symptom_index.md) and the matching symptom cards to narrow the current hypothesis.
   - Start from the symptom index, then read only the one or two symptom cards under [`../triton-npu-optimize-knowledge/references/symptoms/`](../triton-npu-optimize-knowledge/references/symptoms/) that best match the extracted evidence.
   - When the optimize target is `operator` and the bottleneck looks Torch NPU or framework-op specific, also use [`../torch-npu-optimize-knowledge/references/pattern_index.md`](../torch-npu-optimize-knowledge/references/pattern_index.md) and the matching pattern cards under [`../torch-npu-optimize-knowledge/references/patterns/`](../torch-npu-optimize-knowledge/references/patterns/).
   - Use symptom cards as routing aids, not as a replacement for the underlying profile or IR evidence.
7. Decide whether profiler evidence is already sufficient on its own.
   - If the layered profiler signals already explain the likely operator problem well enough, continue to diagnosis.
   - If the profiler signals are suspicious but still not explanatory enough, capture or reuse IR under `opt-round-N/ir/`.
8. Extract IR performance signals as the second analysis path for explanation and attribution.
   - Prefer:
     ```bash
     python3 ../triton-npu-analyze-ir/scripts/inspect_ir.py performance-signals --ir-dir <ir-dir> --format json
     ```
   - Use `list-stages`, `stage-summary`, `find-changes`, or direct file inspection when the heuristic summary points to a specific stage or lowering symptom.
9. Compare with parent or baseline evidence when it already exists and is useful.
   - Do not block the round analysis if comparable evidence is missing.
   - Record missing comparison inputs as an evidence gap rather than guessing.
10. Write `opt-round-N/perf-analysis.md`.
   - Either `profile-only diagnosis` or `profile-plus-IR diagnosis` is acceptable, as long as the document makes the evidence path clear.

## Output contract

Write the analysis as a standalone document with these sections:

1. `# Round Performance Analysis`
2. `## Executive Summary`
3. `## Profile Signals`
4. `## Binary Signals`
5. `## IR Signals`
6. `## Diagnosis`
7. `## Operator Implementation Issues`
8. `## Optimization Suggestions`
9. `## Evidence Gaps`

Inside `## Profile Signals`, prefer these subsections when the evidence exists:

- `### Hotspots`
- `### Pipeline Ratios`
- `### Timeline And Wait`
- `### Host API Overhead`

Inside `## Diagnosis`, prefer these subsections:

- `### Operator Type Fit`
- `### Compute vs Memory Bound`
- `### Pipeline Bottlenecks`
- `### Memory Hierarchy Bottlenecks`
- `### Concurrency And Scheduling Bottlenecks`

## Required reasoning rules

- Treat profile evidence as the default required input.
- Treat `.bin` as a first-class deep-analysis path, not only as a niche fallback.
- Treat IR as optional but strongly preferred when profiler evidence alone does not explain the likely implementation problem.
- Use IR as explanation and attribution for profiler symptoms, not as the default entrypoint.
- Use profiling analysis and IR analysis together when one source alone cannot explain the performance problem confidently.
- Keep artifact ownership here even when IR evidence is used; `triton-npu-analyze-ir` is the IR evidence companion, not the owner of `perf-analysis.md`.
- Distinguish facts from inference.
- Cite the specific profile path, IR path, stage name, or operator name that supports each nontrivial conclusion.
- Do not stop at profiler or IR symptoms. The final diagnosis must point to likely problems in the current operator implementation.
- Keep optimization suggestions tied to those diagnosed implementation problems.
- Do not automatically write the analysis back into `attempts.md` or `summary.md`. `perf-analysis.md` is the formal output of this skill.

Use the profiling and optimization references for detailed signal interpretation and for mapping those signals to likely optimization points. Keep `SKILL.md` focused on workflow, evidence order, and output quality.
