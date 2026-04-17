---
name: triton-npu-analyze-round-performance
description: Use when an optimize round needs deep performance diagnosis from round-local profile and optional IR evidence, especially to explain scalar/vector/cube imbalance, frequent data movement, weak pipeline overlap, or other signals that should be traced back to problems in the current operator implementation.
---

# Analyze Optimize Round Performance

Diagnose one `opt-round-N/` at a time and write the result to `opt-round-N/perf-analysis.md`.

This skill is for deep round analysis inside `triton-npu-optimize`, not for supervisor audits and not for whole-session summaries.

Use two complementary analysis paths to find performance problems:

- profiling analysis to identify where time is going and which hardware-facing symptom dominates
- IR analysis to explain why that symptom appears in the current lowering and operator structure

Use profiler-first layered analysis. Start from profiler evidence, deepen into `.bin` when the CSV-level view is not enough, and use IR as explanation and attribution rather than as the default entrypoint.

Read [references/ascend-npu-profiling-analysis.md](references/ascend-npu-profiling-analysis.md) when the round needs deeper interpretation of `op_summary`, `task_time`, `api_statistic`, `msprof` JSON, or `.bin` signals.
Read [references/ascend-npu-optimization-lessons.md](references/ascend-npu-optimization-lessons.md) when you need higher-level heuristics that connect profiling symptoms, IR findings, chip constraints, and likely optimization directions.

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
     python3 ../triton-npu-profile-operator/scripts/profile_summary.py <profile-dir> --format json
     ```
   - Use this as the default structured extractor for `op_statistic`, `op_summary`, `task_time`, `api_statistic`, `msprof` JSON, and optional `.bin` evidence.
5. Analyze the profiler in layers instead of flattening all artifacts together.
   - Layer 1: `op_statistic` for hotspots and scalar/vector/cube-oriented time distribution.
   - Layer 2: `op_summary` for `aic_*`, `aiv_*`, `cube_utilization(%)`, `Task Wait Time(us)`, and `Block Dim`.
   - Layer 3: `task_time`, `api_statistic`, and `msprof` JSON for task gaps, host overhead, and weak overlap.
   - Layer 4: `.bin` for deeper pipeline, wait, bandwidth, L2, and memory-path signals.
6. Decide whether profiler evidence is already sufficient on its own.
   - If the layered profiler signals already explain the likely operator problem well enough, continue to diagnosis.
   - If the profiler signals are suspicious but still not explanatory enough, capture or reuse IR under `opt-round-N/ir/`.
7. Extract IR performance signals as the second analysis path for explanation and attribution.
   - Prefer:
     ```bash
     python3 ../triton-npu-analyze-ir/scripts/inspect_ir.py performance-signals --ir-dir <ir-dir> --format json
     ```
   - Use `list-stages`, `stage-summary`, `find-changes`, or direct file inspection when the heuristic summary points to a specific stage or lowering symptom.
8. Compare with parent or baseline evidence when it already exists and is useful.
   - Do not block the round analysis if comparable evidence is missing.
   - Record missing comparison inputs as an evidence gap rather than guessing.
9. Write `opt-round-N/perf-analysis.md`.

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
- Distinguish facts from inference.
- Cite the specific profile path, IR path, stage name, or operator name that supports each nontrivial conclusion.
- Do not stop at profiler or IR symptoms. The final diagnosis must point to likely problems in the current operator implementation.
- Keep optimization suggestions tied to those diagnosed implementation problems.
- Do not automatically write the analysis back into `attempts.md` or `summary.md`. `perf-analysis.md` is the formal output of this skill.

## Signals to look for

- Scalar-heavy time in a round that should mainly benefit from vector or cube execution
- Transfer-heavy hotspots such as copy, DMA, or repeated load/store behavior
- High `Task Wait Time(us)` or weak utilization in `op_summary`
- Suspicious `aic_*`, `aiv_*`, or `cube_utilization(%)` ratios that do not fit the expected operator type
- Task gaps, serial regions, or weak overlap in `task_time` and `msprof` JSON
- Host launch, tiling, workspace, or synchronization overhead in `api_statistic`
- Memory-path, bandwidth, L2, vector wait, or memory-load anomalies in `.bin`
- Vector-related IR patterns disappearing or staying unexpectedly weak
- Extra synchronization, waiting, or barrier-heavy stages
- Weak overlap between transfer-heavy and compute-heavy stages
- Suspicious parent-vs-round differences when comparison evidence is available

## Typical conclusions

Useful conclusions usually land on implementation problems such as:

- vectorization-friendly structure was degraded by the current indexing or masking pattern
- data movement is too frequent because the access pattern or tile layout is poor
- scalar fix-up work is dominating because boundary handling is too conservative
- the current load/store organization is breaking coalescing or reuse
- the pipeline is too shallow or insufficiently overlapped for the current operator structure

Be explicit when a conclusion is still heuristic and what extra evidence would strengthen it.
