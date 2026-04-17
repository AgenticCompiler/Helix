# Round Performance Analysis Deepening Design

## Summary

Deepen the existing `triton-npu-analyze-round-performance` skill so it becomes a general Triton NPU operator analysis workflow rather than a light round-summary helper. The upgraded workflow should treat profiler evidence as the primary entrypoint, include `.bin` analysis in the first-class evidence path, keep IR as an explanatory and attribution layer, and produce a more structured `opt-round-N/perf-analysis.md` that maps hardware-facing symptoms back to concrete operator implementation problems.

This design is informed by the profiling guide in `workspace/matmul/ascend-npu-profiling-analysis-guide.md` and by the sample `PROF_*` directory under `workspace/matmul/`.

## Goals

- Keep `triton-npu-analyze-round-performance` as the single round-level performance analysis skill.
- Make the workflow general for Triton NPU operators rather than special-casing one operator family.
- Expand the profiler analysis path beyond `op_statistic` into:
  - `op_summary`
  - `task_time`
  - `api_statistic`
  - `msprof` JSON timeline data
  - profiler `.bin` blocks
- Promote `.bin` analysis into the first-version main path rather than a niche optional fallback.
- Preserve the current requirement that the final diagnosis point to problems in the current operator implementation.
- Keep IR analysis as a secondary evidence layer that explains or confirms the profiler-derived diagnosis.
- Reuse and extend existing scripts rather than introducing a parallel analysis toolchain.

## Non-Goals

- Do not introduce a second round-analysis skill.
- Do not make the workflow matmul-only or cube-only.
- Do not fully automate the final diagnosis or optimization advice.
- Do not make `perf-analysis.md` mandatory for every optimize round.
- Do not move this analysis responsibility to the optimize supervisor.
- Do not require every profiling run to have every auxiliary artifact before any analysis can start.

## Current Problem

The current `triton-npu-analyze-round-performance` skill is useful for round-local analysis, but it still treats profiling at a relatively shallow level:

- `op_statistic` is used well enough for hotspot and core-type summaries.
- IR heuristics can identify vector/copy/sync-like symptoms.
- The final workflow can already produce `perf-analysis.md`.

But it still underuses several high-value profiler artifacts that are already present in real Ascend NPU profiling runs:

- `op_summary` contains operator-level pipeline and utilization fields such as `aic_mac_ratio`, `aic_scalar_ratio`, `aic_mte*`, `aiv_*`, `cube_utilization(%)`, `Block Dim`, and `Task Wait Time(us)`.
- `task_time` can show device-side sequencing, task gaps, and possible overlap issues.
- `api_statistic` can expose host-side launch, tiling, and synchronization overhead.
- `msprof` JSON can capture timeline structure that is difficult to infer from CSVs alone.
- `.bin` blocks expose deeper hardware and memory-path information that is valuable when CSV-level analysis is not sufficient.

As a result, the current skill is still more symptom-oriented than diagnosis-oriented.

## User-Facing Behavior

### Skill Identity

Keep the current skill name:

- `triton-npu-analyze-round-performance`

The enhanced skill should describe itself as a general round-level Triton NPU operator performance analyzer that:

- starts from profiler evidence
- escalates into deeper profiler artifacts when needed
- uses IR to explain or attribute suspicious behavior
- writes a standalone `opt-round-N/perf-analysis.md`

### Default Workflow

1. Resolve the round directory and operator file.
2. Require or collect profiler evidence first.
3. Analyze the profiler in layers instead of reading all artifacts blindly.
4. Include `.bin` analysis as part of the normal deep-analysis path.
5. Reuse or capture IR only when profiler evidence needs additional explanation or attribution.
6. Compare with parent or baseline evidence when it exists and is useful.
7. Write the final `perf-analysis.md`.

### Evidence Priority

The default evidence priority should be:

1. `op_statistic`
2. `op_summary`
3. `task_time`
4. `api_statistic`
5. `msprof` JSON
6. `.bin`
7. IR

This order does not mean IR is less important overall. It means the skill should first establish a profiler-backed diagnosis, then use IR to explain why the current operator implementation likely produced those profiler symptoms.

## Layered Analysis Model

### Layer 1: Hotspot And Target Confirmation

Primary input:

- `op_statistic_*.csv`

Purpose:

- identify the target operator
- confirm whether it is a dominant optimization target
- summarize operator-level runtime share, average cost, and call count
- identify broad core-type distribution

Expected outputs:

- hotspot ranking
- target operator identity
- scalar/vector/cube-oriented core-type summary

### Layer 2: Operator-Level Pipeline And Bound Analysis

Primary input:

- `op_summary_*.csv`

Purpose:

- determine whether the operator appears compute-bound, memory-bound, scalar-overhead-bound, or mixed
- identify pipeline bottlenecks
- capture task wait and block-dimension clues

Key fields include:

- `aic_mac_ratio`
- `aic_scalar_ratio`
- `aic_mte1_ratio`
- `aic_mte2_ratio`
- `aic_mte3_ratio`
- `aiv_vec_ratio`
- `aiv_scalar_ratio`
- `aiv_mte2_ratio`
- `aiv_mte3_ratio`
- `cube_utilization(%)`
- `Task Wait Time(us)`
- `Block Dim`

Expected outputs:

- operator-type guess or confirmation: `cube`, `vector`, or `mix`
- compute-vs-memory diagnosis candidate
- pipeline imbalance signals

### Layer 3: Timeline And Host Overhead Analysis

Primary inputs:

- `task_time_*.csv`
- `api_statistic_*.csv`
- `msprof_*.json`

Purpose:

- detect gaps, serial regions, or weak overlap in the runtime timeline
- identify host-side API overhead that may distort benchmark behavior
- distinguish device-side bottlenecks from launch or setup bottlenecks

Expected outputs:

- task gap or overlap signals
- host API overhead summary
- concurrency or scheduling concerns

### Layer 4: Binary Deep Analysis

Primary input:

- profiler `.bin` artifacts parsed through `parse_bin.py`

This layer is part of the first-version main path.

The skill should use structured signals derived from:

- Block 0: base operator identity and type
- Block 1: pipe utilization
- Block 2: instruction-level and wait breakdown
- Block 3: memory paths, bandwidth, and L2 signals
- Block 4: memory load patterns

Expected outputs:

- stronger pipeline diagnosis
- memory hierarchy bottleneck clues
- L2 and bandwidth interpretation
- operator-type-specific deep evidence

### Layer 5: IR Attribution

Primary input:

- `opt-round-N/ir/`

Purpose:

- explain why the profiler symptoms likely arise from the current lowering and operator structure
- confirm vectorization degradation, excessive copy/load/store/sync behavior, or weak overlap-friendly structure

Expected outputs:

- IR-backed attribution for profiler findings
- more precise mapping from hardware symptoms to code-structure problems

## General Operator Analysis, Not Matmul-Only

The enhanced skill should stay general for Triton NPU operators.

It may still branch its reasoning depending on inferred operator character:

- `cube`
- `vector`
- `mix`

But this should happen within a shared analysis framework instead of through separate per-operator workflows.

That means:

- cube-heavy operators should emphasize MAC ratio, cube utilization, and memory paths around matrix buffers
- vector-heavy operators should emphasize vector utilization, vector wait, UB-related transfer paths, and scalar overhead
- mixed operators should emphasize whether cube and vector activity are complementary or poorly overlapped

## Script Strategy

### Extend `profile_summary.py` Into A Unified Profiler Signal Extractor

`skills/triton-npu-profile-operator/scripts/profile_summary.py` should evolve from a narrow summary script into the main structured extractor for profiler evidence.

It should add or strengthen support for:

- structured hotspot summaries from `op_statistic`
- operator-type and bound guesses from `op_summary`
- pipeline and utilization summaries from `op_summary`
- timeline and wait summaries from `task_time`
- host overhead summaries from `api_statistic`
- binary-derived summaries by delegating to `parse_bin.py`
- JSON output as the preferred machine-readable format

It should still preserve a human-readable Markdown summary mode.

### Extend `parse_bin.py` As A Structured Deep-Signal Parser

`skills/triton-npu-profile-operator/scripts/parse_bin.py` should remain the owner of binary parsing.

Instead of only exposing raw or display-oriented block dumps, it should additionally expose stable structured summaries for:

- base operator info
- pipeline utilization
- vector wait and instruction breakdown
- memory path bandwidth and L2 signals
- memory load signals

It should not own final diagnosis or optimization advice.

### Keep `inspect_ir.py` As The IR-Side Signal Source

`skills/triton-npu-analyze-ir/scripts/inspect_ir.py` should remain responsible for IR heuristics and suspicious stage detection.

This design does not make IR the primary analysis source. IR remains the explanation and attribution layer after profiler-backed diagnosis has already been established.

## `perf-analysis.md` Contract Changes

Keep the existing top-level structure but deepen the content model.

The final document should include:

1. `# Round Performance Analysis`
2. `## Executive Summary`
3. `## Profile Signals`
4. `## Binary Signals`
5. `## IR Signals`
6. `## Diagnosis`
7. `## Operator Implementation Issues`
8. `## Optimization Suggestions`
9. `## Evidence Gaps`

### `Profile Signals`

This section should be organized into fixed subsections:

- `### Hotspots`
- `### Pipeline Ratios`
- `### Timeline And Wait`
- `### Host API Overhead`

### `Binary Signals`

This is a new first-class section.

It should summarize Block 0 through Block 4 in a way that is useful for diagnosis, not just raw reporting.

### `Diagnosis`

This section should be organized into stable dimensions:

- `### Operator Type Fit`
- `### Compute vs Memory Bound`
- `### Pipeline Bottlenecks`
- `### Memory Hierarchy Bottlenecks`
- `### Concurrency And Scheduling Bottlenecks`

### `Operator Implementation Issues`

The final diagnosis must still land on the current operator implementation.

Typical issue classes include:

- poor memory access organization
- excessive scalar, index, or boundary handling
- weak overlap or concurrency structure
- degraded vectorization-friendly layout
- excessive intermediate movement or writeback

## Reference Material Strategy

The guide from `workspace/matmul/ascend-npu-profiling-analysis-guide.md` should not be pasted wholesale into `SKILL.md`.

Instead:

- keep `SKILL.md` focused on workflow and decision points
- add a dedicated reference file under the skill, such as:
  - `skills/triton-npu-analyze-round-performance/references/ascend-npu-profiling-analysis.md`
- have `SKILL.md` explicitly tell the agent when to read that reference

This keeps the skill concise while preserving access to the deeper profiling framework.

## Testing

The implementation should add or extend tests for:

- `op_summary`-derived operator-type and bound summaries
- `task_time` timeline-gap summaries
- `api_statistic` host-overhead summaries
- `.bin` structured signal extraction
- `profile_summary.py` JSON output that includes the new evidence sections
- `triton-npu-analyze-round-performance` skill guidance updates

## Expected Outcome

After this change, the round analysis workflow should be able to:

- diagnose Triton NPU operators through a richer profiler-first framework
- use `.bin` analysis as a normal deep-analysis step
- distinguish hotspot identification from pipeline diagnosis, memory diagnosis, and attribution
- stay general across Triton NPU operators while still adapting its interpretation to cube/vector/mix behavior
- produce `perf-analysis.md` documents that are materially more actionable for operator optimization
