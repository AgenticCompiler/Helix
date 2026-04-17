# Ascend NPU Optimization Lessons

Use this reference when `triton-npu-analyze-round-performance` needs higher-level heuristics that connect profiling evidence, IR evidence, and architecture constraints.

This document does not replace the profiling reference. It adds practical rules for deciding what matters, what to compare, and how to move from diagnosis to optimization direction.

## Core Mental Model

Use two complementary analysis paths:

- profiling analysis tells you where time is going and which hardware-facing symptom dominates
- IR analysis tells you why the current lowering and operator structure create that symptom

The usual flow is:

1. find the hotspot in profiling
2. identify the bottleneck from ratios, waits, bandwidth, and utilization
3. inspect IR to explain the bottleneck
4. map the result back to a concrete operator implementation problem

## Hotspot Selection

### Focus on ratio before absolute detail

If one operator owns almost all runtime in `op_statistic`, optimize that operator first.

Practical rule:

- when `Ratio(%)` is overwhelmingly concentrated in one operator, other operators are usually not worth investigating yet

Do not spend deep-analysis time on non-hot operators when the round is clearly dominated by one target.

## Profiling Heuristics

### Ratio fields are the fastest diagnosis signal

In `op_summary`, the highest ratios usually tell you which pipeline family is limiting performance.

Examples:

- high `aic_mac_ratio`: cube compute is active
- high `aic_mte2_ratio`: data movement from slower memory is dominant
- high `aic_scalar_ratio`: scalar control work is too heavy
- high `aic_fixpipe_ratio`: writeback or post-processing may need attention

Do not overcomplicate the first read. Start by asking which ratio is dominant.

### Block Dim is often more important than micro-optimizations

`Block Dim` tells you how much parallelism the kernel is exposing to AI cores.

If the chip can use many cores but `Block Dim` is very small, under-parallelization may dominate the entire performance story.

Questions to ask:

- is the operator exposing enough blocks?
- did the new round accidentally reduce parallelism?
- is the shape too small to drive many cores?

### `.bin` is the deeper truth when CSVs look normal but performance is still poor

Use `.bin` when aggregated CSV ratios look acceptable but the benchmark is still slower than expected.

High-value checks:

- low L2 hit ratio: reuse or tiling may be poor
- strong GM to L2 traffic without strong downstream reuse: data may be fetched repeatedly
- strong imbalance across vector lanes or blocks: load balance may be poor

### Cross-round comparison should stay simple at first

When comparing two rounds, start from a very small set of metrics:

- `Total Time(us)` or benchmark outcome
- compute-side utilization improving or regressing
- transfer-side pressure improving or regressing

Only widen the comparison when these first checks are inconclusive.

## IR Heuristics

### IR is the bridge between profiling and optimization

Use IR to answer why the profiler symptom appears.

Common mappings:

- high memory-transfer ratios:
  check whether `nd2nz`, `load`, `store`, or copy-like operations are too frequent
- high scalar ratio:
  check whether branching, masking, or shape-handling logic is too heavy
- low cube utilization:
  check tile-related lowering, block structure, and whether the compute path is underfed
- fixpipe-heavy behavior:
  check whether post-processing or layout conversion is fused effectively

### “Profiling says slow, IR says why”

Keep this split clear:

- profiling locates the bottleneck
- IR explains the lowering and code-structure reason

Do not use IR as a substitute for hotspot selection.

## Architecture-Aware Heuristics

### Chip differences matter

Different Ascend chips can reward different optimization choices.

Before drawing strong conclusions, confirm the target chip and avoid assuming that a lesson from one chip transfers directly to another without adjustment.

Examples of architecture-sensitive areas:

- available fusion behavior
- layout optimization support
- buffer sizes
- data handoff between cube and vector paths

### Tiling is bounded by buffer capacity, not by “bigger is always better”

Larger tiles help only when they fit the relevant on-chip buffers and improve reuse without destroying parallelism.

When performance regresses after a code change, consider whether the compiler selected a worse tiling strategy.

Useful checks:

- tile-related IR changes
- `Block Dim` changes
- compute utilization dropping while transfer pressure rises

### Layout conversions are a real cost

If the kernel repeatedly switches between layouts needed by different execution paths, layout conversion can become a hidden bottleneck.

Look for repeated conversion-related operations in IR and movement-heavy symptoms in profiling.

## Host-Side Heuristics

Do not assume the operator kernel is always the real bottleneck.

If `api_statistic` shows large launch or synchronization overhead, the round may be losing time on:

- too many launches
- excessive synchronization
- expensive setup behavior

In that case, operator optimization alone may not fix the measured slowdown.

## Shape Heuristics

Small shapes often have inherently poor efficiency.

When the workload is too small, fixed movement and launch costs can dominate even if the kernel is otherwise reasonable.

Questions to ask:

- is the shape too small for cube-heavy optimization to pay off?
- would batching or fusion help more than micro-optimizing one kernel?
- is low `Block Dim` a shape-driven limitation rather than a code bug?

## Common Regression Pattern

If a change makes the kernel slower, one common cause is not “the idea was wrong” but “the new lowering produced a worse tiling or parallelism shape.”

Check:

- tile-related IR differences
- `Block Dim`
- compute-side ratios
- transfer-side ratios

## Writing Conclusions

The final conclusion should sound like this:

- observed profiling symptom
- supporting IR explanation
- likely implementation problem
- optimization direction that follows from both

Avoid conclusions that stop at “memory-bound” or “scalar ratio is high.” Those are intermediate diagnoses, not the final answer.
