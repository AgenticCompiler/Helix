# Ascend NPU Profiling Analysis Reference

Use this reference when `triton-npu-analyze-round-performance` needs deeper interpretation of profiler evidence for a Triton NPU operator.

The goal is not to restate every profiler field. The goal is to move from profiler symptoms to likely operator implementation problems.

## Evidence Order

Use the profiler in layers:

1. `op_statistic`
2. `op_summary`
3. `task_time`
4. `api_statistic`
5. `msprof` JSON
6. `.bin`
7. IR

Establish the profiler-backed diagnosis first. Use IR to explain or attribute it when needed.

## Layer 1: `op_statistic`

Use this layer to answer:

- Which operator is hottest?
- Is the target operator a dominant optimization target?
- Is the broad time shape scalar-heavy, vector-heavy, cube-heavy, or mixed?
- Are transfer-like operators already consuming too much time?

Useful signals:

- `OP Type`
- `Core Type`
- `Count`
- `Total Time(us)`
- `Avg Time(us)`
- `Ratio(%)`

This layer tells you where to look next. It rarely explains the full root cause by itself.

## Layer 2: `op_summary`

Use this layer to judge operator fit, pipeline shape, and likely bound type.

Important fields:

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

Typical readings:

- Vector-like operator but high scalar ratio:
  suspect degraded vectorization, too much scalar fix-up, overly conservative masking, or indexing that forced scalarized behavior.
- Cube-like operator but weak `aic_mac_ratio` or low `cube_utilization(%)`:
  suspect poor tile shape, weak data feeding, extra waits, or matrix path underutilization.
- High MTE ratios:
  suspect memory-bound behavior, frequent movement, or insufficient reuse.
- High `Task Wait Time(us)`:
  suspect weak overlap, stalls, scheduling gaps, or a producer-consumer imbalance.
- Strange `Block Dim` behavior:
  suspect parallelism or launch shape that does not fit the operator structure.

## Layer 3: `task_time`

Use this layer to inspect device-side sequencing.

Questions to ask:

- Are there large gaps between target tasks?
- Do tasks overlap as expected?
- Is the round showing serial regions that should have been overlapped?

Useful outputs:

- matched task count
- total task time
- task span
- total gap
- max gap
- overlap count

Large gaps are especially interesting when the operator should feed compute continuously.

## Layer 4: `api_statistic`

Use this layer to separate host-side overhead from device-side bottlenecks.

Look for:

- launch-heavy APIs
- compile or execute wrappers
- tiling or workspace preparation overhead
- synchronization-heavy APIs

Interpretation:

- Large host overhead can distort the round benchmark.
- Host overhead is still actionable when the current operator implementation is causing extra launches, extra shape handling, or expensive setup behavior.

## Layer 5: `msprof` JSON

Use this layer for timeline structure when CSVs are too flat.

Look for:

- number of stream-like tracks
- count of complete events
- apparent overlap
- serial regions

This layer is especially useful when `task_time` hints at weak concurrency but you want stronger evidence.

## Layer 6: `.bin`

Treat `.bin` as a first-class deep-analysis source when available.

### Block 0

Use it for base operator identity, duration, operator type, block dimension, and block detail preview.

### Block 1

Use it for pipe utilization.

Interpretation:

- low utilization in the expected dominant pipe suggests the operator is not feeding that pipeline well
- skewed utilization across blocks may indicate imbalance

### Block 2

Use it for instruction-level and wait breakdown.

Interpretation:

- high vector wait suggests stalls or weak feeding of vector compute
- large data-size movement with weak compute suggests transfer pressure

### Block 3

Use it for memory paths, bandwidth, and L2 signals.

Interpretation:

- suspicious memory-path bandwidth or request patterns can indicate poor movement structure
- low L2 hit ratio can indicate weak reuse or poor locality

### Block 4

Use it for memory workload tables and per-block load patterns.

Interpretation:

- skewed tables can indicate imbalance
- advice fields can help prioritize which path to inspect more closely

## Layer 7: IR

Use IR only after profiler evidence has already identified a suspicious behavior.

IR is most useful for confirming:

- degraded vectorization
- excess copy/load/store/sync operations
- weak overlap-friendly structure
- lowering changes that explain profiler regressions

## Diagnosis Patterns

### Vector operator with high scalar ratio

Likely implementation problems:

- indexing or masking pattern blocked vectorization
- edge handling forced too much scalar fix-up
- shape-specialization logic is too conservative

### Frequent data movement

Likely implementation problems:

- poor tile layout
- weak data reuse
- too many intermediate copies
- uncoalesced or fragmented access patterns

### Weak pipeline overlap

Likely implementation problems:

- load/store schedule is too serial
- producer and consumer stages are not interleaved well
- block or task organization does not expose enough concurrency

### Low cube utilization

Likely implementation problems:

- tile shape does not feed cube efficiently
- matrix path is starved by memory movement
- there is too much scalar or vector-side side work around the cube path

## Writing `perf-analysis.md`

The final analysis should connect:

1. observed profiler evidence
2. inferred bottleneck type
3. likely operator implementation issue
4. optimization suggestion tied back to that issue

Avoid stopping at statements like "memory-bound" or "scalar ratio is high." Always ask why the current operator implementation would cause that symptom.
