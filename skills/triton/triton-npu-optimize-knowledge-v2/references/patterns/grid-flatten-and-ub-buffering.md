---
priority: high
---

# Grid Flattening And UB Buffering Pattern

## Summary

Use this pattern when performance is limited by too many logical tasks, uneven per-core work, or tiny per-program transfers after a gather/scatter-style rewrite.

Core idea:
- map oversized logical grids onto physical cores more evenly (**grid flattening**),
- then batch per-core memory movement using small UB slabs (**UB buffering**).

This complements `program-multiple-rows`: that card widens row work *inside* one program, while this card reshapes logical-task-to-core mapping and per-core transfer batching.

## Use When

- Logical task count is far larger than physical core count.
- Work partitioning by batch/sequence causes visible imbalance.
- After a first rewrite, programs still move tiny contiguous chunks one row at a time.
- Grid/index decode overhead (`div/mod` recovery from flattened IDs) is nontrivial.

## Avoid When

- Workload is already near core count and flattening adds loop overhead.
- Destination/source continuity is weak enough that UB slab batching is not valid.
- Main bottleneck is still algorithm shape, scalar traps, or tiling fundamentals.

## Signals

### Code

- `TOTAL_TASKS >> NUM_CORES` style mapping.
- Program-level loops that process many tiny slices.
- Flattened pid decode chains that can be replaced by direct grid mapping.

### Profile

- Launch fragmentation, short bursts, and poor core utilization.
- Large Block Dim / too many thin programs despite contiguous transfer opportunities.

## Optimization Strategy

1. **Flatten logical tasks to physical cores** with uniform per-core work slices.
2. **Keep loop structure regular** (mask tails, avoid divergent early exits).
3. **Batch contiguous rows/chunks in UB** for wider reads/writes.
4. **Gate by size/continuity** so tiny or irregular cases can fallback.
5. **Validate launch feasibility first** (`coreDim`, grid limits), then parent-vs-parent performance.

## Common Repairs

### Logical-to-physical flattening

Distribute logical tasks with host-computed work-per-core and mask the tail.

### Direct grid remap

When natural axes exist (for example `(batch_head, row_block)`), prefer direct multidimensional grids over linearized pid decode.

### Runtime core discovery and task-kind-aware grid choice

Query runtime device properties before freezing one grid assumption:

```python
import torch

print(torch.npu.device_count())
device = torch.npu.current_device()
props = torch.npu.get_device_properties(device)
print(props)
```

If this path fails, if `torch.npu` is unavailable, or if the returned properties do not expose explicit counts, use:

- cube cores: `24`
- vector cores: `48`

Do not assume one observed physical-core count should become the default grid for every optimized kernel. Match the flattened grid to task kind first:

- `cube`-like operators: start from the available Cube-core count and validate with parent-vs-parent benchmarks.
- `vector`-like operators: start from the available Vector-core count and re-check after major transfer/tiling changes.
- `mix` operators: choose the first launch count from the dominant bottleneck side, and try both cube-first and vector-first counts if the profile is mixed.

Treat these counts as starting points for experiments, not immutable rules.

### UB aggregate writes

If each program produces multiple contiguous destination rows, stage a small row group in UB and issue wider stores.

### UB bulk reads

For repeated row-wise loads from contiguous sources, pull a small 2D slab into UB and reuse locally.

### Size-gated flattening

Use smaller per-program bundles for tiny shapes; keep larger bundles only where amortization is real.

### Simplified code sketch

```python
pid = tl.program_id(0)
tasks_per_core = tl.cdiv(TOTAL_TASKS, NUM_CORES)
task_begin = pid * tasks_per_core
task_end = tl.minimum(task_begin + tasks_per_core, TOTAL_TASKS)

for t in range(task_begin, task_end):
    row = t // ROW_CHUNK
    col0 = (t % ROW_CHUNK) * BLOCK_COL
    slab = tl.load(src_ptr + row * stride_row + col0 + tl.arange(0, BLOCK_COL))
    # ... optional local reuse ...
    tl.store(dst_ptr + row * stride_row + col0 + tl.arange(0, BLOCK_COL), slab)
```

## Failure Modes And Anti-signals

- **Flattening without launch-cap checks**: rewritten grids exceed hardware limits.
- **Flatten-only regressions**: restructuring alone can regress unless paired with later routing/reuse refinements.
- **Over-batching in UB**: slab size hurts occupancy or exceeds practical UB footprint.
- **Wrong continuity assumptions**: aggregate buffering applied to genuinely scattered addresses.

## Risks

- Poor offset decoding after flattening can destroy locality.
- Extra in-kernel loops can hurt small workloads.
- UB staging raises code complexity and branch/path interactions.

## What To Verify After Applying

- Grid dimensions stay within launch limits for all benchmarked shapes.
- Tail masks are correct when task count is not divisible by core count.
- Continuity assumptions for UB slabs are explicit and validated.
- Parent-vs-child benchmark results improve on the intended regimes.
- Profiling confirms fewer thin launches or better core utilization.

## Related Patterns

- `program-multiple-rows`: widen row work inside each program before/after flattening as needed.
- `tiling`: tune block sizes after mapping is structurally correct.
- `cache_use`: reduce duplicate passes and memory churn once mapping is stable.
- `autotune`: only after flattening/UB strategy is correct and measurable.
