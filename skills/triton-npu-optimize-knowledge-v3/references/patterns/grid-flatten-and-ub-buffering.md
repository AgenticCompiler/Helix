# Grid Flattening And UB Buffering Pattern

## Summary

Use this pattern when latency is dominated by oversized logical grids, uneven per-core work, or tiny per-program transfers after gather/scatter-style rewrites.

Core idea:
- flatten logical tasks onto physical cores with more uniform work,
- then batch per-core movement using small UB slabs.

This complements `program-multiple-rows`: that pattern widens row work inside a program, while this one reshapes logical-task-to-core mapping and per-core transfer batching.

## Use When

- Logical task count is far larger than physical core count.
- Batch/sequence partitioning causes visible load imbalance.
- Programs still process tiny contiguous rows/chunks one-at-a-time.
- Grid/index decode overhead is nontrivial.

## Avoid When

- Workload is already near physical core count.
- Continuity is too weak for safe UB slab batching.
- Main bottleneck is still algorithm structure, scalar traps, or base tiling.

## Signals

### Code

- `TOTAL_TASKS >> NUM_CORES` style mapping.
- Many thin launches or highly fragmented per-program work.
- Flattened pid decode chains that can be replaced by direct multidimensional mapping.

### Profile

- Poor core utilization with short bursts.
- Launch-side overhead remains high after first structural rewrites.

## Repairs

### Flatten logical tasks

Treat independent logical work as one stream and split evenly across physical cores:

```python
pid = tl.program_id(0)
task_start = pid * TASKS_PER_CORE
offs = task_start + tl.arange(0, TASKS_PER_CORE)
mask = offs < TOTAL_TASKS
```

Compute `TASKS_PER_CORE` on host and pass as `tl.constexpr` when it controls loop/tensor shape. Keep `NUM_TASKS` / `NUM_CORES` explicit in launch contracts so bounds and tails remain clear.
Prefer uniform masked loops over per-core early-return branches to avoid reintroducing control imbalance.

### Map logical grid to physical grid

When logical tasks exceed core count, launch `NUM_CORES` programs and loop logical tasks inside each program.
Keep loop bounds simple/explicit (`NUM_TASKS`, `NUM_CORES`) so compilers can simplify tail handling.

### UB aggregate writes

If one program emits several contiguous destination rows, stage a small row group in UB and issue wider stores.
Do not use aggregate writes for genuinely scattered destinations.

### UB bulk reads

For repeated row-wise reads from contiguous source rows, load a small 2D slab into UB and reuse locally.
This is especially useful for gradient/accumulation-like paths where many neighboring source rows are consumed repeatedly. Keep a fallback for non-continuous row indices.

### Size-gated flattening

Use smaller per-program bundles for small shapes and larger bundles only where amortization is measurable.

## Dependencies

- Apply only after basic access semantics are correct.
- UB read/write batching usually depends on prior flattening that gives each program multiple rows.
- Combine with `autotune` only after structure is stable.
- When autotuning, separate structural knobs (`TASKS_PER_CORE`, block decomposition) from micro knobs (`SUB_BLOCK_SIZE`, warp/stage) so causal attribution stays clear.

## Risks

- Flattening can destroy locality if offsets are decoded poorly.
- In-kernel loops can regress small workloads.
- UB slab sizes can exceed practical footprint or hurt occupancy.
- Gather/scatter continuity assumptions differ; do not reuse batching blindly.

## What To Verify After Applying

- Physical core assumptions are explicit.
- Tail masking is correct when task count is not divisible by core count.
- Launch dimensions stay within hardware limits.
- Parent-vs-child improvements hold on intended regimes.
- If wins depend on row/source continuity, record that condition in round summaries.

## Related Patterns

- `program-multiple-rows`
- `tiling`
- `cache_use`
- `autotune`
