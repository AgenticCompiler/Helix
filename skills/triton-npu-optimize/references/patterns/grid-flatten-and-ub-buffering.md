## Summary

Change work distribution and UB staging when latency is dominated by too many logical tasks, uneven per-core work, or tiny row-wise memory transfers after a gather/scatter style rewrite.

This pattern complements `program-multiple-rows`: that pattern widens row-wise work inside a program, while this one focuses on flattening logical work onto physical cores and then batching memory movement inside each core.

## When To Use

- The logical grid is much larger than the physical AICore or VectorCore count.
- Work is partitioned by batch or sequence buckets with visible load imbalance.
- Each program processes many tiny rows after grid-to-physical-core mapping.
- Gather-like code has continuous destination rows but still stores one row at a time.
- Scatter-weight-gradient-like code has repeated row loads that can be batched from continuous source rows.

## Repairs

### Flatten logical tasks

Treat all independent logical work items as one continuous stream and split that stream evenly across physical cores:

```python
pid = tl.program_id(0)
task_start = pid * TASKS_PER_CORE
offs = task_start + tl.arange(0, TASKS_PER_CORE)
mask = offs < TOTAL_TASKS
```

Compute `TASKS_PER_CORE` on the host and pass it as `tl.constexpr` when it controls tensor shapes or loop bounds. Prefer uniform per-core loop structure and masks over early returns that create uneven control flow.

### Map logical grid to physical grid

When `logical_tasks > num_cores`, launch `num_cores` programs and loop over logical tasks inside the kernel. Keep `NUM_TASKS` and `NUM_CORES` as explicit constants so the compiler can simplify loop bounds.

This helps only when each physical program has enough work to amortize the internal loop. If the original grid is near the physical core count, the extra loop can be overhead.

### UB aggregate writes

After physical-core mapping, if each program produces several rows with continuous destination addresses, stage a small group of rows in UB and issue a wider 2D store. Choose `SUB_BLOCK_SIZE` so the staged rows plus live compute tensors fit UB.

Do not use aggregate writes for genuinely scattered destinations.

### UB bulk reads

For row-wise reductions or gradient accumulation where several consecutive source rows are loaded one at a time, load a small 2D slab into UB and select rows from the slab. Keep a fallback path if row indices are not continuous enough for a bulk read to be correct.

## Dependencies

- Apply `grid-flatten-and-ub-buffering` only after the kernel's basic access semantics are clear.
- UB aggregate write and UB bulk read usually depend on a prior grid-to-physical-core or flattening rewrite that gives each program multiple rows.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` separately enough to explain the result.

## Risks

- Flattening can erase useful multidimensional locality if offsets are decoded poorly.
- Physical-grid loops can hurt small workloads.
- UB staging can exceed UB capacity or reduce occupancy.
- Gather and scatter have different address-continuity requirements; do not reuse aggregate-write logic for scattered stores.

## Verification Checklist

- Record the physical core assumption and how it is discovered or passed.
- Validate edge cases where `TOTAL_TASKS` is not divisible by `NUM_CORES`.
- Benchmark small and large shapes if the operator supports both.
- If the win depends on row continuity, add that condition to the round summary.
