# Discrete Memory Access Staging Pattern

## Summary

When the logical operation is index-driven (`out = x[idx]`-style), avoid per-element scattered global loads on the hot path. Stage contiguous source spans first, then select locally (for example with gather/select from staged data).

This pattern converts "discrete global memory access" into "contiguous movement + local selection," which often lowers scalar address overhead and improves effective memory behavior.

## Use When

- The kernel is dominated by index-driven reads from global memory.
- Workloads have meaningful contiguous structure (for example large `inner_size` spans) even if the API is gather-like.
- Profiling shows scalar-heavy index decode (`//`, `%`, pointer reconstruction) around the data movement loop.

## Avoid When

- Source ranges are too large to stage efficiently for the active branch.
- Accesses are already naturally contiguous and direct loads are not the bottleneck.
- The main issue is launch geometry or kernel decomposition rather than index access shape.

## Signals

- Per-element global gather loads with repeated coordinate decode.
- Hot kernels where integer address math dominates alongside sparse-looking loads.
- Cases where one program can own a contiguous row/span and loop locally.

## Optimization Strategy

1. **Reframe indexing into contiguous views** (for example flattened `[outer, axis, inner]` style layouts).
2. **Stage contiguous spans** into local working data.
3. **Select via local gather/indexing** instead of direct scattered global reads.
4. **Repair launch geometry** to stay within hardware grid limits after widening per-program work.
5. **Validate against parent and baseline** with correctness first.

## Common Repairs

### Replace per-lane decode with row/span ownership

Assign one program a contiguous row/span and iterate inner blocks locally instead of decoding full-rank coordinates for every element.

### Add launch-cap-safe looping

If the first contiguous rewrite creates excessive grid width, keep one program per logical row/span and loop over inner chunks inside the kernel.

### Keep fallback for noncontiguous regimes

Use dispatch when only some regimes benefit from staged contiguous access.

### Simplified code sketch

```python
# Stage contiguous span first.
span = tl.load(src_ptr + row_base + tl.arange(0, INNER_BLOCK))

# Convert discrete global gathers into local indexed selection.
local_idx = tl.load(index_ptr + out_base + tl.arange(0, OUT_BLOCK))
vals = tl.gather(span, local_idx)
tl.store(out_ptr + out_base + tl.arange(0, OUT_BLOCK), vals)
```

## Failure Modes And Anti-signals

- **Over-wide initial grid** after rewrite violates launch limits and must be repaired.
- **Over-staging** large ranges can waste on-chip memory and hurt occupancy.
- **Wrong problem choice**: if bottleneck is elsewhere (tiling/dispatch/scalar traps), staging alone gives weak gains.

## Risks

- More complex pointer/view logic can introduce correctness bugs at boundaries.
- Extra staging may increase temporary footprint.
- Benefit may be strongly shape-dependent.

## What To Verify After Applying

- Correctness across boundary shapes and index extremes.
- Launch geometry remains within hardware limits.
- Parent-vs-child and baseline performance on the same harness.
- Profile confirms reduced scalar index-decode pressure or fewer scattered global reads.

## Related Patterns

- `gather-load`: complementary card when gather semantics are explicit and layout/store interactions dominate.
- `layout-store-and-block-pointers`: use to regularize address shape before or after staging rewrite.
- `scalar-latency-traps`: use when decode arithmetic remains a dominant bottleneck.
- `program-multiple-rows`: combine when contiguous span ownership benefits from wider per-program batching.
