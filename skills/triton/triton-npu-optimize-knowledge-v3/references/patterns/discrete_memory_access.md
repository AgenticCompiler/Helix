# Discrete Memory Access Staging Pattern

## Summary

When the logical operation is index-driven (for example `out = x[idx]`), avoid direct per-element scattered global loads on the hot path. Stage contiguous source spans first, then select locally (for example with `tl.gather` from staged data).

This converts "discrete global memory access" into "contiguous movement + local selection", which often lowers scalar address overhead and improves effective memory behavior on Ascend NPU.

## Use When

- The central bottleneck is discrete indexed access rather than arithmetic.
- Index-driven global loads dominate runtime.
- Contiguous staging plus local selection is feasible for the active shapes.

## Avoid When

- Source spans are too large to stage efficiently.
- Access is already mostly contiguous and indexing is not the bottleneck.
- The primary issue is launch geometry or decomposition rather than access shape.

## Signals

### Code

- Hot loops repeatedly execute direct indexed global loads (`x[idx]` style).
- Per-lane index decode (`//`, `%`, address reconstruction) dominates surrounding math.
- One program could own contiguous rows/spans but current mapping is fully elementwise.

## Optimization Strategy

1. Reframe indexing into contiguous views where possible.
2. Stage contiguous spans from global memory.
3. Select indexed values from staged data.
4. Repair launch mapping if widened per-program work creates grid-limit pressure.
5. Validate parent-vs-parent and baseline correctness/perf.

## Detail

This example shows how to load data efficiently for discrete-memory-access workloads.

### Operation

Implement:

```python
out = x[idx]
```

Inputs:

| Input | Shape |
|-------|-------|
| x     | (M,)  |
| idx   | (N,)  |

Output:

| Input | Shape |
|-------|-------|
| out   | (N,)  |

### Key Difference Summary

- GPU-style path reads discrete values directly from global memory.
- NPU-optimized path stages contiguous source data, then selects indexed values locally.

### Detailed Difference

```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
+   rm = tl.arange(0, M)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

-   # GPU path
-   val = tl.load(x_ptr + idx * stride_x, mask=mask)  # Direct discrete global-memory access
+   # NPU path
+   x_shared = tl.load(x_ptr + rm * stride_x)  # [M] contiguous staging
+   val = tl.gather(x_shared, idx, 0)  # local indexed selection

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

## Failure Modes And Anti-signals

- Over-staging large ranges hurts occupancy/on-chip footprint.
- Initial contiguous remap can violate launch limits before grid repair.
- Wrong-priority application yields small/no gains if bottleneck is elsewhere.

## What To Verify After Applying

- Boundary and index-extreme correctness.
- Launch geometry remains valid for target hardware.
- Parent-vs-child and baseline performance on same harness.
- Profile confirms reduced scalar decode pressure or fewer expensive scattered loads.

## Related Patterns

- `gather-load`
- `layout-store-and-block-pointers`
- `scalar-latency-traps`
- `program-multiple-rows`
