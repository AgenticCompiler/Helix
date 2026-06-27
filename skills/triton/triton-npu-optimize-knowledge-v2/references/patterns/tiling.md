# Hierarchical Tiling Optimization Pattern

## Summary

Use this pattern to reduce per-program working-set size so tiles, intermediates, and multi-tensor live state fit Unified Buffer (UB) safely.

The goal is to preserve the kernel's high-level structure while introducing a second level of chunking that caps peak memory footprint.

## Use When

- Block sizes, live intermediates, or multi-tensor loads risk UB overflow.
- Kernel structure is already mostly correct, but tile footprint is too large for stable execution.
- Performance cliffs appear when increasing tile widths that should otherwise help throughput.
- A two-level scheme (`BLOCK` for scheduling, `SUB_BLOCK` for memory safety) can be applied without changing semantics.

## Avoid When

- UB pressure is not the bottleneck and sub-block loops would only add control overhead.
- The kernel still needs foundational structural rewriting (for example manual reduction should first become regular tiled `tl.dot`).
- Main issue is memory/compute overlap after footprint is already safe (use `software-pipeline` next).
- Access/layout shape is the primary problem rather than working-set size.

## Signals

### Code

- Large block sizes keep too many tensors live in one iteration.
- Temporary tensors have near-output shape and overlap in lifetime.
- Runtime failures or unstable behavior occur when tile size is widened.

### Profile

- Strong regressions or failures at larger tile configurations due to memory pressure.
- Throughput does not scale with bigger tiles because memory footprint dominates.

## Repairs

### Introduce two-level tiling

Keep a larger outer block for launch/task geometry, and process it through inner sub-block loops to cap live memory.

### Bound sub-block live state

Load, compute, and store one sub-block at a time so peak UB residency stays below limits.

### Tune sub-block size by operation complexity

Use smaller sub-blocks when multiple operands/intermediates are live; use larger sub-blocks when arithmetic is simple and UB headroom exists.

### Preserve alignment and predictable access

Choose sub-block shapes that keep contiguous access and alignment-friendly transfer patterns.

### Simplified code sketch

```python
@triton.jit
def kernel(x_ptr, y_ptr, n_elements,
           BLOCK: tl.constexpr, SUB_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK
    n_sub = tl.cdiv(BLOCK, SUB_BLOCK)
    for s in range(n_sub):
        offs = base + s * SUB_BLOCK + tl.arange(0, SUB_BLOCK)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = some_op(x)  # Same math as unsliced path.
        tl.store(y_ptr + offs, y, mask=mask)
```

## Synthesized Guidance

- Apply hierarchical tiling when memory capacity is the blocker, before chasing overlap or micro-optimizations.
- Start with the smallest change that eliminates UB overflow, then retune tile ladders for performance.
- Prefer shape-gated tile policies over one global tile size when workload regimes differ significantly.
- If capacity issues are resolved but wait gaps dominate, transition to `software-pipeline` and overlap tuning.

## Related Patterns

- `software-pipeline`
- `program-multiple-rows`
- `layout-store-and-block-pointers`
- `classic-matmul`

## What To Verify After Applying

- Correctness across full-size and boundary/tail cases.
- No UB overflow or memory-pressure faults on representative largest workloads.
- Kernel and host launch signatures stay consistent with new tile parameters.
- Parent-vs-child benchmarks show net gains or at least stable performance with capacity issues removed.
