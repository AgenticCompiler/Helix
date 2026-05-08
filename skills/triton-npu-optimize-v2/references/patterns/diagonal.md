# Diagonal Block Traversal Pattern

## Summary

Use this pattern when tiled matrix-style kernels suffer from cache contention caused by traversal order, not by missing tiling.

Diagonal traversal reorders block execution so concurrent programs are less likely to hammer the same cache regions at once, which can improve effective reuse and reduce conflict pressure.

## Use When

- Large matrix-style workloads already use sensible tile shapes but still show locality/conflict issues.
- Many programs touch similar source regions concurrently under row-major or simple swizzle traversal.
- Both primary axes span enough blocks that traversal order materially affects reuse behavior.
- The bottleneck looks like cache/traffic scheduling rather than arithmetic throughput.

## Avoid When

- Problem size is small enough that traversal order has little impact.
- Kernel is still missing first-order tiling/layout fixes.
- Overhead of complex traversal mapping outweighs expected locality gains.
- The dominant bottleneck is scalar control, UB capacity, or launch geometry unrelated to cache-region contention.

## Signals

### Code

- Work assignment is row-major/horizontal and causes repeated synchronized access to the same matrix regions.
- Tile math is already stable, but performance remains sensitive to block launch ordering.
- One operand has large footprint where eviction/reload behavior is likely under naive traversal.

### Profile

- Throughput varies with scheduling order despite similar arithmetic work.
- Signs of memory-system contention or reuse loss persist after tile-size tuning.
- Performance degrades as matrix block grid grows, even when per-block kernel math is unchanged.

## Repairs

### Replace naive block order with diagonal progression

Map block coordinates so neighboring program IDs sweep diagonally across block space rather than row-by-row.

### Gate diagonal mode by grid size

Use thresholds so diagonal traversal activates only when block-grid dimensions are large enough to benefit.

### Keep tile math unchanged

Change traversal/scheduling order first; avoid mixing in unrelated math/layout rewrites during initial validation.

### Tune diagonal partition granularity

Adjust diagonal region size to balance mapping overhead against reduced contention.

### Simplified code sketch

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  DIAG_THRESHOLD: tl.constexpr):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    num_blocks = num_m * num_n

    for block_idx in range(pid, num_blocks, tl.num_programs(0)):
        # Row-major fallback.
        m_idx = block_idx // num_n
        n_idx = block_idx % num_n

        # Diagonal remap for large grids.
        if num_m >= DIAG_THRESHOLD and num_n >= DIAG_THRESHOLD:
            d = block_idx % DIAG_THRESHOLD
            q = block_idx // DIAG_THRESHOLD
            m_idx = (q + d) % num_m
            n_idx = (q + 2 * d) % num_n

        # Tile math remains unchanged; only traversal order changes.
        m_start = m_idx * BLOCK_M
        n_start = n_idx * BLOCK_N
        # ... load A/B tiles, accumulate tl.dot over K, store C tile ...
```

## Synthesized Guidance

- Apply this pattern after basic tiling is already correct; it is a scheduling-order optimization.
- Start with a simple thresholded diagonal mode and compare against baseline traversal on large shapes.
- Use it when contention/reuse signals persist after tile-size and block-shape tuning.
- If diagonal ordering helps only marginally, shift attention to layout/store or pipeline overlap patterns.

## Related Patterns

- `tiling`
- `layout-store-and-block-pointers`
- `software-pipeline`
- `compile_hint`

## What To Verify After Applying

- Numerical results are unchanged across all tested shapes.
- Block-to-coordinate mapping covers full output domain with no gaps/overlaps.
- Large-shape benchmarks show measurable improvement over row-major traversal.
- Mapping overhead does not regress small/medium shape performance beyond acceptable limits.
