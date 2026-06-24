# Program Multiple Rows Pattern

## Summary

Map multiple logical rows to one Triton program (`BLOCK_M > 1`) to amortize per-program overhead and improve vector utilization in row-structured kernels.

This is a program-granularity pattern: fewer, heavier programs doing more row work per launch.

## Use When

- Kernel is naturally row-wise (row reductions, row-wise fused epilogues, row-major transforms).
- Current launch maps one row per program and profiling shows many thin programs or scalar-heavy overhead.
- Inner-dimension streaming over `N` can remain single-pass while widening row count.
- Row count is large enough to amortize wider per-program bundles.

## Avoid When

- Row count is tiny and wider bundles cannot amortize setup.
- Increasing `BLOCK_M` introduces second full passes or unstable numeric behavior.
- Main bottleneck is elsewhere (layout/store shape, algorithm structure, unrelated scalar traps).
- Ping-pong/multibuffer variants are introduced without clear MTE-vector overlap evidence.

## Signals

### Code

- `program_id(0)` maps directly to one row.
- Repeated per-row pointer/control setup dominates loop body.
- Inner-dimension tiling exists (`BLOCK_N`), but row axis remains under-batched.

### Profile

- Scalar/control pressure stays high with one-row programs.
- Moderate row batching gives clear gains, but over-widening regresses.
- Useful cues include `aiv_scalar_ratio`, `aiv_mte2_ratio`, and `op_statistic` Avg/Count deltas; treat `BAR` cycles as diagnostic context, not a success metric by itself.
- Barrier/wait growth with many short programs is a common indicator that row granularity is too fine.

Profiler interpretation notes:

- `op_statistic` Avg should be compared on matched shapes/workload; Count changes can otherwise hide regressions.
- If `aiv_mte2_ratio` dominates while scalar ratio is low, row batching may be secondary to transfer/layout levers.
- If scalar ratio remains high after moderate `BLOCK_M` increases, combine with scalar-control cleanups rather than widening blindly.

## Optimization Strategy

1. Introduce `BLOCK_M > 1` and remap row ownership to row blocks.
2. Keep one-pass inner-dimension streaming when possible.
3. Tune `BLOCK_M` progressively with parent-vs-parent checks.
4. Add shape/dtype gates when one global `BLOCK_M` regresses some regimes.
5. Compose with inner-tile and launch-parameter tuning only after each row-batching step is validated.

## Implementation sketch (Triton)

```python
pid_m = tl.program_id(0)
rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
row_mask = rows < B

for n0 in range(0, N, BLOCK_N):
    cols = n0 + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    vals = tl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :] * stride_xn,
                   mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    # row-wise reduction on axis=1, or row-wise fused compute...

tl.store(y_ptr + rows * stride_ym, out_vals, mask=row_mask)
```

This structure also matches common row-wise LSE and fused row-epilogue kernels where per-row running state (`m`, `s`) is carried across the `N` loop.

## Failure Modes And Anti-signals

- Assuming larger `BLOCK_M` is monotonic; it often is not.
- Applying one wide-row setting globally and regressing small/short regimes.
- Introducing a second full pass while widening rows.
- Treating PMR as universal even when another primary lever dominates.

## Risks

- Wider row bundles increase temporary footprint and scheduling pressure.
- Gated dispatch adds maintenance complexity.
- Multi-lever edits can hide regressions without strict parent comparisons.

## What To Verify After Applying

1. Correctness on boundary rows and tail masks.
2. Parent-vs-child benchmark improvement in each representative regime (prefer the project compare-perf authority when available).
3. Fewer launches/programs for intended shapes.
4. No unintended extra global passes.
5. Dispatch gates route correctly for tiny vs large regimes.

## Related Patterns

- `grid-flatten-and-ub-buffering`
- `tiling`
- `parallel`
- `layout-store-and-block-pointers`
- `software-pipeline` (overlap tuning after row granularity is already chosen)
