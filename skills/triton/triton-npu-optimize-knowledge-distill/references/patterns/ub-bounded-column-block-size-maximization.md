# UB-Bounded Column Block Size Maximization

## Summary

For 2D-tiled elementwise kernels, maximize the column block size up to the Unified Buffer capacity limit. Larger column blocks reduce inner-loop trip counts and amortize per-iteration overhead (pointer math, mask construction, loop control). Set per-kernel block-size caps from the number of simultaneously live tensors in each kernel — a kernel with fewer live tensors tolerates a larger safe block size.

## Use When

- The kernel uses 2D tiling with separate row and column tile dimensions.
- The operation is elementwise or row-wise with no inter-element reduction or cross-column dependency.
- An inner column loop iterates over blocks of BLOCK_N, and per-iteration overhead is measurable.
- The UB budget, not parallelism or vector width, is the binding limit on BLOCK_N.
- Multiple kernels in the same operator hold different numbers of simultaneously live tensors, making a single conservative BLOCK_N for all kernels suboptimal.
- Increasing BLOCK_N measurably reduces loop iterations for the dominant input shapes.

## Signals

### Code

- BLOCK_N is set to a conservative constant that leaves substantial UB headroom.
- Forward and backward (or other paired) kernels use the same BLOCK_N even though they hold different numbers of live tensors.
- The inner column loop includes pointer math, `tl.arange`, and mask construction per iteration.

### Profile

- Loop iteration count correlates with kernel latency for shapes with large column extents.
- Increasing BLOCK_N on large-column shapes improves throughput without regressing small shapes (whose BLOCK_N is already alignment-capped).

## Strategy

1. Count simultaneously live tensors per kernel in one inner-loop iteration — input tiles, intermediate upcasts, output tiles.

2. Compute the safe max BLOCK_N from the UB budget:

   ```
   max_N = UB_BYTES // (element_size * BLOCK_M * num_live_tensors)
   ```

   Round down to the nearest alignment boundary.

3. Apply per-kernel caps. Two kernels with different live-tensor counts get different BLOCK_N bounds because each has its own UB budget.

4. Step up BLOCK_N in powers of two (512, 1024, 2048, ...) and benchmark after each step. Stop when latency plateaus or correctness fails. The sweet spot is typically one step below the UB overflow point.

5. Gate by dtype: smaller element sizes allow larger BLOCK_N at the same byte budget.

## Avoid When

- The kernel is compute-bound on a hardware intrinsic with fixed throughput — larger tiles may not help if the compute unit is saturated.
- BLOCK_N is already at the UB limit for the most constrained kernel.
- Alignment requirements on small column extents already cap BLOCK_N below the UB ceiling.
- The inner-loop overhead is negligible relative to per-element compute time.

## What To Verify After Applying

- UB budget formula confirms the new BLOCK_N fits within capacity for each kernel and dtype.
- Correctness tests pass — larger BLOCK_N does not change semantics for elementwise ops.
- Benchmark geomean improves. Large-column shapes gain most; small-column shapes stay neutral.
- Per-kernel BLOCK_N differentiation compiles and runs correctly.

## Related Patterns

- **program-multiple-rows**: Optimizes the row dimension (BLOCK_M) of the same 2D tile. Apply first, then tune BLOCK_N.
- **tiling (hierarchical)**: Reduces tile sizes to prevent UB overflow. Apply only when the tile already overflows.
- **shape-gated-block-size-selection**: Tiered BLOCK_SIZE dispatch for reduction kernels with variable column extents.
