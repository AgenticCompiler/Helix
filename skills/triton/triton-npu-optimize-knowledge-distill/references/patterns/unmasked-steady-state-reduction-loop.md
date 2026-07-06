# Unmasked Steady-State Reduction Loop

## Summary

Split a column-wise reduction loop into two sections: unmasked steady-state iterations that process full blocks without masks or `other=` values, and a masked remainder that handles only the final partial block. Combine with doubled per-iteration width (two BLOCK_SIZE blocks per iteration) to halve the steady-state trip count. Eliminates mask evaluation, boundary checking, and padding-value overhead from the hot path.

## Use When

- A row-wise reduction kernel iterates over a column dimension in fixed BLOCK_SIZE steps.
- Most iterations process full blocks where the next block also fits within the column extent.
- Profiling shows scalar or mask evaluation overhead in the inner loop.
- The column extent is large enough that the code-size increase from splitting the loop is justified by the per-iteration savings.

## Signals

### Code

- Every inner-loop iteration includes a mask expression, `other=` padding value, or bounds check.
- The mask is always `offsets < n_cols` — purely a boundary guard, not algorithmic semantics.
- All full-block iterations compute an all-True mask that the hardware still evaluates.

### Profile

- Scalar time is elevated relative to vector time in the reduction loop.
- Medium-to-large column extents show the most benefit since they produce the most steady-state iterations.

## Strategy

1. Compute the extent of elements that fit in complete double-blocks: `full_iters = (n_cols // (BLOCK_SIZE * 2)) * (BLOCK_SIZE * 2)`.
2. Process the steady-state range `[0, full_iters)` with unmasked loads — drop `mask=` and `other=` arguments.
3. Load and process two BLOCK_SIZE blocks per iteration to halve the trip count.
4. Process the remainder `[full_iters, n_cols)` with masked loads on both block A and block B; use `other=float("-inf")` so a fully OOB block B contributes zero to the running sum.
5. Keep the remainder path unconditional (no `if` branch inside it) so the compiler does not fork the code path at runtime.

Steady-state (unmasked, 2 blocks per iteration):

```python
full_iters = (n_cols // (BLOCK_SIZE * 2)) * (BLOCK_SIZE * 2)
for i in range(0, full_iters, BLOCK_SIZE * 2):
    X_a = tl.load(ptr + i + tl.arange(0, BLOCK_SIZE)).cast(tl.float32)
    max_a = tl.max(X_a, axis=0)
    X_b = tl.load(ptr + i + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).cast(tl.float32)
    max_b = tl.max(X_b, axis=0)
    combined_max = tl.maximum(max_a, max_b)
    m_new = tl.maximum(m, combined_max)
    d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_a - m_new), axis=0) + tl.sum(tl.exp(X_b - m_new), axis=0)
    m = m_new
```

Remainder (masked, unconditional double-block):

```python
if full_iters < n_cols:
    i = full_iters
    mask_a = (i + tl.arange(0, BLOCK_SIZE)) < n_cols
    X_a = tl.load(ptr + i + tl.arange(0, BLOCK_SIZE), mask=mask_a, other=float("-inf")).cast(tl.float32)
    max_a = tl.max(X_a, axis=0)

    mask_b = (i + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) < n_cols
    X_b = tl.load(ptr + i + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)), mask=mask_b, other=float("-inf")).cast(tl.float32)
    max_b = tl.max(X_b, axis=0)

    combined_max = tl.maximum(max_a, max_b)
    m_new = tl.maximum(m, combined_max)
    d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_a - m_new), axis=0) + tl.sum(tl.exp(X_b - m_new), axis=0)
    m = m_new
```

When block B is fully OOB the mask is all-False, `other=float("-inf")` produces a -inf block, and `tl.max` on a -inf vector returns -inf. Since `exp(-inf - m_new) = 0`, block B contributes zero — matching the single-block case without a runtime branch.

## Avoid When

- The column extent is small and every iteration needs a mask (zero steady-state iterations).
- The reduction is already compute-bound and mask overhead is negligible.
- Code-size constraints prevent duplicating the reduction logic for the remainder section.
- An `if/else` branch in the remainder (instead of unconditional double-block) is measurably cheaper — test on the target platform before assuming unconditional is better.

## What To Verify After Applying

- `full_iters = (n_cols // BLOCK2) * BLOCK2` is correct; steady-state never accesses OOB.
- Remainder path handles all three cases: partial block A only, full A with partial B, full A with no B.
- Numeric results match the all-masked baseline for boundary extent values.
- Geomean speedup improves; no case regresses from the loop duplication.

## Related Patterns

- `chunked-loop-unrolling` — also uses doubled per-iteration width, but for DMA scheduling in element-wise kernels rather than mask elimination in reductions.
- `shape-gated-block-size-selection` — chooses BLOCK_SIZE from tiered problem-size thresholds to ensure enough steady-state iterations.
- `scalar-latency-traps` — remove redundant boundary masks that this pattern's split-loop structure makes unnecessary.
