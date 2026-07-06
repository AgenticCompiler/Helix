# Intra-Kernel Pass Fusion

## Summary

Fuse two sequential passes within a single row-wise kernel into one pass when the row fits in one column tile. Eliminates intermediate store-reload cycles by keeping the bridge value in registers. Use a `tl.constexpr` boolean gate to select the fused or two-pass path at compile time.

## Use When

- A row-wise kernel computes an intermediate value in pass 1 (e.g., elementwise fusion), stores it, then reloads it in pass 2 for normalization or further computation.
- The entire row fits within one column tile (BLOCK_SIZE_N == n_cols), so the store-reload cycle is pure memory-traffic overhead with no algorithmic purpose.
- The intermediate is a single-consumer bridge: produced in pass 1, consumed only in pass 2 within the same kernel.
- Profiling shows the intermediate store and reload as significant MTE (memory transfer engine) time.

## Signals

### Code

- A kernel has two sequential column-iteration loops (`for col_offset in range(0, n_cols, BLOCK_SIZE_N)`) inside a single kernel body. The first loop computes an intermediate, stores it via `tl.store(intermediate_ptr + ..., ...)`, and accumulates row-wise statistics. The second loop reloads the intermediate via `tl.load(intermediate_ptr + ..., ...)` from the same pointer. No processing occurs between the two loops other than scalar statistics derivation.
- Both passes iterate over the same column range with the same `BLOCK_SIZE_N` and identical row-iteration pattern. The intermediate store target and reload source are the same buffer (e.g., an `S_ptr` output parameter).
- The row dimension fits within one column tile (`BLOCK_SIZE_N >= n_cols` in practice), so the store-reload is pure memory-traffic overhead with no algorithmic purpose.

### Profile

- MTE time is dominated by store-then-reload of the same logical data within one kernel invocation.
- The two-pass loop structure shows as two sequential MTE-heavy phases with no other work between them.

## Core Rewrite

When the row fits in one tile, compute the intermediate in registers, derive statistics (mean, variance, etc.) directly from it, normalize, and write the final output — all without storing the intermediate to global memory.

Before (two-pass, always):

```python
@triton.jit
def kernel(..., BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    # ... row setup ...
    # Pass 1: compute intermediate, accumulate stats
    for col_offset in range(0, n_cols, BLOCK_SIZE_N):
        X = tl.load(...)
        R = tl.load(...)
        S = X + R                          # intermediate
        tl.store(..., S, mask=...)          # store intermediate
        mean_acc += tl.sum(S_f32, axis=1)
        var_acc += tl.sum(S_f32 * S_f32, axis=1)

    mean = mean_acc / n_cols
    rstd = tl.rsqrt(var_acc / n_cols - mean * mean + eps)

    # Pass 2: reload intermediate, normalize
    for col_offset in range(0, n_cols, BLOCK_SIZE_N):
        S = tl.load(...).to(tl.float32)    # reload intermediate
        W = tl.load(...)
        Y = (S - mean) * rstd * W + bias
        tl.store(..., Y, mask=...)
```

After (fused path when BLOCK_SIZE_N == n_cols):

```python
@triton.jit
def kernel(..., BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
           USE_SINGLE_TILE_FUSED: tl.constexpr):
    # ... row setup ...
    if USE_SINGLE_TILE_FUSED:
        # Single pass: load, compute stats, normalize, write — no intermediate store
        X = tl.load(...)
        R = tl.load(...)
        S = X + R
        S_f32 = S.to(tl.float32)
        mean = tl.sum(S_f32, axis=1) / n_cols
        var = tl.sum(S_f32 * S_f32, axis=1) / n_cols - mean * mean
        rstd = tl.rsqrt(var + eps)
        W = tl.load(...)
        Y = (S_f32 - mean[:, None]) * rstd[:, None] * W + bias
        tl.store(..., Y.to(output_dtype), mask=...)
    else:
        # Two-pass fallback (same as before)
        ...

# Wrapper:
use_fused = BLOCK_SIZE_N == n_cols
```

Key implementation rules:

1. **Use a `tl.constexpr` gate**, not a runtime condition. The compiler must see a compile-time branch to specialize each path independently.
2. **The two-pass path remains as fallback** for multi-tile rows. Both paths in the same kernel source keep maintenance simple.
3. **For fp32 dtypes**, the fused path is always safe because storing to memory and reloading does not change values.
4. **For fp16/bf16 dtypes**, the fused path is safe for single-tile rows because store-reload preserves the exact fp16/bf16 bit pattern. The in-register value is identical to the stored-then-reloaded value.
5. **For bf16 with large columns**, the fused path converts the intermediate to fp32 in registers, which doubles the tile memory footprint. If this exceeds UB capacity, gate the fused path with a column threshold.

## Avoid When

- The row requires multiple column tiles. The two-pass algorithm is needed because pass 2 must see the full-row statistics computed in pass 1.
- The intermediate store is necessary for precision (e.g., intentional truncation to a narrower dtype that the algorithm depends on).
- The intermediate has multiple consumers in different phases — re-materializing it in each consumer may cost more than storing once.
- For bf16 with columns exceeding ~512, the fp32 intermediate conversion in registers may overflow UB. Verify with precise UB budget math.

## What To Verify After Applying

- Correctness matches the two-pass reference for single-tile and multi-tile shapes across all supported dtypes.
- The `USE_SINGLE_TILE_FUSED` gate is set correctly in the wrapper: True only when `BLOCK_SIZE_N == n_cols` (and any dtype-specific column thresholds are met).
- Profiling shows reduced MTE time for single-tile shapes (2 fewer GM operations per element: the intermediate store and reload are eliminated).
- The two-pass fallback path still works correctly and has not regressed.

## Related Patterns

- Complements `program-multiple-rows`: the fused path reduces per-row work, making it safer to increase BLOCK_SIZE_M.
- Differs from `sequential-kernel-fusion`: that pattern fuses separate kernel launches; this pattern fuses two passes within one kernel.
- Differs from `algebraic-optimization` Case 1: that pattern reformulates the math to use fewer statistical passes (S1+S2 formula); this pattern eliminates a memory round-trip while keeping the same math.
- Differs from `fuse-element-wise-intermediates-into-read-once-kernel`: that pattern co-locates multiple element-wise outputs within one spatial pass across PyTorch ops; this pattern fuses passes within a single kernel.
- `in-kernel-reduction-to-cut-memory-traffic` — fusing a host reduction into a kernel is a form of pass fusion at the host-kernel boundary.
