# Exact-Tile No-Boundary Fast Path

## Summary

Split exact-tile hot paths from generic masked kernels when dispatch-time shape guards can prove there are no tail tiles, so Ascend lowering can avoid boundary-only masks, padding values, block-pointer `boundary_check`, and related control branches. Also applies to 1D elementwise kernels (`n_elements % BLOCK_SIZE == 0`).

## Use When

- A dominant benchmark shape is exactly tile-divisible, such as `M % BLOCK_M == 0` and `N % BLOCK_N == 0`, or for 1D elementwise kernels `n_elements % BLOCK_SIZE == 0`.
- Python dispatch can guard the aligned branch before launch and keep the original masked kernel as fallback.
- MLIR, LLVM, or profiler traces still show boundary checks, masks, padding, or branch/control overhead on the exact-tile hot path.
- The kernel is already structurally reasonable, so a bounded control-overhead cleanup can matter.
- The kernel uses `tl.load(..., mask=offsets < n_elements, other=0.0)` or similar boundary-only masks that are redundant when the shape is exactly tile-divisible.

## Avoid When

- The mask is algorithm semantics, not a boundary/tail guard.
- Exact-divisibility cannot be proven at dispatch.
- Tail-heavy or irregular shapes dominate the workload.
- The main bottleneck is clearly random global memory, atomics, or compute throughput and boundary control is negligible.
- The fast path would duplicate too much complex logic and drift from the fallback.

## Signals

### Code

- `tl.load(..., mask=tail_mask, other=...)` where the mask only protects block edges.
- **1D elementwise signal**: `mask = offsets < n_elements` or `mask = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < n_elements` where `n_elements` is a total element count for a flattened tensor.
- `tl.store(..., mask=tail_mask)` on shapes known to be full-tile.
- `tl.make_block_ptr` loads or stores keep `boundary_check` for exact shapes.
- Removing the mask does not change address math for the guarded shape.
- The kernel uses `@triton.autotune` with configs that include a BLOCK_SIZE where at least one config makes `n_elements % BLOCK_SIZE == 0` for common shapes.

### Profile

- Parent kernel is close to the target but still shows scalar/control overhead.
- Expected gain is modest, often small single-digit percent to low double-digit percent.

## Optimization Strategy

1. Identify the hot exact-tile shape and tile divisibility guard.
2. Split a minimal aligned kernel from the generic masked kernel.
3. Remove only boundary/tail masks, padding, and `boundary_check` in the aligned kernel.
4. Keep the generic masked kernel for all non-exact cases.
5. Compare parent-vs-child performance on the exact case and verify fallback coverage.

### Variant: 1D elementwise unmasked fast path

For 1D elementwise kernels that process a flat tensor over `n_elements`, many benchmark shapes (especially powers of 2) are exactly divisible by the BLOCK_SIZE. Split into two kernels:

- **Unmasked kernel**: launched when `n_elements % BLOCK_SIZE == 0`. All loads and stores use bare `tl.load(ptr + offsets)` and `tl.store(ptr + offsets, value)` without `mask=`, `other=`, or boundary predicates. On Ascend NPU, masked loads with `other=0.0` materialize the fill value in UB even for masked-out lanes; removing the mask eliminates both SCALAR predicate overhead and UB fill pressure.
- **Masked kernel**: original kernel with `mask = offsets < n_elements` for all non-exact cases.

### Variant: combining with autotune

When the kernel uses `@triton.autotune`, the exact-tile fast path can still be applied. Remove `@triton.autotune` from the aligned/unmasked kernel and use a fixed BLOCK_SIZE instead, because:

1. The unmasked kernel only runs when `n_elements % BLOCK_SIZE == 0`, so the tile size is determined by the divisibility condition at dispatch, not by autotune search.
2. Autotune on the unmasked kernel would compile masked fallback variants that are never used under the exact-tile guard.
3. Keep `@triton.autotune` on the masked fallback kernel for shapes that do need search.

### Variant: chunk recurrence tail peeling

For chunked recurrence kernels, the whole sequence may not be tile-divisible, but every chunk before the last one is still full. In that case, split the recurrence into a full-chunk hot loop plus one tail block:

- hot loop: `for i_t in range(NT - 1)` with no per-iteration `min`, tail mask, or boundary-only `tl.where`
- tail block: `i_t = NT - 1`, compute `last_idx = min(NT * BT, T) - 1`, and keep the masks needed for the partial final chunk

This is a "mostly exact tile" fast path. It avoids paying tail-control cost in every recurrence iteration while preserving the generic final chunk behavior.

```python
# Full chunks: no scalar min/mask/where in the hot loop.
for i_t in range(NT - 1):
    last_idx = (i_t + 1) * BT - 1
    b_g_last = tl.load(g_base + last_idx)
    b_g = tl.load(p_g_full_chunk, boundary_check=(0,))
    b_v = b_v * exp(b_g_last - b_g)[:, None]

# Tail chunk: may be partial.
i_t = NT - 1
last_idx = min(NT * BT, T) - 1
m_t = (i_t * BT + tl.arange(0, BT)) < T
b_g_last = tl.load(g_base + last_idx)
b_g = tl.load(p_g_tail, boundary_check=(0,))
b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
```

Use this when only the final chunk can be partial. Avoid it if many chunks are irregular, or if the mask is algorithm semantics rather than boundary protection.

## Example

```python
if M % BLOCK_M == 0 and N % BLOCK_N == 0:
    _kernel_aligned_no_boundary[grid](...)
else:
    _kernel_masked_fallback[grid](...)
```

Inside the aligned kernel:

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
value = tl.load(src + offs_m[:, None] * stride_m + offs_n[None, :])
tl.store(dst + offs_m[:, None] * out_m + offs_n[None, :], value)
```

### 1D elementwise kernel

Dispatch logic:

```python
if n_elements % BLOCK_SIZE == 0:
    _kernel_unmasked[grid](..., BLOCK_SIZE=BLOCK_SIZE)
else:
    _kernel_masked[grid](..., BLOCK_SIZE=BLOCK_SIZE)
```

Inside the unmasked kernel, all loads and stores omit `mask=` and `other=`:

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(ptr + offs)          # no mask, no other
tl.store(out_ptr + offs, result)  # no mask
```

## Evidence

NPUKernelBench `20_Gather` rank-2 `dim=0` used this on `bf16 x=(5120,27648), dim=0, index=(2560,27648)`. Splitting an aligned/no-boundary kernel reduced about `4239us -> 3850us` (**~1.10x**). The remaining bottleneck was still random global-memory gather, so treat this as control-overhead cleanup rather than an access-pattern fix.

## What To Verify After Applying

- Fast path and fallback produce identical values on representative exact-tile shapes.
- Fallback still handles non-divisible shapes.
- The aligned kernel IR no longer contains the targeted boundary checks or masks.
- Parent-vs-child benchmark improves on the targeted case without broad regressions.
- For 1D elementwise: verify unmasked path produces bit-identical results on exact-tile shapes, and unmasked kernel uses fixed BLOCK_SIZE (not autotuned) while masked kernel retains autotune.
- For chunk recurrence tail peeling, test `T < BT`, `T == BT`, `T % BT == 0`, `T % BT != 0`, and varlen branches if present.

## Related Patterns

- `compile_hint`
- `padded_row_col_copy`
- `block-pointer-dimensionality`
- `discrete_memory_access`
- `scalar-latency-traps`
- `autotune`
