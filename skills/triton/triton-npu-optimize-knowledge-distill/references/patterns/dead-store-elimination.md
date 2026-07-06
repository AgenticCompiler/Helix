# Dead Store Elimination

## Summary

Remove `tl.store` writes to global memory when the written value is only produced for local register use within the same kernel and the storage tensor is never returned by the wrapper function. Replace the dead output tensor allocation with a zero-sized dummy allocation to avoid the allocation cost.

Eliminating a dead store saves one full global write per element, directly reducing MTE (memory transfer engine) bandwidth consumption without changing any computation.

## Use When

- A kernel computes an intermediate value in registers that is written to global memory via `tl.store`.
- The written-to tensor is never returned by the wrapper or loaded from in any subsequent kernel pass.
- The stored value is used only within the same kernel's register scope for downstream computation (e.g., a reciprocal-standard-deviation vector that feeds into element-wise normalization).
- Profiling shows MTE write time for stores whose destinations have no readers.

## Signals

### Code

- A `tl.store` writes to a tensor pointer (`RSTD_ptr`, `aux_ptr`, `tmp_ptr`, etc.) that is allocated in the wrapper but never appears in the wrapper's return tuple.
- The wrapper allocates a tensor with `torch.empty(...)` whose only use is as the destination of an internal store.
- The stored value is computed from data that is also loaded and reduced within the same kernel — the store is a write-only side effect.

### Profile

- MTE write bandwidth is consumed for a buffer that has no corresponding read in any kernel or host operation.
- Eliminating the store reduces MTE time proportionally without affecting compute or read bandwidth.

## Rewrite

### Before

```python
@triton.jit
def kernel(..., RSTD_ptr, RSTD_row_stride, ...):
    # ... compute rstd_vec in registers ...
    rstd_vec = tl.rsqrt(var + eps)
    tl.store(RSTD_ptr + rows_off * RSTD_row_stride, rstd_vec, mask=rows_mask)

    for col_offset in range(0, n_cols, BLOCK_SIZE_N):
        # ... use rstd_vec for normalization ...
        Y_chunk = (S_chunk * rstd_vec[:, None]) * W_chunk
        tl.store(Y_ptr + ..., Y_chunk, mask=...)

# Wrapper:
RSTD = torch.empty(n_rows, dtype=torch.float32, device=device)
kernel[grid](..., RSTD, RSTD.stride(0), ...)
# RSTD is never returned or read — dead store
```

### After

```python
@triton.jit
def kernel(..., RSTD_ptr, RSTD_row_stride, ...):
    # ... compute rstd_vec in registers — no store needed ...
    rstd_vec = tl.rsqrt(var + eps)

    for col_offset in range(0, n_cols, BLOCK_SIZE_N):
        # ... use rstd_vec for normalization directly ...
        Y_chunk = (S_chunk * rstd_vec[:, None]) * W_chunk
        tl.store(Y_ptr + ..., Y_chunk, mask=...)

# Wrapper:
RSTD = torch.empty((0,), dtype=torch.float32, device=device)
kernel[grid](..., RSTD, RSTD.stride(0), ...)
```

Key rules:

1. **Use a zero-sized tensor as replacement**, not a `None` or omitted argument. The kernel signature still expects the pointer parameter even if no store uses it. A `torch.empty((0,))` satisfies the pointer requirement with zero allocation cost.
2. **Verify no downstream consumer reads the stored value.** Check that the tensor is not in the wrapper's return tuple, not passed to another kernel, and not loaded from within the same kernel after the store.
3. **Keep the pointer in the kernel signature** unless you also restructure the caller. Removing the parameter requires changing the wrapper and all call sites — the allocation change alone is simpler and equivalently effective.

## Avoid When

- The stored tensor is returned by the wrapper or passed to a downstream kernel — the store has a real consumer.
- The stored value is reloaded within the same kernel in a subsequent loop pass — the store serves a purpose (e.g., reducing register pressure by spilling).
- The stored tensor participates in correctness validation or debugging assertions — removing it may break tooling.
- The value is intentionally truncated to a narrower dtype by the store (e.g., fp32 register → bf16 memory), and downstream consumers depend on that truncation.

## What To Verify After Applying

- Correctness: outputs match the reference exactly for all dtypes and shapes.
- The dead tensor allocation is replaced with `torch.empty((0,))`, not removed entirely, to avoid kernel signature mismatches.
- MTE write bandwidth is reduced: profiling shows fewer or shorter DMA write phases.
- No new regressions: the elimination should be a pure removal of dead work — if any case regresses, investigate whether the store was masking a register-pressure issue that now manifests differently.

## Related Patterns

- Complements `merge-adjacent-stores`: after eliminating dead stores, merge remaining adjacent stores into wider writes.
- Complements `intra-kernel-pass-fusion`: that pattern eliminates a store-reload pair by fusing passes; this pattern eliminates a dead store that has no reload.
- Differs from `algebraic-optimization` Case 1: that pattern reformulates math to eliminate entire passes; this pattern removes a single dead write within an otherwise unchanged kernel.
- `multi-output-kernel-writes` — adds a second store to avoid a post-kernel consumer; this pattern removes a store with no consumer.
- `in-kernel-reduction-to-cut-memory-traffic` — similar goal (eliminating writes) but for stores whose result is immediately reduced on the host.
