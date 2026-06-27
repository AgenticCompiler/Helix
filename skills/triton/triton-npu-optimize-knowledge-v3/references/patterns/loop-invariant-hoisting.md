# Loop-Invariant Hoisting Pattern

## Summary

Apply loop-invariant code motion (LICM) in Triton kernels: move work that does not depend on the loop variable out of the hot loop so each iteration only executes truly varying computation.

On Ascend NPU this often reduces scalar/control pressure (address generation, mask bookkeeping, repeated setup) and can improve throughput when loop bodies are control-heavy.

## Use When

- Hot inner loops repeatedly rebuild pointer bases, masks, or scalar setup.
- Profiling shows scalar/control overhead disproportionately high vs useful math.
- Kernel structure is mostly correct, but loop bookkeeping remains heavy.

## Avoid When

- Main bottleneck is still layout/store shape, launch geometry, or algorithm selection.
- Candidate expressions actually vary with loop index.
- Refactor risks numerically sensitive semantics without clear validation.

## Signals

### Code

- Repeated `base(pid, offs) + delta(k)` expressions in loop body.
- Invariant mask fragments rebuilt each iteration.
- Per-iteration scalar setup for launch-invariant terms.

### Profile / IR

- Scalar instruction mix dominates loop runtime.
- Repeated arithmetic/index-cast chains inside loop bodies (for example `muli`/`addi`/`index_cast` motifs under `scf.for` / `scf.while`).
- `WAIT_FLAG_DEVI`-class waits around dot/reduction bodies can indicate loop-control pressure starving downstream compute when scalar setup is heavy.

## Optimization strategy

For expression `E(loop_var)`:

1. Split into loop-invariant base + loop-varying delta.
2. Compute base once outside loop.
3. Keep only delta and minimal combines inside loop.

## Specialization A: Pointer address-generation hoisting

### Before

```python
k = 0
while k < K:
    k_offs = k + offs_k
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
    b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    a = tl.load(a_ptrs, ...)
    b = tl.load(b_ptrs, ...)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

### After

```python
a_base = a_ptr + (offs_m[:, None] * stride_am)
b_base = b_ptr + (offs_n[None, :] * stride_bn)

k = 0
while k < K:
    k_offs = k + offs_k
    a_ptrs = a_base + (k_offs[None, :] * stride_ak)
    b_ptrs = b_base + (k_offs[:, None] * stride_bk)
    a = tl.load(a_ptrs, ...)
    b = tl.load(b_ptrs, ...)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

## Specialization B: Mask / bounds hoisting

Precompute invariant mask terms once and combine with loop-varying mask pieces in-loop.

```python
a_mask_m = offs_m < M
b_mask_n = offs_n < N

k = 0
while k < K:
    k_offs = k + offs_k
    k_mask_row = k_offs[None, :] < K
    k_mask_col = k_offs[:, None] < K
    a_mask = a_mask_m[:, None] & k_mask_row
    b_mask = k_mask_col & b_mask_n[None, :]
    ...
```

## Performance expectations

- Lower scalar/control overhead in large-loop kernels.
- Cleaner loop body for backend scheduling.
- Usually low-risk and incremental when semantics are preserved.

## Pitfalls / risks

- Broadcast orientation mistakes (`[:, None]` vs `[None, :]`).
- Over-hoisting expressions that actually vary with loop index.
- Assuming LICM solves layout/transform costs by itself.

## What To Verify After Applying

1. Correctness on boundary/tail shapes.
2. Parent-vs-child benchmark comparison.
3. Reduced scalar/control pressure in profiler or cleaner loop-body IR.

## Related Patterns

- `compile_hint`
- `software-pipeline`
- `layout-store-and-block-pointers`
- `program-multiple-rows`
- `remove-implicit-transpose`
