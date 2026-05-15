# Remove Implicit Transpose Pattern

## Summary

Eliminate **implicit transpose-style access** on Ascend NPU. This pattern targets two common forms:

1. **Host-side physical layout mismatch**: The operand is stored in one layout (e.g. `[N, K]`) but the kernel accesses it with transpose-like strides as if it were `[K, N]`. Fix by materializing the transposed operand on the host.

2. **Dot-operand expression ordering**: A `tl.dot` operand goes through `tl.trans(x).to(dtype)` before entering Cube work, causing the compiler to insert extra layout transforms. Fix by reordering: do the dtype conversion first, then pass the transpose to `tl.dot` directly so the Cube unit handles it natively.

In both cases the compiler inserts extra **layout transforms** (often visible as `nd2nz` + additional transpose/reorder control), and IR shows marks like **`MayImplicitTransposeWithLastAxis`**.

## Use When

- You implement GEMM / Linear-like kernels where one operand is stored as `[N, K]` but the math needs `[K, N]` (e.g. `y = x @ w.T`).
- Kernel code accesses the operand with **transpose-like strides** (treats `[N, K]` as `[K, N]`).
- A `tl.dot` operand uses `tl.trans(x).to(dtype)` where the transpose is applied before the dtype conversion, and the result feeds directly into `tl.dot`.
- Profiling shows high **scalar/control** and/or large **WAIT_FLAG** time around the matmul path.

## Signals

### Code

- Weight is stored as `weight: [N, K]` (PyTorch `nn.Linear` default).
- Kernel computes `b_ptrs` like `b_ptr + k * stride_bk + n * stride_bn` and relies on strides to emulate `[K, N]`.
- `tl.dot` operands use the pattern `tl.trans(b).to(tl.float16)` where the transpose is applied before the type cast.

### IR

Look for patterns like:

- `annotation.mark {MayImplicitTransposeWithLastAxis}`
- `memref.reinterpret_cast ... sizes: [*, *], strides: [1, ?]` on the B tile (common transpose-style view)

These marks strongly correlate with extra transform work in the backend lowering.

### Profile

- `WAIT_FLAG_DEVI` dominates the CUBE timeline around matmul.
- `MOV_OUT_TO_L1_MULTI_ND2NZ` / `nd2nz` and related fixpipe steps appear frequently.
- AIV shows large scalar `LD_XD_XN_IMM` / `ST_XD_XN_IMM` overhead tied to staging/reorder.

## Optimization Strategy

### Strategy A: Host-side materialization (physical layout mismatch)

Materialize the operand in the exact physical layout the kernel needs:

- Host pre-transform: `b_kn = weight.t().contiguous()` to get `[K, N]` contiguous.
- Pass `b_kn` into the kernel and index it as `[k, n]` directly.

This removes the need for the compiler to infer transpose semantics from strides and typically avoids the `MayImplicitTransposeWithLastAxis` path.

### Strategy B: Dot-operand expression reorder (dtype before transpose)

When the transposed tensor is consumed directly by `tl.dot`, let the Cube unit handle the transpose:

```python
# Before
b = tl.trans(b).to(tl.float16)
acc = tl.dot(a, b)

# After
b = b.to(tl.float16)
acc = tl.dot(a, tl.trans(b))
```

Only apply this when the transposed tensor is directly consumed by `tl.dot`. Pure Vector code or non-dot uses do not benefit from the Cube load path.

## Implementation Sketch (Strategy A)

### Before (implicit transpose-style access)

```python
# weight is [N, K]
b = weight.contiguous()
stride_bn, stride_bk = b.stride()

# kernel treats b as [K, N] via strides
b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
```

### After (host materialized transpose)

```python
# b_kn is [K, N] contiguous
b_kn = weight.t().contiguous()
stride_bk, stride_bn = b_kn.stride()

# kernel uses true [K, N] indexing
b_ptrs = b_kn_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
```

## Performance Impact Expectations

- Often reduces IR-level implicit transpose marks (e.g. `MayImplicitTransposeWithLastAxis` disappears).
- Can reduce CUBE-side waiting (`WAIT_FLAG_DEVI`) by simplifying the transform path.
- Can reduce AIV scalar load/store pressure when the backend previously staged/reordered the tiles.

## Pitfalls / Risks

- **Extra host-side work**: `weight.t().contiguous()` is a real transpose + copy.
  - Good when weights are **reused** (inference/training loops), less good if weights change every call.
- **Memory overhead**: storing both `[N, K]` and `[K, N]` can double weight storage if not managed carefully.
- **Layout mismatch across kernels**: ensure downstream kernels expect the same layout; keep the original weight too if needed.
- **Dot-operand reorder**: only valid when the final consumer is `tl.dot`. Reordering for a Vector-only consumer may not help.

## What To Verify After Applying

1. **Correctness**: compare output against reference for multiple shapes.
2. **IR**: confirm `MayImplicitTransposeWithLastAxis` no longer appears for the matmul operand.
3. **Profiler**: check `WAIT_FLAG_DEVI` and transform ops (`nd2nz`, `MOV_*_ND2NZ`) reduce or become cheaper.
4. **Benchmark discipline**: include warmup because first-run includes compilation/tuning overhead.

## Related Patterns

- Complements **`software-pipeline`**: this pattern fixes operand layout; pipeline fixes overlap.
- Complements **`tiling`**: layout fix can enable better tiling outcomes.
- Often a prerequisite before **`autotune`**: tuning on a structurally suboptimal implicit-transpose layout may mislead.
