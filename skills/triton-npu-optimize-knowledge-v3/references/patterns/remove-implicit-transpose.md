# Remove Implicit Transpose Pattern

## Summary

Eliminate implicit transpose-style operand access by materializing the required physical layout explicitly (often on host) instead of relying on stride tricks in-kernel.

This pattern targets GEMM/linear paths where backend lowering inserts extra transform/reorder work (for example `nd2nz`-class overhead) because an operand is logically transposed but physically stored in the opposite orientation.

## Use When

- Math needs `[K, N]` but operand is stored as `[N, K]`.
- Kernel indexes operand with transpose-emulating strides.
- Profile/IR suggests transform-heavy lowering and wait-heavy matmul execution.

## Signals

### Code

- Operand storage follows framework default (for example `weight: [N, K]`) while kernel consumes `[K, N]`.
- Addressing uses stride tricks to reinterpret layout rather than explicit transformed storage.

### IR / Profile

- IR markers like `MayImplicitTransposeWithLastAxis` and reinterpret casts on dot operand tiles.
- Excessive wait/transform pipeline cost around matmul.

## Optimization Strategy

1. Materialize operand in true consumer layout (for example `b_kn = weight.t().contiguous()`).
2. Pass/layout-bind this tensor directly to the kernel.
3. Index it in natural `[k, n]` order.
4. Benchmark against immediate parent; retain only if host-transform overhead is amortized.

## Implementation sketch (Triton)

### Before (implicit transpose-style)

```python
# weight is [N, K]
b = weight.contiguous()
stride_bn, stride_bk = b.stride()
b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
```

### After (materialized layout)

```python
# b_kn is [K, N] contiguous
b_kn = weight.t().contiguous()
stride_bk, stride_bn = b_kn.stride()
b_ptrs = b_kn_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
```

## Pitfalls / risks

- Host-side transpose+copy cost can dominate if weights are not reused.
- Memory overhead increases if both layouts are retained.
- Downstream kernels must agree on chosen layout contract.

## What To Verify After Applying

1. Correctness against reference on representative shapes.
2. IR no longer follows implicit-transpose lowering path.
3. Wait/transform profile signals improve on target cases.
4. Benchmarks include warmup and compare against immediate parent.

## Related Patterns

- `software-pipeline`
- `tiling`
- `autotune`
