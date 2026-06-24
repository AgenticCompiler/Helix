# Classic Tiled Matmul Rewrite Pattern

Use this reference when a Triton Ascend NPU kernel is logically matmul-like but the current hot loop is manual reduction code, row-wise multiply-sum code, or scalar-heavy pointer math around `K`.

## Summary

Rewrite a manual matmul or K-reduction hot loop into a regular tiled `tl.dot` matmul so the kernel shape matches what Ascend Triton lowers well.

This is a structure-first pattern. It is often the right move before launch/pipeline micro-tuning, but it is not universally beneficial for every regime.

## Use When

- The kernel computes an `M x N` output tile with regular reduction over `K`.
- Current code is effectively `sum_k A[..., k] * B[..., k]`.
- Profile/IR shows heavy scalar address/control overhead in the hot loop.
- Partial pointer/layout fixes helped but the loop is still not a regular matmul skeleton.
- Dtype- or shape-specialized dispatch is acceptable if one regime clearly benefits.

## Avoid When

- Purely elementwise kernels.
- Gather/scatter-dominated kernels.
- Very small shapes where tile setup does not amortize.
- Workloads where the kernel is already a solid tiled matmul and remaining issues are tile/pipeline/hint details.

If the main issue is overlap and load/compute scheduling, use `software-pipeline`.
If the main issue is tile footprint/UB pressure, use `tiling`.

## Rewrite Goal

Turn the hot loop into standard tiled form:

- `pid_m`, `pid_n`
- `offs_m`, `offs_n`, `offs_k`
- masked `tl.load` for `A[BLOCK_M, BLOCK_K]`
- masked `tl.load` for `B[BLOCK_K, BLOCK_N]`
- `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`
- `acc += tl.dot(a, b)`
- fused epilogue after the dot loop

## Skeleton

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.static_range(0, K, BLOCK_K):
        k_idx = offs_k + k
        a_ptrs = A + offs_m[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = B + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_idx[None, :] < K)
        b_mask = (k_idx[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b)
```

## Layout Rule

If a weight tensor is stored as `[N, K]` but the dot expects `[BK, BN]`, keep source storage and materialize the logical orientation via pointer math:

```python
w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_idx[:, None] * stride_wk
```

This is usually cleaner than forcing transpose-like post-load transforms.

## Precision Rule

Separate:

1. input tile dtype
2. accumulator dtype
3. store dtype

On Ascend Triton, keep `tl.dot` operand dtypes matched. Do not blindly force fp16 when operator semantics or accuracy expectations require fp32 behavior.

Typical default:

- load `a`/`b` in their real input dtype when already matched,
- keep `acc` in fp32,
- keep fused epilogue in fp32,
- store in output dtype.

Use explicit casts only when dtype mismatch exists and the cast is both numerically acceptable and performance-positive.

When one unified rewrite fails correctness/perf across regimes, prefer dispatch instead of discarding the approach:

- fp16 + larger `M`: tiled matmul path
- fp32 or small shapes: baseline-style reduction path

## Host Launch Template

```python
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

matmul_kernel[grid](
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M=BLOCK_M,
    BLOCK_N=BLOCK_N,
    BLOCK_K=BLOCK_K,
    num_warps=4,
    num_stages=1,
)
```

Treat these values as starting points, not universal settings.

## Dispatch Rule

Do not force one path for every regime. Use measured thresholds:

```python
if x.dtype == torch.float16 and M >= LARGE_M_THRESHOLD:
    launch_tiled_matmul(...)
else:
    launch_baseline_reduction(...)
```

Thresholds must come from benchmark evidence.

## Expected Benefit

- More regular lowering of the `K` loop.
- Better fit for `tl.dot`/matmul codegen.
- Less scalar-heavy reduction structure.
- Better epilogue amortization on larger shapes.

## Main Risks

- Tile setup overhead hurts small shapes.
- Larger tiles increase register/UB pressure.
- Forced operand downcast changes semantics.
- Replacing an already-optimized vendor GEMM path can catastrophically regress.
- Aggressive grouped/aspect sweeps are non-monotone; nearby tuples can collapse performance.

## Related Patterns

- `software-pipeline`
- `tiling`
- `remove-implicit-transpose`

## What To Verify After Applying

- Logical `A`/`B`/`C` shape contracts.
- Second operand arrives in expected `[BK, BN]` orientation.
- Correct masks on `M`, `N`, `K` boundaries.
- Fused epilogue broadcast correctness.
- Each dispatched branch is validated independently.
- Performance is compared against immediate parent.
- Document whether gains came from reduced scalar pressure, better lowering, or epilogue amortization.
