# Classic Tiled Matmul Rewrite Pattern

Use this reference when a Triton Ascend NPU kernel is logically matmul-like but the current hot loop is written as manual reduction code, row-wise multiply-plus-sum code, or scalar-heavy pointer math around the `K` axis.

## When To Use

- the kernel computes an `M x N` output tile with a regular reduction over `K`
- the current implementation is effectively `sum_k A[..., k] * B[..., k]`
- profiling or IR suggests the hot loop is spending too much effort on scalar address generation or repeated reduction structure
- a block-pointer rewrite reduced one scalar chain but the full loop is still not a regular matmul

Do not use this pattern for:

- purely elementwise kernels
- gather/scatter dominated kernels
- tiny shapes where tile setup cost is unlikely to amortize

Choose this pattern when the main problem is **kernel structure**. The question it answers is:

- should this manual reduction loop become a regular tiled matmul at all

If the kernel is already a reasonable tiled matmul and the remaining problem is memory/compute overlap, use `software-pipeline` instead.
If the kernel is failing because tiles or intermediates are too large for UB, use `tiling` instead.

## Rewrite Goal

Turn the hot loop into the standard tiled form:

- `pid_m`, `pid_n`
- `offs_m`, `offs_n`, `offs_k`
- explicit masked `tl.load` for `A[BLOCK_M, BLOCK_K]`
- explicit masked `tl.load` for `B[BLOCK_K, BLOCK_N]`
- `acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)`
- `acc += tl.dot(a, b)`
- fused bias/activation epilogue after the dot loop

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
        a = a.to(tl.float16)
        b = b.to(tl.float16)
        acc += tl.dot(a, b)
```

## Layout Rule

If a weight tensor is stored as `[N, K]` but the dot wants `[BK, BN]`, keep the source tensor in place and materialize the tile in the needed logical orientation through pointer math:

```python
w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_idx[:, None] * stride_wk
```

This is usually cleaner than forcing a transpose-like transform after load.

## Precision Rule

Separate these choices:

1. input tile dtype
2. accumulator dtype
3. final store dtype

On Ascend Triton, make the two `tl.dot` operand dtypes explicit when there is any chance they may differ. A common pattern is:

- `a`, `b`: `fp16`
- `acc`: `fp32`
- fused epilogue: `fp32`
- final store: output tensor dtype

If the operator must preserve a true `fp32` matmul path, treat that as a separate validated path rather than assuming the backend handles it the same as mixed precision.

When a single tiled rewrite is fast but fails the existing correctness contract for some dtypes or shape regimes, prefer **dtype-specialized or shape-specialized dispatch** over discarding the idea entirely. A common fallback is:

- `fp16` and sufficiently large `M`: tiled matmul path
- `fp32` or small shapes: baseline-style reduction path

Use this when the performance win is real for one operating regime, but a unified replacement changes accumulation order or precision behavior too much for another regime.

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

Treat those tile values as a starting point, not as a universal rule.

## Dispatch Rule

Do not assume the classic tiled path must replace every input regime.

Prefer a dispatched design when evidence shows:

- tiled matmul is clearly faster for `fp16` or larger shapes
- baseline-style reduction is more accurate or cheaper for `fp32` or smaller shapes
- the correctness gate rejects one unified implementation

Typical host-side decision pattern:

```python
if x.dtype == torch.float16 and M >= LARGE_M_THRESHOLD:
    launch_tiled_matmul(...)
else:
    launch_baseline_reduction(...)
```

The exact threshold should come from benchmark evidence, not from a fixed guess.

## Expected Benefit

- more regular lowering of the main `K` loop
- better fit for `tl.dot`/matmul code generation
- less scalar-heavy reduction structure
- better amortization of fused epilogues on larger `M` and `N`

## Main Risks

- tile setup overhead can hurt small shapes
- larger tiles can increase register or UB pressure
- forcing `fp16` dot inputs can change `fp32` input semantics

## Boundary With Nearby Patterns

### vs `software-pipeline`

- `classic-matmul` changes the loop into a standard tiled load-plus-dot structure
- `software-pipeline` assumes that tiled structure already exists and then overlaps current-tile compute with next-tile load

Use `classic-matmul` first when the current hot loop is still written as manual reduction code.
Use `software-pipeline` after that only if profiling still shows load-then-compute stalls.

### vs `tiling`

- `classic-matmul` is about choosing a matmul-shaped kernel structure
- `tiling` is about reducing UB pressure or overlarge block working sets

If your evidence says the kernel is structurally matmul-like but scalar-heavy, prefer `classic-matmul`.
If your evidence says the kernel already has the right structure but block size or intermediate footprint is too large, prefer `tiling`.

## Validation Checklist

- confirm logical `A`, `B`, and `C` shapes
- confirm the second operand really arrives as `[BK, BN]`
- confirm masks on `M`, `N`, and `K`
- confirm fused bias or activation broadcasting
- if you introduce dispatch, validate each dispatched regime explicitly
- benchmark against the previous implementation
- record whether the win came from lower scalar pressure, better matmul lowering, or better epilogue amortization
