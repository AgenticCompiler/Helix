# Classic Tiled Matmul Rewrite Pattern

Use this reference when a Triton Ascend NPU kernel is logically matmul-like but the current hot loop is written as manual reduction code, row-wise multiply-plus-sum code, or scalar-heavy pointer math around the `K` axis.

## Summary

Rewrite a manual matmul or K-reduction hot loop into a regular tiled `tl.dot`-based matmul so the kernel structure matches what Ascend Triton lowers well.

## Use When

- the kernel computes an `M x N` output tile with a regular reduction over `K`
- the current implementation is effectively `sum_k A[..., k] * B[..., k]`
- profiling or IR suggests the hot loop is spending too much effort on scalar address generation or repeated reduction structure
- a block-pointer rewrite reduced one scalar chain but the full loop is still not a regular matmul
- dtype-specialized or shape-specialized paths are acceptable when one tiled regime is clearly better but a unified rewrite would change numerics too much
- simulation `report.txt` shows CUBE instr = 0 (or CUBE row absent) AND kernel source contains `tl.sum(a*b)` or equivalent K-reduction loop instead of `tl.dot`

## Avoid When

- purely elementwise kernels
- gather/scatter dominated kernels
- tiny shapes where tile setup cost is unlikely to amortize
- `tl.dot` may cause Bisheng compiler SIGSEGV depending on CANN version and kernel shape (confirmed on A5 and 910B2); verify `tl.dot` compiles on the target before relying on it, and restructure grid so N ≥ 2 when needed
- CUBE activity is low but non-zero — `tl.dot` exists but tile sizes or compiler lowering are suboptimal; route to `tiling` or `software-pipeline` instead
- inherently lightweight kernels (trivial load+store, touch kernels, standalone reduction without preceding multiply) — no matmul structure to rewrite

Choose this pattern when the main problem is **kernel structure**. The question it answers is:

- should this manual reduction loop become a regular tiled matmul at all

If the kernel is already a reasonable tiled matmul and the remaining problem is memory/compute overlap, use `software-pipeline` instead.
If the kernel is failing because tiles or intermediates are too large for UB, use `tiling` instead.

## Signals

### Code

- The hot loop uses `tl.sum(tile_a * tile_b, axis=K_dim)` or equivalent elementwise multiply + reduction instead of `tl.dot` — the CUBE engine is idle while VECTOR does matmul work.
- The K-reduction loop is a `while` loop or `range` loop instead of `tl.static_range`, preventing compiler pipelining of `tl.dot`.

### Profile

- Simulation `report.txt` overall `[Pipe Distribution]` has no CUBE row or CUBE instr = 0.
- Simulation `report.txt` overall `[CUBE/MMA]` section absent or MMAD = 0.
- Simulation `report.txt` overall `[Pipe Distribution]` SCALAR instr% > 50% (scalar-heavy address computation typical of manual reduction).
- Simulation `report.txt` overall `[TRACE Events]` arithmetic breakdown has MADD > 0 (multiply-accumulate pattern present but routed to SCALAR/VECTOR instead of CUBE).
- Simulation `report.txt` overall `[Pipeline Flows]` has no CUBE-related flows (CUBEToFIXP, CUBEToMTE1, MTE1ToCUBE all absent).

## Rewrite Goal

Apply in two phases. Phase 1 is always applicable; Phase 2 depends on `tl.dot` compiling successfully on the target.

**Phase 1 — Structural optimization (always apply first):**

- eliminate redundant loads in the K-reduction loop (restructure loop nest so input tiles are shared)
- widen output tiling (increase BLOCK_OUT/BLOCK_N to amortize fixed overhead)
- use `tl.static_range` for the K loop to enable compiler pipelining

**Phase 2 — `tl.dot` replacement (apply if Phase 1 is already done and `tl.dot` compiles):**

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

On Ascend Triton, make the two `tl.dot` operand dtypes match, but do not lower them to `fp16` by default when the real operator inputs are already `fp32` or `fp16`. The default pattern is:

- keep `a` and `b` in the loaded input dtype when both sides already match
- use `acc`: `fp32`
- keep the fused epilogue in `fp32`
- store in the output tensor dtype

Only add an explicit cast when the two operands would otherwise differ in dtype and you have validated that the cast is both numerically acceptable and performance-positive.

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

**Phase 1 (structural):**
- reduced redundant data movement through MTE2 (often the dominant win — 10-30× observed)
- better MTE2 amortization from wider output tiling
- compiler loop pipelining from `tl.static_range`

**Phase 2 (`tl.dot` replacement):**
- more regular lowering of the main `K` loop
- better fit for CUBE/matmul code generation
- less scalar-heavy reduction structure
- better amortization of fused epilogues on larger `M` and `N`

## Main Risks

- tile setup overhead can hurt small shapes
- larger tiles can increase register or UB pressure
- forcing `fp16` dot inputs can change `fp32` input semantics
- `tl.dot` may crash the Bisheng compiler depending on CANN version and kernel shape (confirmed on A5 and 910B2); verify `tl.dot` compiles before relying on it

## Matrix×Vector Grid Restructuring (N=1 → N≥2)

When each program currently processes one output row (matrix×vector, N=1), `tl.dot` may crash the Bisheng compiler depending on CANN version and kernel shape. Restructure the grid so each program handles multiple output rows:

```python
# detect: grid=(B,) — one program per batch row, output is 1D
pid = tl.program_id(axis=0)
x_row_ptr = x_ptr + pid * X_STRIDE
# ... computation produces a single scalar or 1D result per program
```

```python
# restructure: each program handles ROWS_PER_PROGRAM rows → N = ROWS_PER_PROGRAM ≥ 2
pid = tl.program_id(axis=0)
row_start = pid * ROWS_PER_PROGRAM
row_offs = row_start + tl.arange(0, ROWS_PER_PROGRAM)
row_mask = row_offs < B

# Load x as [ROWS_PER_PROGRAM, BLOCK_IN] instead of [BLOCK_IN]
x_ptrs = x_ptr + row_offs[:, None] * X_STRIDE + in_offs[None, :]
x_block = tl.load(x_ptrs, mask=row_mask[:, None] & in_mask[None, :], other=0.0)

# Now w_tile [BLOCK_OUT, BLOCK_IN] × x_block [BLOCK_IN, ROWS_PER_PROGRAM] works
# N dimension = ROWS_PER_PROGRAM > 1 → tl.dot works
```

## Structural Optimization Before `tl.dot`

When `tl.dot` cannot be used (compiler crash, shape constraint, or UB overflow), the kernel can still benefit from structural improvements that reduce redundant data movement. Apply these **before** attempting `tl.dot` — they often yield the majority of the speedup.

### Eliminate redundant loads in the K-reduction loop

When the same input tile is loaded multiple times across outer-loop iterations (e.g., per output element or per window), restructure the loop nest so each input tile is loaded once and shared across all output elements that need it.

```python
# detect: x loaded inside the outer loop — redundant per output element
for out_tile in range(0, OUT_F, BLOCK_OUT):
    for in_tile in range(0, IN_F, BLOCK_IN):
        x_vec = tl.load(x_ptr + in_offs, ...)          # reloaded every out_tile iteration
        w_tile = tl.load(w_ptr + out_offs * IN_F + in_offs, ...)
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)
```

```python
# restructure: x loaded once per in_tile, shared across out_tile
for in_tile in range(0, IN_F, BLOCK_IN):
    x_vec = tl.load(x_ptr + in_offs, ...)              # loaded once
    for out_tile in range(0, OUT_F, BLOCK_OUT):
        w_tile = tl.load(w_ptr + out_offs * IN_F + in_offs, ...)
        acc += tl.sum(w_tile * x_vec[None, :], axis=1)  # reused across out_tile
```

### Widen output tiling

Increase the number of output elements computed per program to amortize fixed overhead (pointer setup, accumulator init) and improve MTE2 throughput.

```python
# detect: BLOCK_OUT=32 — many small output tiles, high per-tile overhead
```

```python
# widen: BLOCK_OUT=128 — 4× fewer tiles, better MTE2 amortization
```

### Use `tl.static_range` for the K loop

Replace `while` or `range` with `tl.static_range` to enable compiler loop pipelining, even when the body still uses `tl.sum(a*b)` instead of `tl.dot`.

### Conv2d via im2col with manual K-reduction

Conv2d kernels that flatten to im2col matmul but use `tl.sum(a*b)` instead of `tl.dot`. Unlike pure matmul, the K loop requires coordinate decoding (`//` and `%`) to map flat indices back to `(ci, kh, kw)`, and input positions depend on decoded coordinates with padding/stride/dilation.

Phase 1 — structural: replace `//`/`%` with compile-time-known strides where possible (same as Cat 1 Manifestation A), ensure x loads are not redundant across K iterations.

Phase 2 — `tl.dot` replacement: the key change is replacing `tl.sum(a_tile * b_tile, axis=1)` with `tl.dot(a_tile, b_tile_t)` **and** loading the weight tile in transposed layout `[BLOCK_K, BLOCK_N]` instead of `[BLOCK_N, BLOCK_K]`.

```python
# detect: conv kernel with im2col, K loop doing tl.sum(a*b)
# Each program handles one output tile, grid = (B * OH * OW, CO)
pid_m = tl.program_id(axis=0)
pid_n = tl.program_id(axis=1)
m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

# Decode m -> (b, oh, ow) and k -> (ci, kh, kw) with scalar //
ohow = OH * OW
b_idx = m_offs // ohow
rem = m_offs - b_idx * ohow
oh = rem // OW
ow = rem - oh * OW

acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for k0 in tl.static_range(0, K, BLOCK_K):
    k_offs = k0 + tl.arange(0, BLOCK_K)
    khkw = KH * KW
    ci = k_offs // khkw
    remk = k_offs - ci * khkw
    kh = remk // KW
    kw = remk - kh * KW

    ih = oh[:, None] * STRH - PADH + kh[None, :] * DILH
    iw = ow[:, None] * STRW - PADW + kw[None, :] * DILW
    in_bounds = (ih >= 0) & (iw >= 0) & (ih < H) & (iw < W)

    x_offsets = b_idx[:, None] * x_stride_n + ci[None, :] * x_stride_c + ih * x_stride_h + iw * x_stride_w
    a_tile = tl.load(x_ptr + x_offsets, mask=in_bounds, other=0.0).to(tl.float32)

    w_offsets = n_offs[None, :] * K + k_offs[:, None]   # weight [CO, K] layout
    b_tile = tl.load(w_ptr + w_offsets, mask=..., other=0.0).to(tl.float32)

    acc += tl.sum(a_tile * b_tile, axis=1)               # ← manual K-reduction, NOT tl.dot
```

```python
# transform: replace tl.sum(a*b) with tl.dot + weight layout transpose
# a_tile stays [BLOCK_M, BLOCK_K], b_tile becomes [BLOCK_K, BLOCK_N]

for k0 in tl.static_range(0, K, BLOCK_K):
    k_offs = k0 + tl.arange(0, BLOCK_K)
    khkw = KH * KW
    ci = k_offs // khkw
    remk = k_offs - ci * khkw
    kh = remk // KW
    kw = remk - kh * KW

    ih = oh[:, None] * STRH - PADH + kh[None, :] * DILH
    iw = ow[:, None] * STRW - PADW + kw[None, :] * DILW
    in_bounds = (ih >= 0) & (iw >= 0) & (ih < H) & (iw < W)

    x_offsets = b_idx[:, None] * x_stride_n + ci[None, :] * x_stride_c + ih * x_stride_h + iw * x_stride_w
    a_tile = tl.load(x_ptr + x_offsets, mask=in_bounds, other=0.0).to(tl.float32)

    # Load weight tile as transposed block: [BLOCK_K, BLOCK_N]
    w_offsets_t = k_offs[:, None] + n_offs[None, :] * K
    b_tile_t = tl.load(w_ptr + w_offsets_t, mask=..., other=0.0).to(tl.float32)

    acc += tl.dot(a_tile, b_tile_t)                      # CUBE-side matmul
```

## Related Patterns

- `software-pipeline`: use it after `classic-matmul` when the tiled structure already exists but profiling still shows load-then-compute stalls.
- `tiling`: use it when the kernel already has the right tiled structure, but block size or intermediate footprint is still too large for UB.

## What To Verify After Applying

- confirm logical `A`, `B`, and `C` shapes
- confirm the second operand really arrives as `[BK, BN]`
- confirm masks on `M`, `N`, and `K`
- confirm fused bias or activation broadcasting
- if you introduce dispatch, validate each dispatched regime explicitly
- benchmark against the previous implementation
- record whether the win came from lower scalar pressure, better matmul lowering, or better epilogue amortization
