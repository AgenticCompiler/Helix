---
id: pooling-clip-window-closed-divisor
priority: normal
---

# Pooling Clip-Window And Closed-Form Divisor Pattern

## Summary

Repair sliding-window pooling kernels by aligning the inner loop nest and divisor logic with the PyTorch/CUDA reference: compute each output window's **clipped input bounds once**, use a **closed-form window volume** for the average divisor, and iterate **input coordinates inside the clipped range** instead of scanning the full `KERNEL_D×KERNEL_H×KERNEL_W` cube with per-tap validity masks and runtime counting.

This is a **portable structural repair** for avg/max-like window reductions over affine NCDHW/NCHW layouts. It is complementary to, not a substitute for, outer launch tiling (`flat-index-decode-tiling`), A5 SIMT-only launch mode, or inner-W slab gather on HIVM paths.

## Structural Repair Precedence

Apply in this order for discrete pooling kernels on A5:

1. **Outer structure**: if the kernel walks flat `numel(out)` with hot-path `//` / `%` coordinate decode, repair with `flat-index-decode-tiling` first (row-column or rank-aware tiles).
2. **Inner structure (this pattern)**: replace kernel-index loops + per-tap divisor counting with clip-window bounds + closed-form divisor.
3. **Launch mode**: if the kernel remains scalar/index dominated on A5, evaluate `a5-force-simt-only-discrete-access` after steps 1–2.
4. **W-slab gather**: evaluate `pooling-inner-w-slab-gather` only when profiling shows the hot path is HIVM-friendly and slab+gather beats SIMT gather on the target device. Do not assume W-slab is always better than SIMT discrete access.

Closed-form divisor is **related to but not the same as** generic loop-invariant hoisting (`loop-invariant-hoisting`): hoisting moves unchanged expressions out of a loop, while this pattern **replaces an O(window_volume) accumulation** (`valid_count += 1`) with an **O(1) algebraic formula** derived from the same clip bounds used for loads.

## Use When

- The operator is **AvgPool / MaxPool** or another **fixed-kernel window reduction** over an affine layout (NCDHW, NCHW, or collapsed batch×channel rows).
- The hot path uses **`for kd/kh/kw in range(KERNEL_*)`** with per-tap **`valid_*` / `safe_*` / `window_mask`** and/or **`count += tl.where(...)`** to derive the average divisor.
- Padding, `count_include_pad`, `ceil_mode`, or edge outputs make many lanes use **partial windows**, but the mapping from output coordinates to input bounds is still **static and affine**.
- Correctness can be checked against PyTorch `avg_pool{2,3}d` / `max_pool{2,3}d` across padding, `count_include_pad=False`, and boundary shapes.
- You want a low-risk structural cleanup **before** autotune, W-slab, or SIMT-only experiments.

## Avoid When

- Indices are value-dependent or the window is not a fixed affine span (true gather/scatter with irregular offsets).
- The kernel is already dominated by Cube/matmul work; scalar window bookkeeping is not the bottleneck.
- You cannot prove equivalence between padded-window divisor semantics and clipped-window load semantics (`count_include_pad=True` uses **pre-clip padded volume** for divisor but **post-clip input coordinates** for loads — same as PyTorch/CUDA).
- Applying clip-window would require dynamic loop trip counts per lane that the backend cannot lower safely, and the fallback kernel-index path is already fast enough.
- The target path is HIVM W-slab and a measured winner; do not stack conflicting compile modes (SIMT-only vs HIVM slab).

## Signals

### Code

- Innermost work is **`for kw in range(KERNEL_W):`** with **`in_w = start_w + kw`**, then **`valid_w = (in_w >= 0) & (in_w < in_w)`** and **`safe_w = tl.where(valid_w, in_w, 0)`**.
- Average pooling uses **`valid_count += tl.where(window_mask, 1, 0)`** or **`padded_count += ...`** inside the `kd/kh/kw` loops.
- **`start_* = o* * STRIDE_* - PAD_*`** is computed once per output lane, but clip bounds are **not** reused for divisor and loads consistently.
- **`x_base = n * stride_n + c * stride_c`** is recomputed inside every kernel tap instead of before the inner loops.

### Profile

- High **`aiv_scalar_ratio`** on a pooling kernel whose math is mostly load + add + divide.
- Many compare/mask/where operations relative to useful loads, especially on padded or `count_include_pad=False` cases.
- Gains from SIMT-only or block tuning plateau while inner loop body remains mask-heavy.

### IR

- Repeated **`subi/minsi/maxsi`** and **`select`** inside lowered loops over **`kd/kh/kw`**.
- Integer increment chains for divisor counting inside the same loops as loads.

## Optimization Strategy

### Step 1 — Compute clip bounds once per output lane

Mirror PyTorch/CUDA avg pool forward. For each output coordinate, first compute the **padded logical window**, then clip to the input tensor:

```python
start_d = od * STRIDE_D - PAD_D
start_h = oh * STRIDE_H - PAD_H
start_w = ow * STRIDE_W - PAD_W

tend_d = tl.minimum(start_d + KERNEL_D, in_d + PAD_D)
tend_h = tl.minimum(start_h + KERNEL_H, in_h + PAD_H)
tend_w = tl.minimum(start_w + KERNEL_W, in_w + PAD_W)

tstart_d = tl.maximum(start_d, 0)
tstart_h = tl.maximum(start_h, 0)
tstart_w = tl.maximum(start_w, 0)

tend_d = tl.minimum(tend_d, in_d)
tend_h = tl.minimum(tend_h, in_h)
tend_w = tl.minimum(tend_w, in_w)

d_len = tl.maximum(tend_d - tstart_d, 0)
h_len = tl.maximum(tend_h - tstart_h, 0)
w_len = tl.maximum(tend_w - tstart_w, 0)
```

Empty window (`d_len == 0` or `h_len == 0` or `w_len == 0`): leave accumulator at zero; set divisor fallback so `0 / 1 = 0`, matching reference early-return behavior.

### Step 2 — Closed-form divisor before the load loops

Do **not** count taps inside `kd/kh/kw`.

```python
if HAS_DIVISOR_OVERRIDE:
    divisor = DIVISOR_OVERRIDE
elif COUNT_INCLUDE_PAD:
    # use pre-clip padded spans
    tend_d_p = tl.minimum(start_d + KERNEL_D, in_d + PAD_D)
    tend_h_p = tl.minimum(start_h + KERNEL_H, in_h + PAD_H)
    tend_w_p = tl.minimum(start_w + KERNEL_W, in_w + PAD_W)
    d_len_div = tl.maximum(tend_d_p - start_d, 0)
    h_len_div = tl.maximum(tend_h_p - start_h, 0)
    w_len_div = tl.maximum(tend_w_p - start_w, 0)
else:
    d_len_div, h_len_div, w_len_div = d_len, h_len, w_len

divisor = (d_len_div * h_len_div * w_len_div).to(tl.float32)
divisor = tl.where(divisor > 0, divisor, 1.0)
```

For interior tiles with **`padding=0`** and fully in-bounds windows, this collapses to the constant **`KERNEL_D * KERNEL_H * KERNEL_W`** — a host-dispatch fast path is optional.

### Step 3 — CUDA-style clip-window load loop

Replace kernel-offset iteration with **absolute clipped input indices**:

```python
x_base = n * stride_n + c * stride_c  # hoist before loops
acc = tl.zeros(..., dtype=tl.float32)

for kd in range(KERNEL_D):
    ti = tstart_d + kd
    d_ok = kd < d_len
    for kh in range(KERNEL_H):
        hi = tstart_h + kh
        h_ok = kh < h_len
        for kw in range(KERNEL_W):
            wi = tstart_w + kw
            w_ok = kw < w_len
            load_mask = lane_mask & d_ok & h_ok & w_ok
            off = x_base + ti * stride_d + hi * stride_h + wi * stride_w
            acc += tl.load(x_ptr + off, mask=load_mask, other=0.0).to(tl.float32)

tl.store(..., acc / divisor, mask=lane_mask)
```

Notes for Triton/Ascend:

- Keep **`range(KERNEL_D)`** as a **constexpr upper bound**; use **`kd < d_len`** to skip invalid taps when per-lane clip lengths differ. This is the practical equivalent of CUDA's `for (ti = tstart; ti < tend; ++ti)` under SIMT divergence.
- Remove **`valid_*` / `safe_*`** when loads use **`tstart + kd`** and **`kd < d_len`**.
- On Ascend, prefer **`tl.zeros([BLOCK], dtype=tl.float32)`** over **`tl.zeros_like(..., dtype=...)`** when creating divisor tensors.

### Step 4 — Optional host fast paths (CUDA-inspired)

| Fast path | Guard | Effect |
|-----------|-------|--------|
| Constant divisor | `padding=0`, window fully inside input, `count_include_pad=True`, no override | `divisor = KERNEL_D*KERNEL_H*KERNEL_W` |
| No length masks | same as above | drop `d_ok/h_ok/w_ok`; inner loops become unmasked loads |
| `KERNEL_W` specialization | `kernel_w in 1..7` | separate `@triton.jit` instances; helps unrolling on CUDA-like backends |
| Interior / boundary split | mixed tiles | run clip-window on interior; keep generic masked path only on boundary bands |

### Step 5 — Combine with outer tiling and SIMT

- Row-column tiling: compute clip bounds on **`(row, col)`** tiles; reuse the same formulas with broadcast shapes `(BLOCK_ROWS, 1)` and `(1, BLOCK_SIZE)`.
- After this structural repair, retune **`BLOCK_*`**, grid decomposition, and **`force_simt_only=True`** on A5 if profiling still shows scalar dominance.

## Code Transformation

### Anti-pattern (kernel-index + per-tap count)

```python
for kd in range(KERNEL_D):
    in_d_index = start_d + kd
    valid_d = (in_d_index >= 0) & (in_d_index < in_d)
    safe_d = tl.where(valid_d, in_d_index, 0)
    for kh in range(KERNEL_H):
        ...
        for kw in range(KERNEL_W):
            window_mask = valid_d & valid_h & valid_w
            acc += tl.load(x_ptr + off(safe_d, safe_h, safe_w), mask=window_mask, other=0.0)
            if not HAS_DIVISOR_OVERRIDE:
                valid_count += tl.where(window_mask, 1, 0)
divisor = valid_count.to(tl.float32)
```

### Target (clip bounds + closed-form divisor + clip-window loads)

```python
tstart_d, tstart_h, tstart_w, d_len, h_len, w_len = clip_window(...)
divisor = closed_divisor(start_*, ..., COUNT_INCLUDE_PAD, ...)
x_base = n * stride_n + c * stride_c
for kd in range(KERNEL_D):
    ti = tstart_d + kd
    for kh in range(KERNEL_H):
        hi = tstart_h + kh
        for kw in range(KERNEL_W):
            wi = tstart_w + kw
            if kd < d_len and kh < h_len and kw < w_len:  # use vector masks in Triton
                acc += tl.load(x_ptr + x_base + ti*stride_d + hi*stride_h + wi*stride_w)
```

## Failure Modes And Anti-signals

- Using **post-clip lengths** for divisor when **`count_include_pad=True`** — changes semantics vs PyTorch.
- Using **pre-clip coordinates** for loads — reads out of bounds or wrong padding behavior.
- Applying only closed-form divisor but leaving kernel-index + `valid_*` loops — partial gain only.
- Assuming clip-window removes all masks on **border outputs**; edge lanes still need `kd < d_len` guards.
- Replacing discrete SIMT gather with W-slab on A5 without measurement — may regress vs pure SIMT (HIVM / compile-path conflict).
- Nested `@triton.jit` helpers that use unsupported APIs (`tl.zeros_like(..., dtype=...)`) on Ascend.

## What To Verify After Applying

- Correctness vs PyTorch for all dtypes in the operator contract, especially:
  - `padding > 0`
  - `count_include_pad=False`
  - `ceil_mode=True` boundary outputs
  - `divisor_override`
  - empty / partial windows on every spatial edge
- Inner hot path no longer contains **`valid_count +=`** / **`padded_count +=`** in `kd/kh/kw` loops.
- Clip bounds match CUDA reference on sampled outputs (compare against manual golden for one lane).
- Benchmark on representative JSON cases: interior-heavy shapes vs padding-heavy shapes separately.
- If combined with SIMT-only, record A5 evidence and retune `num_warps` / block sizes after the structural change.
- Document whether W-slab was considered and rejected or deferred.

## Related Patterns

- `flat-index-decode-tiling` — outer output traversal / row-column tile before inner window repair
- `loop-invariant-hoisting` — hoist `x_base`, masks, and bounds; closed-form divisor is algebraic LICM plus counting elimination
- `a5-force-simt-only-discrete-access` — launch mode after structural repair
- `pooling-inner-w-slab-gather` — alternative inner-W memory strategy; compare on device, do not assume dominance
- `exact-tile-no-boundary-fast-path` — drop `d_ok/h_ok/w_ok` when tile is fully interior
- `scalar-latency-traps` — remove redundant `safe_*`, narrow masks, and unsupported APIs
- `algebraic-optimization` — closed-form divisor is an algebraic replacement for incremental counting
