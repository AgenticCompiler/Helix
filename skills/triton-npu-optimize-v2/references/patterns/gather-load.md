# NPU Gather Operation Optimization Pattern

## Summary

Stage gather-like input through contiguous loads before selecting indexed values so the kernel reduces expensive discrete global-memory reads on Ascend NPU.

## Problem Description

On Huawei Ascend NPU devices, direct discrete memory access patterns (gather operations) suffer from poor performance when accessing global memory. The NPU architecture favors contiguous memory access and has significantly higher bandwidth in shared memory compared to global memory for discrete access patterns.

## Optimization Strategy

Convert direct discrete global memory access into a two-phase approach:
1. **Contiguous data loading**: Load the entire source array from global memory to shared memory using contiguous access patterns
2. **Discrete selection**: Perform gather operations on the fast shared memory instead of slow global memory

### Key Principles

1. **Identify discrete access patterns**: Look for code that uses index arrays to access non-contiguous memory locations
2. **Leverage shared memory**: Utilize NPU's high-bandwidth shared memory for discrete operations
3. **Maintain semantic equivalence**: Ensure the logical result remains identical to the original implementation
4. **Consider memory footprint**: Only apply when the source array fits reasonably in shared memory

## Detection Pattern

Look for code patterns like:

```python
# Problematic: Direct discrete global memory access on NPU
idx = tl.load(idx_ptr + rn * stride_idx)
val = tl.load(x_ptr + idx * stride_x)  # Discrete access pattern

# Problematic: Index-based memory access
indices = compute_indices()
data = tl.load(base_ptr + indices * stride)  # Scattered loading
```

## Optimization Example

### Before Optimization (GPU-style)

```python
@triton.jit
def pick_kernel(
    x_ptr, idx_ptr, y_ptr,
    stride_x, stride_idx, stride_y,
    M: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    # Load indices
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Problem: Direct discrete global memory access (slow on NPU)
    val = tl.load(x_ptr + idx * stride_x, mask=mask)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

### After Optimization (NPU-optimized)

```python
@triton.jit
def pick_kernel(
    x_ptr, idx_ptr, y_ptr,
    stride_x, stride_idx, stride_y,
    M: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    rm = tl.arange(0, M)  # Full range for source array
    rn = tl.arange(0, N)  # Range for indices

    # Load indices
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Optimization: Two-phase approach for NPU
    # 1. Contiguous load of entire source array to shared memory
    x_shared = tl.load(x_ptr + rm * stride_x)

    # 2. Discrete access on fast shared memory using tl.gather
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

## Use When

1. **Discrete access patterns**: When using index arrays to access non-contiguous memory
2. **Small to medium source arrays**: When the source array can fit in shared memory
3. **Performance-critical sections**: Where gather operations are bottleneck

## Signals

### Code

- Code uses index arrays to access non-contiguous memory locations on the hot path.
- The gather source array is small or medium enough that contiguous staging in shared memory is plausible.
- Direct global-memory gather reads dominate more than the surrounding arithmetic.

## Avoid When

1. **Large source arrays**: When M is too large for shared memory capacity
2. **Already contiguous access**: When memory access patterns are already sequential
3. **GPU targets**: This optimization is NPU-specific and may not benefit GPU architectures
4. **Single-element access**: When only accessing a few discrete elements

## What To Verify After Applying

- Verify the source array size `M` is still reasonable for shared memory after the rewrite.
- Verify the kernel stages the source array contiguously before calling `tl.gather`.
- Verify boundary masking and semantic equivalence with the original gather behavior.

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `18_Index`
- `20_Gather`

## NPUKernelBench round narratives (pilot: `18_Index`, 2026-05-08, log-backed)

*Operator in this excerpt: **`18_Index`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `18_Index`

**`opt-round-1` (parent `baseline`)** — `18_Index/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline `index_select` behaved like scattered gather reads plus per-lane rank decode, with dominant cases actually contiguous in `inner_size`.
- **Change:** Replaced linearized elementwise gather with row-copy-style programs over contiguous `inner_size` windows; repaired launch shape from `(selected_row, inner_block)` (coreDim overflow) to per-selected-row launches with inner loops.
- **Evidence:** Correctness passed; `compare-perf` vs baseline in `attempts.md` reported **Avg +74.8%**, **Geomean 11.03x**, **Total 41.17x**; promoted.
- **Interpretation:** `18_Index` is the canonical "gather semantics, copy-like layout" case: stage/select contiguous rows instead of issuing discrete global gather traffic per element.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators in this file’s excerpt: **`20_Gather`** (other batch-4 kernels map to different pattern cards). Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `20_Gather`

**`opt-round-2` (parent `opt-round-1`)** — `20_Gather/opt-round-2/attempts.md`

- **Kernel / round / parent:** `20_Gather` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 simplified addresses but case-5 still spent **`~350 ms`** in **`_gather_copy_kernel`** with **`int64`** index loads on **`int64`** indices (`attempts.md`).
- **Change:** **`int32` index fast path** when axis size fits **signed 32-bit**—wrapper narrows contiguous **`int64`** indices pre-launch; kernel skips inner casts.
- **Evidence:** Correctness passed; **Avg +12.1%**, **1.16×** geomean vs baseline (`attempts.md`); **promoted** until r3; added **`Cast`** host cost noted in attempts.
- **Interpretation:** Gather is **index-bandwidth** sensitive—narrow indices help, but the dominant dim-0 case still needed **shape specialization** next.

**`opt-round-3` (parent `opt-round-2`)** — `20_Gather/opt-round-3/attempts.md`

- **Kernel / round / parent:** `20_Gather` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Profiler showed **scalar-only** hot kernel with extreme **Block Dim** on dominant case (`attempts.md`—see **`scalar-latency-traps.md`**).
- **Change:** **Rank-2 `dim=0` specialization**: multiple output rows + wide column block per program; generic kernel fallback.
- **Evidence:** Correctness passed; **Avg +23.1%**, **1.34×** geomean vs baseline (`attempts.md`); **promoted** parent for final **`layout-store-and-block-pointers.md`** cleanup in r5.
- **Interpretation:** When gather axis aligns with **contiguous source rows**, rewrite from “indexed gather” toward **memcpy-shaped row tiles**—this card overlaps **`layout-store-and-block-pointers.md`** for the contiguous slices.
