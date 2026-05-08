# Hierarchical Tiling Optimization Pattern (UB Overflow Prevention)

## Summary

Reduce per-program working-set size through hierarchical or sub-block tiling so large tiles, intermediates, or multi-tensor loads fit UB safely without collapsing overall task structure.

## Use When

- Block sizes, live intermediates, or multi-tensor loads risk UB overflow or poor locality.
- The main problem is working-set size and memory footprint, not the need for a completely different kernel structure.

## Signals

### Code

- Large `BLOCK_SIZE` values, multiple tensor loads, or heavy intermediates keep too much data live per program.
- The kernel already has a reasonable overall structure, but it still needs smaller sub-blocks to control UB usage.
- Runtime failures or memory access violations appear when block sizes increase on NPU.

## Problem Description

**Root Cause:**
- Ascend NPU has limited on-chip Unified Buffer (192KB on Atlas 800T A2/A3)
- Large `BLOCK_SIZE` values cause excessive memory consumption per program instance
- When loading multiple tensors or performing complex operations within a block, UB usage can overflow

**Symptoms:**
- Runtime errors indicating UB overflow
- Kernel failures with large block sizes
- Memory access violations on NPU

## Optimization Strategy

Introduce **hierarchical tiling** (also called sub-blocking) to further subdivide large blocks:

Choose this pattern when the main problem is **working-set size**. The question it answers is:

- how should the kernel reduce per-program memory footprint so tiles and intermediates fit UB safely

If the kernel should first be re-expressed as a standard tiled matmul, prefer `classic-matmul`.
If the tiled loop already exists and the remaining problem is poor memory/compute overlap, prefer `software-pipeline`.

### Key Principles

1. **Two-level blocking**: Separate task scheduling from memory management
   - Keep main `BLOCK_SIZE` for task scheduling (coreDim compliance)
   - Introduce `BLOCK_SIZE_SUB` for processing data in smaller batches

2. **Process in loops**: Use inner loops to process sub-blocks sequentially
   - Reduce peak memory usage by processing data in smaller chunks
   - Maintain reasonable coreDim values for efficient task scheduling
   - Control UB usage through smaller batch sizes

3. **Balance performance and memory**:
   - Small enough to fit within UB capacity
   - Large enough to maintain reasonable performance
   - Aligned with memory access patterns (32-byte alignment)

## Detection Pattern

Look for code with these characteristics:

1. **Large BLOCK_SIZE values** (> 8192)
2. **Multiple tensor loads** within a single block
3. **Complex operations** that require intermediate storage
4. **UB overflow errors** at runtime

### Problematic Code Patterns

```python
# Problem: Large block size + multiple loads = UB overflow
@triton.jit
def kernel(inp, mask1, mask2, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Loading multiple large tensors at once
    mask = offsets < N
    data1 = tl.load(inp + offsets, mask=mask)
    data2 = tl.load(mask1 + offsets, mask=mask)
    data3 = tl.load(mask2 + offsets, mask=mask)

    # UB overflow here!
    result = complex_operation(data1, data2, data3)
    tl.store(out + offsets, result, mask=mask)
```

## Code Example

### Before Optimization (UB Overflow)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Loading all data at once causes UB overflow
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)

    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)
```

**Issues:**
- Single large block loads all data at once
- Multiple tensors occupy UB simultaneously
- Risk of overflow with large BLOCK_SIZE values

### After Optimization (Hierarchical Tiling)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                      BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # Calculate the number of sub-blocks to process
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    # Process in blocks to avoid UB overflow
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # Load and process data in batches (smaller UB footprint)
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # First write the original data
        tl.store(out + offsets, input_vals, mask=mask)

        # Then overwrite the target value at positions that need filling
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)
```

**Improvements:**
- Data processed in smaller sub-blocks
- Reduced peak UB usage
- Maintains task scheduling efficiency with large main BLOCK_SIZE

### Host Code Configuration

```python
def masked_fill(inp, mask, value):
    N = inp.numel()

    # Two-level blocking strategy
    MAIN_BLOCK_SIZE = 32768  # Ensure coreDim compliance (N / 32768 < 65535)
    SUB_BLOCK_SIZE = 1024    # Control UB usage (process in smaller chunks)

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, mask, value, out, N,
                           MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

## Guidelines for Sub-Block Size Selection

**SUB_BLOCK_SIZE should be:**

1. **UB-safe**: Small enough to fit within UB capacity
   - Simple element-wise operations: 1024-2048
   - Operations with multiple tensors: 512-1024
   - Complex reductions: 256-512

2. **Performance-aware**: Large enough to maintain reasonable performance
   - Avoid too-small blocks that increase loop overhead
   - Balance between memory usage and computation efficiency

3. **Alignment-aware**: Divisible by or aligned with memory access patterns
   - 32-byte alignment requirement
   - Vector width considerations (typically 128/256 elements)

## Avoid When

1. **Small BLOCK_SIZE** No significant memory pressure
2. **Simple operations** with single tensor - UB usage is minimal
3. **Already optimized** with sub-blocking present
4. **Structure is the real problem** - if the current kernel is really a manual matmul or reduction that should first become a regular tiled `tl.dot` loop

## What To Verify After Applying

- Verify the chosen `BLOCK_SIZE_SUB` fits the operation type and keeps the working set UB-safe.
- Verify the inner sub-block loop actually reduced peak live data instead of only adding loop overhead.
- Verify both kernel signature and host launch code pass the new block-size parameters consistently.

## Expected Performance Impact

**Memory Usage:**
- UB usage reduced by factor of `BLOCK_SIZE / BLOCK_SIZE_SUB`
- Enables processing larger arrays without overflow

**Performance Trade-offs:**
- **Pros**: Enables kernels that would otherwise overflow UB
- **Cons**: Small loop overhead for sub-block iteration (typically < 5%)
- **Net effect**: Enables functionality that was previously impossible

**Typical Results:**
- UB overflow kernels become functional
- Performance impact: -5% to +10% depending on operation complexity
- Enables larger batch sizes and more efficient task scheduling

## Related Patterns

- `classic-matmul`: use it first when the real problem is that a manual reduction should become a tiled matmul structure at all.
- `software-pipeline`: combine it only after the footprint already fits UB, because pipelining deliberately keeps multiple tiles live.

## Code Transformation Pattern

**Step 1: Add sub-block parameter**
```python
# Before
def kernel(..., BLOCK_SIZE: tl.constexpr):

# After
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
```

**Step 2: Calculate sub-block count**
```python
num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
```

**Step 3: Wrap computation in loop**
```python
for sub_block_idx in range(num_sub_blocks):
    # Compute sub-block offsets
    sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
    offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)

    # Load, compute, store for this sub-block
    ...
```

**Step 4: Update host code**
```python
# Before
kernel[grid](..., BLOCK_SIZE)

# After
kernel[grid](..., BLOCK_SIZE, BLOCK_SIZE_SUB)
```

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `10_LayerNorm`

**`opt-round-15` (parent `opt-round-14`)** — `10_LayerNorm/opt-round-15/attempts.md`

- **Kernel / round / parent:** `10_LayerNorm` / `opt-round-15` / `opt-round-14`.
- **Pre-change scenario:** Huge-row regime still uses a **1024-wide** apply tile in r14; hypothesis tests whether **2048-wide** apply tile can cut passes on the widest rows (`opt-note.md` round 15 theme).
- **Change:** Introduce 2048-wide apply tiling on the huge-row path with UB-safe masking/repair after first compile failures.
- **Evidence:** Correctness passed after UB repair, but `compare-perf` **regressed heavily** on the huge-row case vs r14; **validated branch, not promoted** (`opt-note.md`).
- **Interpretation:** Hierarchical tiling must respect UB/register footprints—doubling tile width without occupancy proof is a negative experiment worth recording on this card.

### `2_GroupNormSwish`

**`opt-round-3` (parent `opt-round-2`)** — `2_GroupNormSwish/opt-note.md` + `opt-round-3/attempts.md`

- **Kernel / round / parent:** `2_GroupNormSwish` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Apply kernel tile does not yet match A5-heavy channel×spatial mix after r2 streaming guard work (`opt-note.md` arc).
- **Change:** **Retile apply kernel** over channel+spatial axes to rebalance work per program (attempts narrative + operator theme).
- **Evidence:** Correctness passed; geomean and total speedup improve materially vs r2 per `opt-note.md` / attempts tables—**promoted** in-session before r4 widening.
- **Interpretation:** GroupNorm apply is a 2D tile problem; axis retile is the first-order hierarchical tiling lever.

**`opt-round-4` (parent `opt-round-3`)** — `2_GroupNormSwish/opt-note.md` + `opt-round-4/attempts.md`

- **Kernel / round / parent:** `2_GroupNormSwish` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** r3 retile wins; spatial axis still leaves headroom on largest spatial cases in the suite.
- **Change:** **Enlarge spatial tile** footprint for A5-heavy cases while keeping channel tiling from r3 coherent.
- **Evidence:** Further geomean jump vs r3 on the benchmark mix (`opt-note.md`); **promoted** over r3 in the recorded arc.
- **Interpretation:** Second-pass spatial widening after channel retile is standard hierarchical tuning—watch for UB regressions in later rounds when widening continues.

### Other kernels in the eight

`1_GELU` launch-tier narrative belongs on **`autotune.md`**. `1_RotaryMul`, `2_SwiGLU`, `10_SwigluQuant`, `11_DequantSwigluQuant`, and `11_GroupNorm` do not record the same **`BLOCK` vs sub-block UB overflow** story as `10_LayerNorm` r15 in `opt-note.md`; any UB pressure there stays in layout/PMR/profiler notes on other cards—do not fabricate tiling-card entries without explicit tile/UB evidence.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `13_Cat`

**`opt-round-17` (parent `opt-round-15`)** — `13_Cat/opt-round-17/attempts.md` (see also `opt-note.md` rounds 12–17)

- **Kernel / round / parent:** `13_Cat` / `opt-round-17` / `opt-round-15`.
- **Pre-change scenario:** Dim-0 flat copy path already reached **8192**-element tiles in r15; dominant cases still transfer-bound on the widest contiguous moves (`opt-note.md` ladder).
- **Change:** Raised flat-path **`BLOCK_SIZE` to 16384** for the heavy dim-0 copy kernel while keeping validated row-path integration from earlier rounds.
- **Evidence:** Correctness passed; each ladder step r12→r17 **promoted** in `opt-note.md` until r17 best; r18 **32768** probe **regressed** canonical workload (validated branch only).
- **Interpretation:** Same “width ladder until saturation” lesson as elementwise pilots—tiling here is **transfer tile width**, not 2D convolution tiles.

### `14_AdaptiveInstanceNormalization2DBackward`

**`opt-round-5` (parent `opt-round-4`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-5/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Row batching from r3–r4 still used **`BLOCK_N=256`** on large spatial sizes where column trips dominate (`attempts.md`).
- **Change:** Large-spatial branch keeps **`BLOCK_M=8`** but raises **`BLOCK_N` to 512** when `spatial_size >= 2048`.
- **Evidence:** First headline **beat baseline** overall: **+0.7%** avg, **1.07×** geomean, **1.58×** total vs baseline (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** 2D backward input kernels need explicit **spatial tile widening** once row programs are sane.

**`opt-round-13` (parent `opt-round-10`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-13/attempts.md` + `opt-note.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-13` / `opt-round-10`.
- **Pre-change scenario:** Gated streaming path (r10) wins on huge spatial + high rows; inner **`BLOCK_N`** still left margin on very large spatial (`opt-note.md` round 13 theme).
- **Change:** Widen the **high-row large** streaming tile for very large `spatial_size` (attempts + operator note).
- **Evidence:** Correctness passed; **+8.4%** avg, **1.24×** geomean, **2.14×** total vs baseline; **promoted** over r10 (`opt-note.md`).
- **Interpretation:** Streaming inner loops still obey hierarchical tiling—widen inner `N` tile only where row batching already amortizes launch.

### `12_Permute`, `12_KvRmsnormRopeCache`, `13_InterleaveRope`, `14_Split`

Tiling-card narratives for these four in batch 2 are either **subsumed under PMR/UB on other cards** (`12_KvRmsnormRopeCache` co-tune appears on **`program-multiple-rows.md`**) or **layout-first** (`12_Permute`, `13_InterleaveRope`, much of `14_Split` on **`layout-store-and-block-pointers.md`** / **`autotune.md`**). Do not duplicate the same round here without a distinct **`BLOCK` / `SUB_BLOCK` / inner-axis ladder** story in `attempts.md`.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-1` (parent —)** — `15_AttentionSoftmaxWithSoftcappingAndDropout/opt-round-1/attempts.md`

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-1` / first round.
- **Pre-change scenario:** Single-level `BLOCK` for QK tiles plus softmax scratch exceeded UB on wide heads.
- **Change:** Introduced hierarchical sub-blocking along `K` for partial softmax accumulators while preserving outer head/seq tiling.
- **Evidence:** UB estimate in `attempts.md`; previously failing shapes now pass in `summary.md`.
- **Interpretation:** Attention is a multi-buffer UB pressure case; sub-block before pipeline overlap.

**`opt-round-16` (parent `opt-round-15`)** — `15_AttentionSoftmaxWithSoftcappingAndDropout/opt-round-16/attempts.md`

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-16` / `opt-round-15`.
- **Pre-change scenario:** After pipeline widening (r15), dropout + softmax temporaries still collided in UB on wide heads.
- **Change:** Sub-blocked dropout mask application so softmax state and mask buffers are not simultaneously peak-resident.
- **Evidence:** Peak live buffer chart in `attempts.md`; correctness + perf in `summary.md`.
- **Interpretation:** Epilogue fusion must respect hierarchical UB limits, not only QK tiles.

### `15_Pad`

**`opt-round-5` (parent `opt-round-4`)** — `15_Pad/opt-round-5/attempts.md`

- **Kernel / round / parent:** `15_Pad` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** After layout vectorization (r4), very large constant-fill still tried to hold an entire slice live in UB per program.
- **Change:** Added outer `BLOCK` with inner `SUB_BLOCK` fill loop to cap UB while preserving host grid simplicity.
- **Evidence:** `attempts.md` sub-block size rationale; `summary.md` multi-megabyte tensors.
- **Interpretation:** Pad kernels can overflow UB on huge contiguity; hierarchical tiling is the safe fix.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `16_Batched2DRopePositionEncodingBackward`

**`opt-round-1` (parent `baseline`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-1/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline fused sin/cos rope backward used **1024×1** programs with heavy scalar decode (`opt-note.md` / attempts).
- **Change:** Introduced **multi-row programs** (`BLOCK_M>1`) with contiguous row tiles and fused trig math.
- **Evidence:** Correctness passed; **+35.6%** avg, **1.55×** geomean vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Rope backward is tile-limited before math fusion—raise row batching before chasing trig micro-opts.

**`opt-round-17` (parent `opt-round-16`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-17/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-17` / `opt-round-16`.
- **Pre-change scenario:** After ladder tuning, **1024×1** still dominated worst-case launches (`opt-note.md` round 17 theme).
- **Change:** **Size-based dispatch**: small hidden uses **1024×1**; large hidden uses **512×2** with tuned `BLOCK_M` ladder.
- **Evidence:** Correctness passed; **+2.0%** avg vs baseline, **1.60×** geomean (`opt-note.md`); **promoted** as session best at that point.
- **Interpretation:** Dispatch by hidden width is tiling policy—one grid recipe rarely spans 1k-wide vs 4k-wide rope paths.

### `16_Repeat`

**`opt-round-9` (parent `opt-round-8`)** — `16_Repeat/opt-round-9/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Full-tile hot path still launched **one program per output row** on width-224 exact tiles (`opt-note.md` round 9 theme).
- **Change:** **Composed** full-tile loads with **`BLOCK_M=2`** row batching on the exact-tile branch.
- **Evidence:** Correctness passed; **+95.5%** avg, **28.05×** geomean vs baseline (`opt-note.md`); **promoted** session best.
- **Interpretation:** Narrow repeat widths need explicit row tiling even when loads are already vectorized.

### `17_EmbeddingWithInitialLayernormBackward`

**`opt-round-8` (parent `opt-round-7`)** — `17_EmbeddingWithInitialLayernormBackward/opt-round-8/attempts.md`

- **Kernel / round / parent:** `17_EmbeddingWithInitialLayernormBackward` / `opt-round-8` / `opt-round-7`.
- **Pre-change scenario:** 4096-wide backward still used **`BLOCK_M=1`** on the dominant fixed-width path (`opt-note.md` round 8 theme).
- **Change:** Raised **`BLOCK_M=2`** on the 4096 hidden fast path while keeping partial-sum correctness.
- **Evidence:** Correctness passed; **+6.0%** avg vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Layernorm backward is row-tile sensitive—validate `BLOCK_M` on the widest hidden before micro-math.

**`opt-round-20` (parent `opt-round-19`)** — `17_EmbeddingWithInitialLayernormBackward/opt-round-20/attempts.md`

- **Kernel / round / parent:** `17_EmbeddingWithInitialLayernormBackward` / `opt-round-20` / `opt-round-19`.
- **Pre-change scenario:** Largest-case partial-sum path still left occupancy on the table at **`BLOCK_M=2`** (`opt-note.md` round 20 theme).
- **Change:** **`BLOCK_M=3`** only on the largest partial-sum recipe; smaller shapes keep smaller `BLOCK_M`.
- **Evidence:** Correctness passed; **+8.0%** avg vs baseline (`opt-note.md`); **promoted** final best.
- **Interpretation:** Partial-sum paths need **shape-gated** `BLOCK_M`—do not broadcast the largest-case tile to all configs.

### `18_FusedAddRmsnorm`

**`opt-round-9` (parent `opt-round-8`)** — `18_FusedAddRmsnorm/opt-round-9/attempts.md`

- **Kernel / round / parent:** `18_FusedAddRmsnorm` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Large-row exact-4096 cases still used smaller inner **`BLOCK_SIZE`** than UB allowed (`opt-note.md` round 9 theme).
- **Change:** Raised inner **`BLOCK_SIZE` to 2048** on the large-row exact-4096 branch.
- **Evidence:** Correctness passed; **+6.5%** avg vs baseline (`opt-note.md`); **promoted** final best.
- **Interpretation:** RMS row reductions are inner-tile limited—widen hidden-axis tiles when UB headroom exists.

### `19_FusedResidualRmsNormBackward`

**`opt-round-10` (parent `opt-round-9`)** — `19_FusedResidualRmsNormBackward/opt-round-10/attempts.md`

- **Kernel / round / parent:** `19_FusedResidualRmsNormBackward` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** 4096-wide backward still used **`BLOCK_M=1`** when `rows >= 4096` (`opt-note.md` round 10 theme; cites **`program-multiple-rows.md`**).
- **Change:** **`BLOCK_M=2`** gated to **`hidden_size==4096` and `rows>=4096`**.
- **Evidence:** Correctness passed; **+4.5%** avg vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Residual backward matches embedding backward—row tiling wins on wide hidden + tall batches.

### `19_IndexPut`

**`opt-round-3` (parent `opt-round-2`)** — `19_IndexPut/opt-round-3/attempts.md`

- **Kernel / round / parent:** `19_IndexPut` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Inner accumulate loop still used **small fixed inner blocks** on wide value tensors (`opt-note.md` round 3 theme).
- **Change:** Increased inner **tile width** on accumulate path while keeping index uniqueness checks.
- **Evidence:** Correctness passed; **+3.5%** avg vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** IndexPut is dominated by inner hidden tiling—treat accumulate like a reduction epilogue.

### Other operators in this batch (`17_AdamW`, `18_Index`)

`17_AdamW` batch-2 story is **two-chunk programs** (`program-multiple-rows.md`) plus **LICM** (`loop-invariant-hoisting.md`). `18_Index` row batching (`opt-round-6`) **regressed**—see **`program-multiple-rows.md`** anti-signal, not tiling wins.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `23_HyenaFftSizePaddingRfft`

**`opt-round-7` (parent `opt-round-3`)** — theme in `23_HyenaFftSizePaddingRfft/opt-note.md` + `opt-round-7/attempts.md`

- **Kernel / round / parent:** `23_HyenaFftSizePaddingRfft` / `opt-round-7` / `opt-round-3`.
- **Pre-change scenario:** **1024+** wide row paths still launched too many thin row blocks on the dominant wide FFT regime (`opt-note.md`).
- **Change:** **Fewer row blocks** on the **1024+** column path while preserving contiguous padding semantics.
- **Evidence:** Correctness passed; session geomean lifted to **5.50×** vs baseline (`opt-note.md`); **promoted** over r3.
- **Interpretation:** Pad-to-FFT kernels are **column-tile / row-grid** coupled—narrowing row launch count can beat micro-hints once width saturates.

**`opt-round-9` (parent `opt-round-8`)** — `23_HyenaFftSizePaddingRfft/opt-round-9/attempts.md`

- **Kernel / round / parent:** `23_HyenaFftSizePaddingRfft` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** **`fft_size=2048`** still used a **512-wide** inner sweep that split awkwardly across real vs zero-padded halves (`attempts.md`).
- **Change:** **`BLOCK_ROWS=16`**, **`BLOCK_COLS=1024`** dispatch for **`fft_size >= 2048`** so each row uses one real-data tile and one zero tile.
- **Evidence:** Correctness passed; **Avg +66.8%**, **6.18×** geomean, **53.51×** total vs baseline (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** **Problem-shaped tiles** (half data / half pad) beat generic wider-but-more-sweeps when the pad boundary aligns to constexpr geometry.

**`opt-round-10` (parent `opt-round-9`)** — theme in `23_HyenaFftSizePaddingRfft/opt-note.md`

- **Kernel / round / parent:** `23_HyenaFftSizePaddingRfft` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** Larger **row batch** on the 2048 regime looked attractive for occupancy (`opt-note.md`).
- **Change:** Increased row batch on the 2048 path (session “larger row batch” experiment).
- **Evidence:** Correctness passed but **parent geomean 0.97×** vs r9 (`opt-note.md`); **not promoted**.
- **Interpretation:** After half-row-aligned tiles win, **extra row batching** can regress—validate parent deltas on the exact pad layout.

### `23_RepeatInterleave`

**`opt-round-5` (parent `opt-round-3`)** — theme in `23_RepeatInterleave/opt-note.md` + `opt-round-5/attempts.md`

- **Kernel / round / parent:** `23_RepeatInterleave` / `opt-round-5` / `opt-round-3`.
- **Pre-change scenario:** **`repeats=2`** row-tile path still under-tiled large contiguous slices (`opt-note.md`).
- **Change:** **Profiler-backed** larger row/column tiles on the **`repeats=2`** fast path only.
- **Evidence:** Correctness passed; large-case latency dropped sharply (`opt-note.md`); **promoted**.
- **Interpretation:** RepeatInterleave is **store-bandwidth** limited—retile from profile evidence, not guesswork.

**`opt-round-10` (parent `opt-round-9`)** — theme in `23_RepeatInterleave/opt-note.md` + `opt-round-10/attempts.md`

- **Kernel / round / parent:** `23_RepeatInterleave` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** Very-large **`repeats=2`** float32 slices still left top-end store width on the table after r9 (`opt-note.md`).
- **Change:** Narrower **`4 × 4096`** float32 top-end store regime on the dominant very-large branch.
- **Evidence:** Correctness passed; dominant large cases improved again (`opt-note.md`); **final best**.
- **Interpretation:** **Output store shape** is the last lever—treat outer dims as epilogue tile knobs once loads are saturated.

### `24_EmbeddingDenseBackward`

**`opt-round-6` (parent `opt-round-5`)** — `24_EmbeddingDenseBackward/opt-note.md` (round 6 theme)

- **Kernel / round / parent:** `24_EmbeddingDenseBackward` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** No-padding fast path still used narrower hidden-axis tiles on **divisible** large hidden sizes (`opt-note.md`).
- **Change:** **`512 × 1`** no-padding specialization when hidden size allows even divisibility.
- **Evidence:** Correctness passed; all five cases improved; **1.69×** geomean vs baseline (`opt-note.md`); **promoted** until r9 dispatch win.
- **Interpretation:** Embedding backward benefits from **constexpr hidden-width tiles** once padding masks are gone.

### `22_Nonzero`

**`opt-round-2` (parent `baseline`)** — `22_Nonzero/opt-round-2/attempts.md`

- **Kernel / round / parent:** `22_Nonzero` / `opt-round-2` / baseline.
- **Pre-change scenario:** Round-1 tile-prefix rewrite **regressed all cases** by paying sparse-style **`tl.cumsum`** on overwhelmingly **dense** tiles (`attempts.md`).
- **Change:** **Dense-tile fast path**: when **`tile_count == valid_count`**, write contiguous index ranges directly; keep local compaction only for mixed-density tiles.
- **Evidence:** Correctness passed; large wins on dense/near-dense cases (`opt-note.md`); **promoted** trunk for routing arc.
- **Interpretation:** Nonzero is a **density classifier** problem first—tiling policy must branch on measured tile occupancy.

### Other operators in this batch (`20_FusedRope*`, `20_Gather`, `21_Gaussian*`, `21_Scatter`, `22_HybridAttention*`, `24_KvCache*`)

`20_FusedRopeWithQkNormAndKvCacheUpdate` final structure combines **PMR** with **fixed-shape specialization**—see **`program-multiple-rows.md`** and **`compile_hint.md`**. `20_Gather` wins are **specialization + block pointers**, not inner-tile ladders. `21_GaussianTopkSparseActivation` **`opt-round-10`** composes smaller two-row mean tiles with existing kernels (`opt-note.md`)—cross-check **`program-multiple-rows.md`** and **`compile_hint.md`**. `21_Scatter` **`opt-round-8`** widens tiles on `outer_size==1` only (`opt-note.md`). `22_HybridAttentionMaskPreparation` tile story is **`autotune.md`**. `24_KvCacheUpdateWithRopeBackward` **`opt-round-5`** is PMR on **`program-multiple-rows.md`**.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators: **`25_MaskedSoftmaxWithAttentionDropoutBackward`**, **`25_NLLLoss`**, **`26_AvgPool3d`**, **`26_MoeGroupScoreAggregationAndMasking`**, **`27_MaxPool3d`**, **`27_MultiMaskAttentionAggregation`**, **`28_Interpolate`**, **`28_MultimodalRopePositionComputationWithGridBasedIndexing`**, **`29_DynamicQuant`**, **`29_TanhGatedResidualAddBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `25_MaskedSoftmaxWithAttentionDropoutBackward`

**`opt-round-7`–`opt-round-10` (parent chain `opt-round-6` → … → `opt-round-9`)** — `25_MaskedSoftmaxWithAttentionDropoutBackward/opt-note.md` + per-round `attempts.md` / `summary.md`

- **Kernel / round / parent:** `25_MaskedSoftmaxWithAttentionDropoutBackward` / **`opt-round-7`–`10`** / prior best after r6 host-path cleanup.
- **Pre-change scenario:** After **no-dropout Triton** (r5–r6) and **split dropout kernels**, largest **dropout-on** widths still needed wider **flat block** tiers on A5 (`opt-note.md` rounds 7–10 themes).
- **Change:** **Tiered `BLOCK`** thresholds for dropout kernel only (including extra large-width tier r8 and **4096-threshold** nudges r9–r10 per `attempts.md` headers).
- **Evidence:** Correctness passed; **final `opt-round-10`:** **Avg +45.1%**, **1.95×** geomean, **2.11×** total vs baseline (`opt-note.md`).
- **Interpretation:** Softmax-backward+dropout is **width-ladder tiling** once host/tensor-move overhead is gone—see also **`attention-cv-pipeline.md`**.

### `26_AvgPool3d`

**`opt-round-5` (parent `opt-round-4`)** — `26_AvgPool3d/opt-round-5/attempts.md`

- **Kernel / round / parent:** `26_AvgPool3d` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Profiler-backed diagnosis (partial `msprof` artifacts) still pointed at **in-kernel vector/mask** overhead on **`padding == (0,0,0)`** cases (`attempts.md`).
- **Change:** **`NO_PADDING_FASTPATH`** routing for **zero padding** and **`ceil_mode == False`** with **unmasked in-bounds** loads.
- **Evidence:** Correctness passed; **Avg +29.5%**, **1.57×** geomean vs baseline (`opt-note.md`); **promoted** trunk toward r7.
- **Interpretation:** 3D pool kernels need **padding-dispatch** before micro-arithmetic cleanup.

**`opt-round-7` (parent `opt-round-6`)** — theme in `26_AvgPool3d/opt-note.md` + `26_AvgPool3d/opt-round-7/attempts.md`

- **Kernel / round / parent:** `26_AvgPool3d` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** `out_w==1` micro-specialization and general-path cleanup plateaued; general kernel still carried redundant predicates (`opt-note.md`).
- **Change:** **Predicate cleanup** on general path after r6 narrow-tile branch.
- **Evidence:** Correctness passed; **Avg +31.0%**, **1.59×** geomean vs baseline (`opt-note.md`); **final best** after flat r8 / regressed r9–r10.
- **Interpretation:** End-game is **mask discipline**, not wider tiles—r9–r10 validated **not promoted** per `opt-note.md`.

### `27_MaxPool3d`

**`opt-round-6` (parent `opt-round-5`)** — `27_MaxPool3d/opt-round-6/attempts.md`

- **Kernel / round / parent:** `27_MaxPool3d` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** After **border/interior dispatch** (r5), case-5 hotspot moved to **`_max_pool3d_full_window_value_kernel`** with **high `aiv_mte2_ratio`**—**strided width loads** not scalar decode (`attempts.md` `perf-analysis.md`).
- **Change:** **Strip staging**: load **one contiguous width strip** per `(kd,kh)` plane and gather overlapping **`kw`** window locally.
- **Evidence:** Correctness passed; **Avg +64.7%**, **3.41×** geomean, **5.56×** total vs baseline; **1.56×** geomean vs r5 (`attempts.md`); **promoted**.
- **Interpretation:** Max-pool interior is a **memory-structure** problem—reuse staged strips before widening programs.

**`opt-round-9` (parent `opt-round-8`)** — theme in `27_MaxPool3d/opt-note.md` + `27_MaxPool3d/opt-round-9/attempts.md`

- **Kernel / round / parent:** `27_MaxPool3d` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Strip staging ported to padded fallback (r8); dominant interior still had tile headroom (`opt-note.md`).
- **Change:** **Widened large full-window tile** on the dominant interior regime.
- **Evidence:** Correctness passed; **Avg +75.6%**, **4.60×** geomean, **11.03×** total vs baseline (`opt-note.md`); **final best** (r10 blocked by NPU health per `opt-note.md`).
- **Interpretation:** Interior max-pool ends on **UB/tile width** trade—profiler-led tile growth after dispatch is stable.

### `28_Interpolate`

**`opt-round-4` (parent `opt-round-2`)** — theme in `28_Interpolate/opt-note.md` + `28_Interpolate/opt-round-4/attempts.md`

- **Kernel / round / parent:** `28_Interpolate` / `opt-round-4` / `opt-round-2`.
- **Pre-change scenario:** r3 launch-shape experiment **regressed** (`opt-note.md`); need **exact-scale** fast paths.
- **Change:** **Exact half-downsample bilinear** Triton fast path.
- **Evidence:** Correctness passed; **Avg +22.2%**, **1.79×** geomean vs baseline (`opt-note.md`); **promoted** before dedicated **`2×2`** kernel in r5.
- **Interpretation:** Interpolate wins from **constexpr scale factors**—match kernel template to harness-known ratios.

**`opt-round-5` (parent `opt-round-4`)** — theme in `28_Interpolate/opt-note.md`

- **Kernel / round / parent:** `28_Interpolate` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Dominant **`2×2`** downsample case still used generic bilinear (`opt-note.md`).
- **Change:** **Dedicated `2×2` downsample** Triton kernel.
- **Evidence:** Correctness passed; **Avg +22.8%**, **2.53×** geomean vs baseline (`opt-note.md`); **promoted**.

**`opt-round-9` (parent `opt-round-8`)** — `28_Interpolate/opt-round-9/attempts.md`

- **Kernel / round / parent:** `28_Interpolate` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** `msprof` on case-4 **`_bilinear2x_upsample2d_kernel`** showed huge **`Block Dim`** / launch fragmentation, not MTE gaps (`attempts.md`).
- **Change:** Grew exact **`2×` bilinear upsample** tile **`(16,32) → (32,64)`**.
- **Evidence:** Correctness passed; **Avg +58.7%**, **8.57×** geomean, **22.33×** total vs baseline; case-4 **−10.55%** vs r8 (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** Exact-scale upsample is **launch-granularity** limited—widen output tiles when profile shows vector-only hotspot.

### `28_MultimodalRopePositionComputationWithGridBasedIndexing`

**`opt-round-10` (parent `opt-round-9`)** — `28_MultimodalRopePositionComputationWithGridBasedIndexing/opt-round-10/attempts.md`

- **Kernel / round / parent:** `28_MultimodalRopePositionComputationWithGridBasedIndexing` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** **`hidden_size=4096`** still largest absolute workload after r6–r9 gather retile ladder (`attempts.md`).
- **Change:** Added **`BLOCK_SIZE=1024`** tier when **`hidden_size >= 2048`**, preserving smaller **`128/256/512`** tiers elsewhere.
- **Evidence:** Correctness passed; **Avg +32.6%**, **1.68×** geomean, **2.08×** total vs baseline (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** Multimodal rope+bilinear gather needs **explicit hidden-width tile ladder** on A5 after host-side overhead is trimmed.

### `29_DynamicQuant`

**`opt-round-4` (parent `opt-round-3`)** — theme in `29_DynamicQuant/opt-note.md` + `29_DynamicQuant/opt-round-4/attempts.md`

- **Kernel / round / parent:** `29_DynamicQuant` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Wide steady-state path still launched **one chunk per program** on the largest rows (`opt-note.md`).
- **Change:** **Two full chunks per wide steady-state program** before grid advance.
- **Evidence:** Correctness passed; **Avg +47.2%**, **1.90×** geomean, **1.83×** total vs baseline (`opt-note.md`); **promoted** toward r9.
- **Interpretation:** Quant row kernels benefit from **chunked flattening** along width—pair with **`grid-flatten-and-ub-buffering.md`** and **`program-multiple-rows.md`**.

### Other operators in this batch (`25_NLLLoss`, `26_Moe*`, `27_MultiMask*`, `29_TanhGated*`)

`25_NLLLoss` **`opt-round-2`** is primarily **grid remap** on **`program-multiple-rows.md`**. `26_MoeGroupScoreAggregationAndMasking` PMR story is **gated specialization** on **`program-multiple-rows.md`**. `27_MultiMaskAttentionAggregation` is **reduction fusion + LICM** (`loop-invariant-hoisting.md`). `29_TanhGatedResidualAddBackward` **`opt-round-4`** is **split launch / tail masks** (`grid-flatten-and-ub-buffering.md`, **`scalar-latency-traps.md`**).
