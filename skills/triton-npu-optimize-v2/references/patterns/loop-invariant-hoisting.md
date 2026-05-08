# Loop-Invariant Hoisting Pattern

## Summary

Apply **Loop-Invariant Code Motion (LICM)** to Triton kernels: move computations that do **not** depend on the loop induction variable out of the loop, so each iteration performs only the minimal work that truly varies.

On Ascend NPU, LICM most often reduces **AIV scalar/control** overhead (address generation, compares, index casts) and can indirectly reduce **CUBE starvation** (`WAIT_FLAG_DEVI`) by simplifying the loop body.

## Use When

- The kernel has a hot inner loop (often a K loop in GEMM-like kernels).
- Each loop iteration repeats substantial pointer math, mask construction, type casts, or shape bookkeeping.
- Profiling shows scalar/control work is disproportionately high relative to useful compute.

## Signals

### Code

- Inner loop recomputes expressions of the form:
  - `base(pid, offs) + delta(loop_var)`
  - e.g. `a_ptr + offs_m*stride_am + k*stride_ak`
- Masks are rebuilt each iteration even when parts are invariant:
  - e.g. `a_mask_m = offs_m < M` is invariant, but recomputed into `a_mask` each iter.

### IR

- Repeated arithmetic chains (`muli/addi/index_cast`) inside `scf.while` / `scf.for` bodies.
- Loop bodies contain repeated `subi/minsi/maxsi` patterns for bounds handling.

### Profile

- AIV scalar dominated by `LD_XD_XN_IMM`, `ST_XD_XN_IMM`, `ADD(_IMM)`, `CMP_IMM`.
- Timeline shows CUBE waiting on flags around the loop, while AIV performs control-heavy work.

## Optimization strategy

For any expression `E(loop_var)`:

1. Split it into **loop-invariant base** and **loop-varying delta**:
   - `E(loop_var) = BASE + DELTA(loop_var)`
2. Compute `BASE` once outside the loop.
3. Compute only `DELTA` inside the loop, and combine.

This pattern has several common specializations in Triton.

## Specialization A: Pointer address-generation hoisting (formerly “hoist-base-pointers”)

### Goal

Reduce per-iteration address-generation by hoisting invariant pointer bases.

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

## Specialization B: Mask / bounds hoisting (partial LICM)

### Goal

Hoist invariant parts of masks and bounds checks outside the loop.

### Example

- Invariant:
  - `a_mask_m = offs_m < M`
  - `b_mask_n = offs_n < N`
- Varying with `k_offs`:
  - `k_mask = k_offs < K`

Inside the loop build masks from precomputed invariants:

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

## Performance impact expectations

- Lower AIV scalar/control overhead, especially on large-K loops.
- Cleaner loop bodies can improve backend scheduling and reduce flag/wait overhead.
- LICM is typically a **low-risk, incremental** optimization: does not change math, only where expressions are computed.

## Pitfalls / risks

- **Broadcast orientation mistakes**: `[:, None]` vs `[None, :]` must be preserved.
- **Over-hoisting**: do not hoist expressions that depend on `k_offs` or other loop-varying values.
- LICM does not eliminate transform costs (e.g. ND2NZ) by itself; treat layout issues as separate patterns.

## What To Verify After Applying

1. **Correctness**: compare against reference across boundary shapes (non-multiples of block sizes).
2. **Profiler**: reduced scalar instruction mix (`LD/ST/ADD/CMP`) and improved wall time.
3. **IR sanity**: fewer repeated arithmetic ops inside loop bodies (qualitative evidence).

## Related Patterns

- Complements **`compile-hint`**: after LICM, add alignment/contiguity hints.
- Complements **`software-pipeline`**: LICM simplifies loop bodies; pipeline overlaps remaining transfer/compute.
- Complements **`remove-implicit-transpose`**: layout fixes reduce transform work; LICM reduces residual loop control cost.

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `11_DequantSwigluQuant`
- `12_KvRmsnormRopeCache`
- `15_Pad`
- `20_FusedRopeWithQkNormAndKvCacheUpdate`
- `20_Gather`
- `21_GaussianTopkSparseActivation`
- `27_MaxPool3d`
- `27_MultiMaskAttentionAggregation`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `11_DequantSwigluQuant`

**`opt-round-1` (parent `baseline`)** — `11_DequantSwigluQuant/opt-round-1/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline shows many wrapper ops (`Mul`, `Slice`, `Sigmoid`, `Abs`, `ReduceMax`, `RealDiv`, `Round`) repeating the same row/column pointer bases each step while a thin Triton multiply sits in the middle (`attempts.md` + `baseline/perf.txt` narrative).
- **Change:** Fused single-group dynamic fast path in Triton that keeps **loop-invariant row/column pointer bases** hoisted for the fused region instead of re-derived each wrapper op.
- **Evidence:** `run-test` + `compare-result` passed; `compare-perf` Geomean **1.07×**, Total **1.09×** vs baseline (`attempts.md`); `opt-note.md` marks round 1 as **validated branch** with the same headline outcome.
- **Interpretation:** LICM-style structuring is a prerequisite before PMR/tiling passes in later rounds can dominate.

### `2_GroupNormSwish`

**`opt-round-1` (parent `baseline`)** — `2_GroupNormSwish/opt-round-1/attempts.md`

- **Kernel / round / parent:** `2_GroupNormSwish` / `opt-round-1` / baseline.
- **Pre-change scenario:** Separate stats and apply passes repeat channel/spatial indexing and affine parameters across kernels (`attempts.md` fusion hypothesis).
- **Change:** Fuse **stats into apply** in one kernel; first attempt keeps Swish in kernel then **repairs numerics** by keeping Swish in wrapper while still fusing stats/affine when strict diff failed (`attempts.md` outcome narrative).
- **Evidence:** Correctness path documented in attempts; fusion direction validated as structural win subject to activation placement trade-offs (`opt-note.md` early rounds).
- **Interpretation:** LICM/fusion interacts with numerics—hoisting invariant stats is still this card even when Swish stays outside for correctness.

### Other kernels in the eight

Archived `attempts.md` headers for `10_SwigluQuant`, `10_LayerNorm`, `1_GELU`, `1_RotaryMul`, `2_SwiGLU`, and `11_GroupNorm` in this pilot either **do not cite** `loop-invariant-hoisting.md` as supporting evidence or list **other patterns first** (for example `program-multiple-rows.md`, `scalar-latency-traps.md`, `autotune.md`). Map those rounds on their dominant cited cards; do not duplicate them here without LICM-specific evidence.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `12_KvRmsnormRopeCache`

**`opt-round-5` (parent `opt-round-4`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** RMS scale factors and rope rotation constants were recomputed inside the inner loop over hidden lanes.
- **Change:** Hoisted RMS inverse-rms and cos/sin slices that depend only on `program_id` / constexpr head to tensor temps above the inner loop.
- **Evidence:** `attempts.md` duplicated-expr diff; `summary.md` vs r4.
- **Interpretation:** Norm+rope fusion duplicates many invariants until an explicit LICM pass.

### `15_Pad`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Pad value and stride products were rebuilt for every vector lane in the interior fill loop.
- **Change:** Hoisted `pad_value` broadcasts and stride linear combinations outside `tl.arange` bodies.
- **Evidence:** Scalar instruction drop in profiler notes; `summary.md` large tensor case.
- **Interpretation:** Even simple fills benefit from LICM when ranks > 1.

**`opt-round-10` (parent `opt-round-7`)** — `15_Pad/opt-round-10/attempts.md`

- **Kernel / round / parent:** `15_Pad` / `opt-round-10` / `opt-round-7`.
- **Pre-change scenario:** Wide-row **interior** copy still recomputed **row base pointers** inside the inner column loop after r7’s regime split (`opt-note.md` round 10 theme).
- **Change:** Hoisted row-base pointer arithmetic **above** the interior `BLOCK_COLS` loop so each row tile pays decode once.
- **Evidence:** Correctness passed; parent vs r7 **~+0.3%** avg—effectively flat headline (`opt-note.md`); **validated branch**.
- **Interpretation:** LICM micro-pass after major PMR/tiling wins—measure noise floor before declaring failure.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Causal / padding mask predicates were recomputed from sequence indices inside every softmax inner step.
- **Change:** Hoisted mask fragments that depend only on block row/col and host `seqlen` into prebuilt tensors before reduction.
- **Evidence:** `attempts.md` mask hoisting diagram; `summary.md` varlen case.
- **Interpretation:** Mask work is classic LICM fodder once dependencies are proven loop-invariant.

**`opt-round-17` (parent `opt-round-16`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-17` / `opt-round-16`.
- **Pre-change scenario:** After hierarchical epilogue tiling (r16), softcapping parameters (`softcap` value, scale combos) were still reloaded per inner softmax step even though constant per tile.
- **Change:** Moved softcap constants and fused scale factors above softmax reduction loop.
- **Evidence:** `attempts.md` cites constexpr promotion; `summary.md` softcap-on path.
- **Interpretation:** Matches attention-cv guidance: separate “vector epilogue constants” from data-dependent logits.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `17_AdamW`

**`opt-round-1` (parent `baseline`)** — `17_AdamW/opt-round-1/attempts.md`

- **Kernel / round / parent:** `17_AdamW` / `opt-round-1` / baseline.
- **Pre-change scenario:** Fused elementwise update still computed **`beta1_power` / `beta2_power` correction terms and paired per-element divisions** inside the hot vector body even though they are invariant across the launch (`attempts.md`).
- **Change:** **Precomputed invariant AdamW coefficients on the host** and algebraically simplified the kernel update so the device path drops redundant invariant scalar math.
- **Evidence:** Correctness passed; `compare-perf` **Avg +0.4%**, **1.00×** geomean vs baseline—small but session-positive (`attempts.md`); **promoted** over baseline.
- **Interpretation:** Optimizer kernels are LICM-shaped at the **launch-invariant** level first—host hoisting beats inner-loop micro-tuning when math is purely parametric.

### `18_FusedAddRmsnorm`

**`opt-round-1` (parent `baseline`)** — `18_FusedAddRmsnorm/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_FusedAddRmsnorm` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline split fused add+RMSNorm across **three kernels**, including a standalone **`_fused_add_rmsnorm_inv_rms_kernel`** row sweep that recomputed **`inv_rms`** from **`sq_sums`** then stored it globally (`attempts.md`).
- **Change:** **Removed** the dedicated inverse-RMS kernel and buffer; compute **`inv_rms = rsqrt(sq_sum / n_cols + eps)` inside the apply kernel** immediately after loading each row’s `sq_sums`.
- **Evidence:** Correctness passed; **Avg +36.6%**, **1.58×** geomean and total vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** LICM card overlaps **fusion** here—the invariant-per-row inverse pass should live next to its sole consumer, not in a separate global round-trip.

### Other operators in this batch (`16_*`, `16_Repeat`, `17_Embedding*`, `18_Index`, `19_*`)

`16_Batched2DRopePositionEncodingBackward` and `16_Repeat` batch-3 arcs are documented on **`program-multiple-rows.md`**, **`tiling.md`**, and **`layout-store-and-block-pointers.md`**. `17_EmbeddingWithInitialLayernormBackward` emphasizes **tiling / `BLOCK_M`** and **compile hints** more than inner-loop LICM in the cited attempts. `18_Index` and `19_IndexPut` emphasize **layout**, **hints**, and **scalar/IR** loops—map those rounds on those cards unless `attempts.md` foregrounds explicit hoists.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `21_GaussianTopkSparseActivation`

**`opt-round-1` (parent `baseline`)** — `21_GaussianTopkSparseActivation/opt-round-1/attempts.md`

- **Kernel / round / parent:** `21_GaussianTopkSparseActivation` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline ran **five kernels**, **three full passes** over **`x`**, used **atomics** on row reductions, and forced **`inputs.to(torch.float32)`** before Triton (`attempts.md`).
- **Change:** Collapsed to **two-kernel** stats+activation pipeline; **removed eager fp32 staging**; folded **threshold math into the sparse ReLU output kernel** while keeping numerically safe variance kernel (abandoned unsafe one-pass variance).
- **Evidence:** Correctness passed; **Avg +21.9%**, **1.29×** geomean vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Pipeline fusion + **eliminating redundant dtype staging** is LICM/hierarchy-shaped work before row batching (`program-multiple-rows.md`) and hints.

### Other operators in this batch (`20_FusedRope*`, `20_Gather`, `21_Scatter`, `22_*`, `23_*`, `24_*`)

`20_FusedRopeWithQkNormAndKvCacheUpdate` early rounds that **trimmed loads** regressed (`opt-note.md`)—not LICM wins. `20_Gather`, `21_Scatter`, `22_HybridAttentionMaskPreparation`, `22_Nonzero`, `23_*`, and `24_*` sessions emphasize **layout**, **tiling**, **autotune**, **routing**, or **dispatch** in their cited attempts.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`27_MultiMaskAttentionAggregation`** (other batch-5 kernels map to other cards). Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `27_MultiMaskAttentionAggregation`

**`opt-round-6` (parent `opt-round-5`)** — `27_MultiMaskAttentionAggregation/opt-round-6/attempts.md`

- **Kernel / round / parent:** `27_MultiMaskAttentionAggregation` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Fused **float32** mean kernel still rebuilt **row/class pointer bases inside the `K` loop** (`attempts.md`; cites **`loop-invariant-hoisting.md`**).
- **Change:** Hoisted **row-base pointers** and **`K`-local mask offsets**; repaired compile failure from illegal tensor-style pointer indexing by using **Triton-supported pointer arithmetic** only.
- **Evidence:** Correctness passed; **Avg +35.5%**, **1.80×** geomean vs baseline; **1.01×** geomean vs r5 with all cases slightly improved (`attempts.md`); **promoted** before ref_len specializations.
- **Interpretation:** Multi-mask class reductions are classic **K-loop LICM** once tile shape stabilizes (r5 row batching).

### `27_MaxPool3d`

**`opt-round-3` (parent `opt-round-2`)** — `27_MaxPool3d/opt-round-3/attempts.md`

- **Kernel / round / parent:** `27_MaxPool3d` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** No-index path still performed per-lane `valid_*` / `safe_*` checks and `tl.where` fixups even on common `padding=0, dilation=1` in-bounds windows.
- **Change:** Added `_max_pool3d_full_window_value_kernel` specialization for fully valid windows, seeding max from the first guaranteed-valid sample and removing inner-loop boundary bookkeeping from the hot path.
- **Evidence:** Correctness passed; `compare-perf` vs baseline in `attempts.md` reported **Avg +45.2%**, **Geomean 1.94x**, **Total 1.37x**, with strong wins on full-window cases; promoted as best.
- **Interpretation:** For pool kernels, hoisting/eliminating invariant boundary predicates is effectively LICM at the loop-structure level and can dominate over later micro-tiling.

### `20_FusedRopeWithQkNormAndKvCacheUpdate`

**`opt-round-1` (parent `baseline`)** — `20_FusedRopeWithQkNormAndKvCacheUpdate/opt-round-1/attempts.md`

- **Kernel / round / parent:** `20_FusedRopeWithQkNormAndKvCacheUpdate` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline path loaded full-row intermediates and carried redundant setup work in the rope/qk-norm fused flow.
- **Change:** First-round load/simplification pass targeting invariant work and redundant movement.
- **Evidence:** Correctness passed but performance regressed (`opt-note.md`); not promoted.
- **Interpretation:** Useful LICM anti-signal: trimming obvious invariants was insufficient until later PMR and fixed-shape specialization rounds.

### `20_Gather`

**`opt-round-10` (parent `opt-round-9`)** — `20_Gather/opt-round-10/attempts.md`

- **Kernel / round / parent:** `20_Gather` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** Guarded dispatch path was stable but still had residual flat generic overhead on inner-size-1 fallback regimes.
- **Change:** Added flat generic fast path for inner-size-1 gathers (a control/overhead simplification pass).
- **Evidence:** Correctness passed and improved round 9 slightly on targeted fallback cases, but did not beat round 5 overall (`opt-note.md`).
- **Interpretation:** Late-round gather control simplifications can help tails, but primary wins still came from earlier rank/dim specialization and layout cleanup.
