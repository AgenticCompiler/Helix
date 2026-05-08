# Compiler Hint Pattern

## Summary

Try the following compile hints:

  - Before call to `tl.dot` in matrices, use `tl.compile_hint(a, "dot_pad_only_k")` for matrices
    involved in the product.
  - Use `tl.multiple_of` (resp. `tl.max_contiguous`) to specify tensor slices that are known to be
    aligned (resp. contiguous).

## Use When

- The kernel structure already looks close to good, but the compiler still lacks explicit alignment or contiguity information.
- `tl.dot` tiles, slices, or pointer math are known to satisfy stronger layout assumptions than the code currently expresses.

## Signals

### Code

- `tl.dot` inputs are already aligned in `M` and `N`, so only the `K` direction still needs padding hints.
- Pointer slices are known contiguous or aligned, but the code does not yet communicate that with `tl.max_contiguous` or `tl.multiple_of`.

## What To Verify After Applying

- Verify the alignment or contiguity assumptions encoded in the hint are actually true for the rewritten slices.
- Verify the compiler hints changed lowering or performance without changing the logical result.

## Detail

### dot_pad_only_k

Try using "dot_pad_only_k" to specify that in a Cube operation, only the `k` direction need
to be padded (the `m` and `n` directions are already aligned). For example: in the following
code for matmul:

```python
for k_start in range(0, K, BLOCK_K):
    mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
        k_start + tl.arange(0, BLOCK_K)
    )[None, :]
    mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
        (k_start + tl.arange(0, BLOCK_K)) < K
    )[None, :]
    mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
    tl.compile_hint(mat_a_block, "dot_pad_only_k")   # add compile hint
    mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
        n_start + tl.arange(0, BLOCK_N)
    )[None, :]
    mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
        (n_start + tl.arange(0, BLOCK_N)) < N
    )[None, :]
    mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
    tl.compile_hint(mat_b_block, "dot_pad_only_k")  #add compile hint
    mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
```

### max_contiguous and multiple_of

Set `tl.max_contiguous` to specify the loaded data is contiguous. Set `tl.multiple_of`
to specify the loaded data is aligned up to multiple of the second parameter. For example:

```python
@triton.jit
def write_req_to_token_pool_triton_optimize(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)
    extend_len = seq_len - pre_len

    cumsum_start = 0
    for i in range(pid_batch):
        cumsum_start += tl.load(extend_lens + i)

    token_start = pid_token * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    actual_offset = token_start + offset
    mask = actual_offset < extend_len

    src_ptr = out_cache_loc + cumsum_start + actual_offset
    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)  # used here
    value = tl.load(src_ptr, mask=mask)
    dst_ptr = (
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride
        + actual_offset
        + pre_len
    )
    dst_ptr = tl.max_contiguous(tl.multiple_of(dst_ptr, BLOCK_SIZE), BLOCK_SIZE)  # used here

    tl.store(dst_ptr, value, mask=mask)
```

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `1_RotaryMul`
- `10_SwigluQuant`
- `11_DequantSwigluQuant`
- `11_GroupNorm`
- `12_KvRmsnormRopeCache`
- `13_InterleaveRope`
- `15_Pad`
- `16_Batched2DRopePositionEncodingBackward`
- `17_AdamW`
- `17_EmbeddingWithInitialLayernormBackward`
- `18_Index`
- `19_FusedResidualRmsNormBackward`
- `19_IndexPut`
- `21_GaussianTopkSparseActivation`
- `22_HybridAttentionMaskPreparation`
- `23_HyenaFftSizePaddingRfft`
- `23_RepeatInterleave`
- `24_KvCacheUpdateWithRopeBackward`
- `20_FusedRopeWithQkNormAndKvCacheUpdate`
- `24_EmbeddingDenseBackward`
- `27_MultiMaskAttentionAggregation`
- `29_TanhGatedResidualAddBackward`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `1_GELU`

**`opt-round-4` (parent `opt-round-3`)** — `1_GELU/opt-round-4/attempts.md`

- **Kernel / round / parent:** `1_GELU` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Profiling highlights case 5 on the masked elementwise path; compiler lacks proof that interior blocks are fully aligned/contiguous (`profile-summary.json` / `perf-analysis.md` cited in attempts).
- **Change:** Add **full-block fast path** with unmasked `tl.load` / `tl.store` plus `tl.max_contiguous(tl.multiple_of(...))` on hot offsets; keep masked tail for partial blocks.
- **Evidence:** Geomean slightly **below** r3; **validated branch**, not promoted (`opt-note.md` round 4 theme); correctness passed.
- **Interpretation:** Compile hints and fast paths only win when alignment preconditions truly hold on the benchmarked mix—measure against parent, not only baseline.

### `10_SwigluQuant`

**`opt-round-10` (parent `opt-round-8`)** — `10_SwigluQuant/opt-round-10/attempts.md`

- **Kernel / round / parent:** `10_SwigluQuant` / `opt-round-10` / `opt-round-8`.
- **Pre-change scenario:** Fused kernels stable after r8 trunk; compiler still lacks explicit contiguity/alignment facts on hot slices (`opt-note.md` round 10 theme).
- **Change:** Add **compiler contiguity hints** (`tl.max_contiguous` / `tl.multiple_of` style annotations per attempts) on the stable fused kernels without changing arithmetic.
- **Evidence:** Geomean **1.91×**, Total **2.25×** vs baseline; **promoted** best in that arc (`opt-note.md`); rounds **11–12** later ablate hints and regress (~**0.98–0.99×** geomean vs r10), confirming sensitivity.
- **Interpretation:** Hints are a sharp tool—keep a parent-only regression gate after each hint batch.

### `11_DequantSwigluQuant`

**`opt-round-10` (parent `opt-round-8`)** — `11_DequantSwigluQuant/opt-round-10/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-10` / `opt-round-8`.
- **Pre-change scenario:** Same structural situation as `10_SwigluQuant`: fused multiply/quantize paths stable but slices are stronger than the type system shows (`opt-note.md`).
- **Change:** Apply contiguity / alignment hints on fused kernels analogous to the SwiGLU-quant session’s r10 pass.
- **Evidence:** Geomean **1.91×**, Total **2.25×** vs baseline; **promoted** over r8 per `opt-note.md` before later tail specialization rounds.
- **Interpretation:** Dequant+SwiGLU+quant fusion benefits from the same compile-hint discipline as standalone quant operators once layout is fixed.

### Other kernels in the eight

`1_RotaryMul` **round 9** adds narrow alignment hints on the broadcast-head path and **regresses** mean vs r7 (`opt-note.md`)—documented anti-signal, not repeated here as a success. `2_SwiGLU`, `2_GroupNormSwish`, `10_LayerNorm`, and `11_GroupNorm` pilot-span work is dominated by **layout**, **PMR**, and **autotune** citations instead of dedicated compile-hint rounds; map those attempts on their primary cards.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Same five-field template as prior pilot.*

### `12_KvRmsnormRopeCache`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** RMSNorm + RoPE fused loads used pointer slices the compiler treated as weakly contiguous along the head axis.
- **Change:** Applied `tl.max_contiguous` / `tl.multiple_of` on KV and rope table slices with host-proven alignment.
- **Evidence:** `attempts.md` alignment table vs JSON strides; `summary.md` micro-benchmark on long cache entries.
- **Interpretation:** Cache-fused kernels need explicit contiguity on table-driven loads.

**`opt-round-4` (parent `opt-round-3`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Any `tl.dot`-like reduction on partial RMS stats still lacked `dot_pad_only_k` style hints where only `K` needed padding.
- **Change:** Added `tl.compile_hint` for dot operands after verifying `M`/`N` alignment already held.
- **Evidence:** IR commentary in `attempts.md`; stable perf on padded-head cases in `summary.md`.
- **Interpretation:** Matches this card’s “structure first, then hints” ordering.

### `13_InterleaveRope`

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Interleaved rope indices produced strided gather windows the compiler could not prove contiguous.
- **Change:** Strengthened `multiple_of` on index ranges derived from host `head_dim` and rope period constexprs.
- **Evidence:** `attempts.md` documents constexpr preconditions; correctness on odd `head_dim` tails.
- **Interpretation:** Rope interleave kernels are hint-sensitive once gather staging lands.

### `14_Split`

**`opt-round-1` (parent —)**

- **Kernel / round / parent:** `14_Split` / `opt-round-1` / first round.
- **Pre-change scenario:** Split output segments were logically contiguous but sliced with generic offsets; lowering emitted conservative MTE.
- **Change:** Marked split output `tl.load`/`tl.store` windows with `max_contiguous` along the fast split axis.
- **Evidence:** `summary.md` first-pass win; `attempts.md` slice shapes.
- **Interpretation:** Split is mostly a layout/transfer problem; hints are low-risk early.

### `15_Pad`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Large contiguous fill regions still used generic pointer increments; compiler did not know fill width was vector-aligned.
- **Change:** `tl.multiple_of` on `tl.arange` spans for benchmark-aligned pad widths; documented unsupported odd widths.
- **Evidence:** Vector-width checklist in `attempts.md`; `summary.md` aligned vs misaligned case split.
- **Interpretation:** Pad kernels benefit from the same contiguity hints as copy-like ops.

**`opt-round-9` (parent `opt-round-7`)** — `15_Pad/opt-round-9/attempts.md`

- **Kernel / round / parent:** `15_Pad` / `opt-round-9` / `opt-round-7`.
- **Pre-change scenario:** Very-wide-row **interior** copies were contiguous in memory but the compiler still emitted conservative MTE on the hot `tl.load`/`tl.store` spans (`opt-note.md` round 9 theme).
- **Change:** Added **`tl.max_contiguous` / `tl.multiple_of`** hints on the interior copy `tl.arange` windows for the benchmark-aligned 512-column wide path.
- **Evidence:** Correctness passed; parent vs r7 effectively **flat** (`Avg ~-0.3%` per `opt-note.md`); **validated branch**.
- **Interpretation:** Hint-only passes need parent deltas—flat is useful negative evidence on this card.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-11` (parent `opt-round-10`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-11` / `opt-round-10`.
- **Pre-change scenario:** Score `tl.dot` tiles were aligned on host for `M`/`N` but `K` still carried implicit padding cost after the `exp` / state convention pass (r10).
- **Change:** `tl.compile_hint(score, "dot_pad_only_k")` on QK matmul path after verifying head divisibility.
- **Evidence:** `attempts.md` IR note; `summary.md` wide-head case.
- **Interpretation:** Attention QK is a primary `dot_pad_only_k` consumer once masks respect the same bounds.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `16_Batched2DRopePositionEncodingBackward`

**`opt-round-19` (parent `opt-round-18`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-19/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-19` / `opt-round-18`.
- **Pre-change scenario:** Profiler-led **full-tile specialization** from r18 was already fragile; steady-state still looked movement-heavy on huge regimes (`attempts.md`).
- **Change:** Added **`tl.multiple_of`** plus **`tl.max_contiguous`** on steady-state offsets **on top of** the r18 full-tile kernel.
- **Evidence:** Correctness passed; headline **Avg +56.2%**, **2.97×** geomean vs baseline—but **regressed vs r17 and r18** on parent comparison (`attempts.md`); **not promoted**.
- **Interpretation:** Stacking contiguity hints on an already-specialized fast path can **hurt**—validate parent deltas, not only baseline deltas.

**`opt-round-20` (parent `opt-round-17`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-20/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-20` / `opt-round-17`.
- **Pre-change scenario:** After r19, the open question was whether hints help the **best base kernel (r17)** without the r18 branch (`attempts.md`).
- **Change:** Applied the same **`tl.multiple_of` / `tl.max_contiguous`** pattern to **r17’s** masked load/store offsets before the inner loop.
- **Evidence:** Correctness passed; **Avg +57.4%**, **3.05×** geomean vs baseline—**tied** best rounded geomean but **did not beat r17 on total speedup** (`attempts.md`); **not promoted**.
- **Interpretation:** Hint-only passes need **clear dominance vs parent**; ties against a simpler kernel are negative evidence for promotion.

### `17_EmbeddingWithInitialLayernormBackward`

**`opt-round-6` (parent `opt-round-5`)** — `17_EmbeddingWithInitialLayernormBackward/opt-round-6/attempts.md`

- **Kernel / round / parent:** `17_EmbeddingWithInitialLayernormBackward` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Dispatch thresholds plateaued; the **4096-wide fixed path** still used flattened pointer expressions only (`attempts.md`).
- **Change:** Added **`tl.max_contiguous`** and safe row-base **multiple-of** annotations on **fixed-path** loads/stores only; arithmetic unchanged.
- **Evidence:** Correctness passed; **Avg regression narrowed to ~1.7%** vs baseline—best non-baseline average so far but **large fixed-path case still regressed** (`attempts.md`); **validated branch**, not promoted over baseline.
- **Interpretation:** Movement-leaning kernels still need **per-case proof** that hints help the dominant shape, not only the average.

### `18_Index`

**`opt-round-4` (parent `opt-round-3`)** — `18_Index/opt-round-4/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Row-copy kernels already used contiguous **`inner_size`** blocks; compiler still lowered conservatively (`attempts.md`).
- **Change:** Applied **`tl.max_contiguous`** to contiguous source and destination pointer ranges in **both** row-copy kernels.
- **Evidence:** Correctness passed; **Avg +73.5%**, **11.11×** geomean vs baseline—but **case-1 regressed** enough that **geomean fell below r3** (`attempts.md`); **not promoted**.
- **Interpretation:** Hints can trade small-shape latency for large-slice wins—headline geomean is the gate, not baseline-only uplift.

### `19_IndexPut`

**`opt-round-8` (parent `opt-round-7`)** — `19_IndexPut/opt-round-8/attempts.md`

- **Kernel / round / parent:** `19_IndexPut` / `opt-round-8` / `opt-round-7`.
- **Pre-change scenario:** IR still showed **scratch-buffer copies** of contiguous **`index` / `values`** slices ahead of a scalarized update loop (`attempts.md`).
- **Change:** Wrapped contiguous load addresses with **`tl.multiple_of(..., 16)`** plus **`tl.max_contiguous(..., BLOCK_SIZE)`** while preserving r7 block thresholds.
- **Evidence:** Correctness passed; **Avg +23.0%**, **1.33×** geomean vs baseline with **modest parent win** over r7 (`attempts.md`); **promoted** session best.
- **Interpretation:** When IR proves the compiler missed contiguity on inputs, hints are a **low-risk epilogue** after block policy is frozen.

### Other operators in this batch (`17_AdamW`, `18_FusedAddRmsnorm`, `19_FusedResidualRmsNormBackward`)

`17_AdamW` **r2** documents profile-backed **contiguity / multiple-of** annotations on the 1D offsets (`17_AdamW/opt-round-2/attempts.md`)—mostly **flat vs r1**, validated branch. `18_FusedAddRmsnorm` and `19_FusedResidualRmsNormBackward` batch-3 emphasis is **tiling / PMR** on other cards.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `20_FusedRopeWithQkNormAndKvCacheUpdate`

**`opt-round-10` (parent `opt-round-9`)** — `20_FusedRopeWithQkNormAndKvCacheUpdate/opt-round-10/attempts.md`

- **Kernel / round / parent:** `20_FusedRopeWithQkNormAndKvCacheUpdate` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** Fixed-shape **64-dim / 128-dim** kernels already encoded contiguous half-row geometry (`attempts.md`).
- **Change:** Added **`tl.multiple_of`** and **`tl.max_contiguous`** to fixed-shape **`full_offsets` / `half_offsets`** vectors inside those kernels only.
- **Evidence:** Correctness passed; **Avg +65.4%**, **3.49×** geomean vs baseline; **1.01×** geomean vs r9 (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** End-of-session **hint cleanup** on proven-specialized paths—small but strictly positive parent deltas.

### `21_GaussianTopkSparseActivation`

**`opt-round-7` (parent `opt-round-5`)** — `21_GaussianTopkSparseActivation/opt-round-7/attempts.md`

- **Kernel / round / parent:** `21_GaussianTopkSparseActivation` / `opt-round-7` / `opt-round-5`.
- **Pre-change scenario:** Post-r6 profile showed remaining hot kernels **memory-heavy** on contiguous row-block slices (`attempts.md`).
- **Change:** **`tl.multiple_of`** + **`tl.max_contiguous`** on **`col_offsets`** slices in **`_gaussian_row_sum_kernel`**, **`_gaussian_row_var_sum_kernel`**, and **`_gaussian_sparse_relu_kernel`**.
- **Evidence:** Correctness passed; **Avg +36.5%**, **1.59×** geomean vs baseline (`attempts.md`); **promoted** over r5.
- **Interpretation:** Multi-kernel stats pipelines still benefit from **explicit contiguous column facts** once row batching stabilizes.

### `23_HyenaFftSizePaddingRfft`

**`opt-round-3` (parent `opt-round-2`)** — theme in `23_HyenaFftSizePaddingRfft/opt-note.md` + `23_HyenaFftSizePaddingRfft/opt-round-3/attempts.md`

- **Kernel / round / parent:** `23_HyenaFftSizePaddingRfft` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Row-tiled contiguous padding kernel from r1–r2 still exposed conservative lowering on steady-state offsets (`opt-note.md`).
- **Change:** **`tl.multiple_of` / `tl.max_contiguous`** on contiguous **row and column** offset vectors (`opt-note.md` round 3 theme).
- **Evidence:** Correctness passed; **Avg +66.8%**, **5.13×** geomean vs baseline (`opt-note.md`); **promoted** before width dispatch ladder.
- **Interpretation:** Pad kernels are memcpy-shaped—hints are a natural companion to **tiling** wins.

### `23_RepeatInterleave`

**`opt-round-4` (parent `opt-round-3`)** — theme in `23_RepeatInterleave/opt-note.md` + `23_RepeatInterleave/opt-round-4/attempts.md`

- **Kernel / round / parent:** `23_RepeatInterleave` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Row-tile path for **`repeats=2`** was stable after r3 (`opt-note.md`).
- **Change:** **Compile hints** on row-tile **column slices** (contiguity/alignment guidance per attempts).
- **Evidence:** Correctness passed; stayed **near r3** without clear promotion (`opt-note.md`); **validated branch**.
- **Interpretation:** Hint-only passes after big layout wins often **plateau**—keep as branch unless parent beats r3.

### `24_KvCacheUpdateWithRopeBackward`

**`opt-round-4` (parent `opt-round-3`)** — theme in `24_KvCacheUpdateWithRopeBackward/opt-note.md` + `24_KvCacheUpdateWithRopeBackward/opt-round-4/attempts.md`

- **Kernel / round / parent:** `24_KvCacheUpdateWithRopeBackward` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Unit-stride inner-dim path from r3 still left compiler without explicit contiguous span facts (`opt-note.md` round 4 theme).
- **Change:** Added **contiguity hints** on the inner-dimension loads/stores alongside r3’s 2D grid.
- **Evidence:** Correctness passed; **roughly tied r3**, slight overall regression (`opt-note.md`); **not promoted**.
- **Interpretation:** Negative / flat hint evidence—**2D grid + PMR** carried the real win, not this pass.

### `22_HybridAttentionMaskPreparation`

**`opt-round-9` (parent `opt-round-6`)** — `22_HybridAttentionMaskPreparation/opt-round-9/attempts.md`

- **Kernel / round / parent:** `22_HybridAttentionMaskPreparation` / `opt-round-9` / `opt-round-6`.
- **Pre-change scenario:** Round-6 kernel already had winning autotuned tiles and **block-pointer** stores (`attempts.md`).
- **Change:** **`tl.multiple_of` / `tl.max_contiguous`** on **`offs_m` / `offs_n`** before predicates and block-pointer store.
- **Evidence:** Correctness passed but **severe regression** vs r6 on headline and per-case mix (`opt-note.md` round 9 theme; `attempts.md` shows large parent drop); **not promoted**.
- **Interpretation:** **Anti-signal**: hints on mask index vectors can fight the compiler once autotune + block pointers already fit the schedule.

### Other operators in this batch (`20_Gather`, `21_Scatter`, `22_Nonzero`, `24_EmbeddingDenseBackward`)

`20_Gather` **`opt-round-4`** max_contiguous experiment **hurt** case-1 vs r3 (`20_Gather/opt-round-4/attempts.md`—see batch-4 **`layout-store-and-block-pointers.md`** / **`scalar-latency-traps.md`**). `21_Scatter` session best is **layout + int32 host path** (`layout-store`, **`scalar-latency-traps.md`**). `22_Nonzero` is **routing**-first. `24_EmbeddingDenseBackward` **`opt-round-4`** is **constexpr / no-padding fast path**—structural dispatch more than hints (`tiling.md`, **`loop-invariant-hoisting.md`** style constexpr gates).

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`27_MultiMaskAttentionAggregation`**, **`29_TanhGatedResidualAddBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `27_MultiMaskAttentionAggregation`

**`opt-round-7` (parent `opt-round-6`)** — `27_MultiMaskAttentionAggregation/opt-round-7/attempts.md`

- **Kernel / round / parent:** `27_MultiMaskAttentionAggregation` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Fused float32 kernel had **contiguous `ref_len` slices** but compiler lacked explicit facts (`attempts.md`; cites **`compile_hint.md`**).
- **Change:** Added **`tl.max_contiguous`** on **`K` slices**; **removed** invalid **`tl.multiple_of`** claim when indices are not multiples of **`BLOCK_K`** (repair in attempts).
- **Evidence:** Correctness passed; **Avg +34.6%**, **1.77×** geomean vs baseline but **0.99×** geomean vs r6—regressed cases **1–2** (`attempts.md`); **validated branch**, not promoted.
- **Interpretation:** **Anti-signal**: `multiple_of` must match **true alignment**; partial hints can still lose on tiny cases.

### `29_TanhGatedResidualAddBackward`

**`opt-round-7` (parent `opt-round-4`)** — `29_TanhGatedResidualAddBackward/opt-round-7/attempts.md`

- **Kernel / round / parent:** `29_TanhGatedResidualAddBackward` / `opt-round-7` / `opt-round-4`.
- **Pre-change scenario:** r4 **split-launch** prefix/tail structure was best; hypothesis tested **`tl.multiple_of` + `tl.max_contiguous`** on **unmasked prefix** only (`attempts.md`).
- **Change:** Hint-only pass on both streaming kernels’ prefix path.
- **Evidence:** Correctness passed; **Avg +11.9%**, **1.15×** geomean vs baseline but **0.99×** geomean vs r4 (`attempts.md`); **not promoted**.
- **Interpretation:** After **mask split** wins, hint-only passes are **noise-sensitive**—keep r4 structure as anchor.

### Other operators in this batch (`25_*`, `26_*`, `28_*`, `29_DynamicQuant`)

`25_MaskedSoftmaxWithAttentionDropoutBackward` **`opt-round-4`** contiguity hints **slipped average below baseline** (`opt-note.md`). `26_AvgPool3d` / `28_Interpolate` / `29_DynamicQuant` emphasize **tiling**, **profiler dispatch**, and **`autotune.md`** more than compile hints in their cited `opt-note.md` arcs.
