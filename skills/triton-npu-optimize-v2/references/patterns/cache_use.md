# Cache And UB Reuse Pattern

## Summary

Analyze memory access patterns, try to make use of cache and UB as much as possible. Make note of L2
cache (96MB, shared by all cores) and size of L1 and UB (512KB, 256KB, respectively).

## Use When

- The bottleneck looks memory-hierarchy bound rather than purely compute bound.
- Repeated reloads, weak reuse, or poor locality suggest that L2, L1, or UB usage can be improved through better data placement or tile sizing.

## Detail

Make use of information about sizes of caches and UB to optimize parameters by computation.
Take note that the UB size is 192KB (used for most operations), and the L1 cache for the Cube core
is 512KB (used for both input matrices of a matrix multiplication only).

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `1_RotaryMul`
- `11_DequantSwigluQuant`
- `11_GroupNorm`
- `16_Repeat`
- `21_GaussianTopkSparseActivation`
- `21_Scatter`
- `22_Nonzero`
- `25_MaskedSoftmaxWithAttentionDropoutBackward`
- `28_MultimodalRopePositionComputationWithGridBasedIndexing`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `1_RotaryMul`

**`opt-round-7` (parent `opt-round-5`)** — `1_RotaryMul/opt-round-7/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-7` / `opt-round-5`.
- **Pre-change scenario:** After r5 block-pointer layout, the interleave hotspot still paid repeated **broadcast coefficient** traffic (cos/sin / head tables) on the hottest branch (`opt-note.md` theme: specialize for head-broadcast coefficients).
- **Change:** Specialize the interleave hotspot so broadcasted coefficients are **reused** across inner work rather than re-fetched per micro-tile pattern implied by the generic path.
- **Evidence:** Mean **110.7256→102.3796µs** vs r5; correctness passed; **promoted** as session best per `opt-note.md`.
- **Interpretation:** Fits this card as **hierarchy locality**: fewer redundant vector loads of read-mostly tables once layout is stable.

**`opt-round-8` (parent `opt-round-7`)** — `1_RotaryMul/opt-round-8/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-8` / `opt-round-7`.
- **Pre-change scenario:** r7 minimized broadcast traffic; experiment tests **wider reuse** by sharing one coefficient tile across a head pair (`opt-note.md`).
- **Change:** Reuse one broadcast coefficient tile across a head pair on the specialized path.
- **Evidence:** Mean **102.3796→104.8354µs** vs r7; **not promoted** (`opt-note.md`).
- **Interpretation:** Aggressive reuse can increase live ranges or serialize lanes—**cache/L1 pressure** can worsen despite fewer logical loads.

### `11_DequantSwigluQuant`

**`opt-round-7` (parent `opt-round-6`)** — `11_DequantSwigluQuant/opt-round-7/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Profiler-led diagnosis in attempts highlights **large launch gaps** and wrapper-side vector traffic around `_swiglu_multiply_kernel`, i.e. fragmented memory/compute phases between fused and fallback paths.
- **Change:** Widen the fused dynamic fast path across column tiles and simplify kernel-side quantization so more work stays inside one fused region with fewer hand-offs (`opt-note.md` round 7 theme).
- **Evidence:** Correctness passed; Geomean **1.84×**, Total **2.07×** vs baseline; **promoted** per `attempts.md` Decision / `opt-note.md`.
- **Interpretation:** Keeps hot tensors **resident in fewer kernel phases**, improving effective reuse vs ping-ponging through thin kernels.

### Other kernels in the eight

No archived `attempts.md` in this pilot set cites **`cache_use.md`** as primary supporting evidence for `1_GELU`, `2_GroupNormSwish`, `2_SwiGLU`, `10_LayerNorm`, `10_SwigluQuant`, or `11_GroupNorm` in the same triage style as above. Their dominant evidence trails are narrated under **`autotune.md`**, **`tiling.md`**, **`program-multiple-rows.md`**, and **`layout-store-and-block-pointers.md`**. Do not invent five-field cache_use entries without log-backed cache hierarchy claims.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `12_KvRmsnormRopeCache`

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** KV cache segments were re-read from global memory for RMS and rope phases that could share the same resident tile.
- **Change:** Staged hot KV rows in UB across norm + rope application before optional write-back.
- **Evidence:** `attempts.md` live-tensor timeline; `summary.md` long-sequence case.
- **Interpretation:** Fusion exists to reuse data; measure reload elimination explicitly.

**`opt-round-5` (parent `opt-round-4`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Rope cos/sin tables were accessed with poor L2 locality when programs chased fine-grained head indices.
- **Change:** Reordered passes so each program walks contiguous table ranges once per task tile.
- **Evidence:** MTE2 ratio notes in round evidence; `summary.md` batch×head sweep.
- **Interpretation:** Table-driven kernels are hierarchy-bound even when compute is light.

### `13_Cat`

**`opt-round-17` (parent `opt-round-15`)** — `13_Cat/opt-round-17/attempts.md` + `opt-note.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-17` / `opt-round-15`.
- **Pre-change scenario:** Dim-0 flat concat path was already wide through **8192** elements; dominant cases remained **MTE/transfer** bound on the steady-state copy loop (`opt-note.md` ladder r12–r16).
- **Change:** Widened flat-path **`BLOCK_SIZE` to 16384** so each program moves more bytes per inner trip before launch overhead dominates.
- **Evidence:** Correctness passed; **promoted** each ladder step through r17 in `opt-note.md`; r18 **32768** overshoot **regressed** canonical mix (validated branch only).
- **Interpretation:** Wider vectorized copy improves **effective bandwidth** until instruction or schedule saturation—then switch to **launch flattening** (`grid-flatten-and-ub-buffering.md` r19 narrative).

### `13_InterleaveRope`

**`opt-round-4` (parent `opt-round-3`)**

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Staging buffers for interleaved indices were sized for correctness but evicted between paired halves unnecessarily.
- **Change:** Extended UB residency to cover both halves of the interleave pair before flushing to global outputs.
- **Evidence:** `attempts.md` footprint math; `summary.md` paired-index stress case.
- **Interpretation:** Paired-index patterns should reuse staging until consumers finish.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-4` (parent `opt-round-3`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Q and K tiles were reloaded across softmax statistics passes that could reuse the same tile in UB.
- **Change:** Kept Q/K fragments live through score max and partial softmax reduction where numerics allowed.
- **Evidence:** Reload count estimate in `attempts.md`; `summary.md` attention-length sweep.
- **Interpretation:** Attention is the archetypal “reuse across epilogue stages” workload.

**`opt-round-18` (parent `opt-round-17`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-18` / `opt-round-17`.
- **Pre-change scenario:** After LICM on softcap constants (r17), dropout mask loads still competed with softmax writes for L2 bandwidth on wide heads.
- **Change:** Batched mask consumption with softmax outputs in shared UB staging before scatter to global.
- **Evidence:** Profiler bandwidth buckets in round notes; `summary.md` dropout-on vs off comparison.
- **Interpretation:** Epilogue fusion must be planned as a single hierarchy story, not isolated ops.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `18_FusedAddRmsnorm`

**`opt-round-1` (parent `baseline`)** — `18_FusedAddRmsnorm/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_FusedAddRmsnorm` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline materialized **`sq_sums` and `inv_rms`** globally even though **`inv_rms` is consumed once** by the apply pass (`attempts.md`).
- **Change:** Dropped the standalone inverse-RMS kernel and **`inv_rms` tensor**, computing inverse RMS **inside the apply kernel** after loading `sq_sums`.
- **Evidence:** Correctness passed; resolved kernel set shrank; **Avg +36.6%**, **1.58×** geomean vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Removing a global intermediate is a **memory-hierarchy win**—fewer MTE round-trips—documented in parallel on **`loop-invariant-hoisting.md`** as fusion/LICM-shaped work.

### Cross-card note

Other batch-3 kernels in this pilot (**`16_Batched*`**, **`16_Repeat`**, **`17_AdamW`**, **`17_Embedding*`**, **`18_Index`**, **`19_IndexPut`**, **`19_FusedResidualRmsNormBackward`**) foreground **tiling**, **PMR**, **compile hints**, or **scalar loops** in their cited attempts; treat **`cache_use.md`** as secondary unless fresh profiler evidence names **reuse / eviction** across stages.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `21_GaussianTopkSparseActivation`

**`opt-round-1` (parent `baseline`)** — `21_GaussianTopkSparseActivation/opt-round-1/attempts.md`

- **Kernel / round / parent:** `21_GaussianTopkSparseActivation` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline performed **three full reads** of **`x`**, **atomics** on reductions, and **`float32` staging** before Triton (`attempts.md`).
- **Change:** Two-kernel pipeline with fused stats layout; **removed eager fp32 tensor copy** and dropped extra full-pass traffic while preserving safe variance kernel.
- **Evidence:** Correctness passed; **Avg +21.9%**, **1.29×** geomean vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Fewer global passes and no redundant staging are **hierarchy wins**—pair with **`loop-invariant-hoisting.md`** narrative.

### `21_Scatter`

**`opt-round-4` (parent `opt-round-3`)** — `21_Scatter/opt-round-4/attempts.md`

- **Kernel / round / parent:** `21_Scatter` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Contiguous fast path still did **`updated = x.contiguous().clone()`** then **`x.copy_(updated)`**, duplicating bytes around the hot kernel (`attempts.md`).
- **Change:** Use **`x` in-place** when already contiguous; keep temp buffer only for non-contiguous inputs.
- **Evidence:** Correctness passed; **Avg +37.2%**, **2.10×** geomean vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Wrapper-level **TensorMove** elimination is a real cache/MTE win—do not benchmark through extra full-tensor clones.

### `22_Nonzero`

**`opt-round-9` (parent `opt-round-8`)** — `22_Nonzero/opt-round-9/attempts.md`

- **Kernel / round / parent:** `22_Nonzero` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Small inputs still launched **`_count_nonzero_tiles_kernel`** even when the **8-tile probe** already covered **100%** of tiles (`attempts.md`).
- **Change:** Reused **probe counts** to choose fill/fallback/compact without a **second full count launch** when the probe window covers every tile.
- **Evidence:** Correctness passed; **Avg +65.4%**, **7.86×** geomean vs baseline; **1.38×** geomean vs r8 (`attempts.md`); **promoted** toward r10.
- **Interpretation:** **Launch reuse** is a hierarchy story—avoid duplicate global passes when host-side evidence is already complete.

### Cross-card note

Other batch-4 operators either do not foreground **reuse / fewer global phases** in cited attempts (**`20_Gather`**, **`22_HybridAttentionMaskPreparation`**, **`23_*`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**) or document wins primarily on **`program-multiple-rows.md`** / **`tiling.md`** (**`20_FusedRopeWithQkNormAndKvCacheUpdate`**).

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`25_MaskedSoftmaxWithAttentionDropoutBackward`**, **`28_MultimodalRopePositionComputationWithGridBasedIndexing`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `25_MaskedSoftmaxWithAttentionDropoutBackward`

**`opt-round-6` (parent `opt-round-5`)** — `25_MaskedSoftmaxWithAttentionDropoutBackward/opt-round-6/attempts.md` + `opt-note.md`

- **Kernel / round / parent:** `25_MaskedSoftmaxWithAttentionDropoutBackward` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** No-dropout Triton path (r5) still triggered **`BroadcastTo` + `Cast`** from **host-expanded `int8` mask** (`attempts.md`).
- **Change:** Load **compact attention mask** in-kernel; broadcast singleton head dim when valid; eliminate **`expand(...).contiguous().to(int8)`** on that branch.
- **Evidence:** Correctness passed; **no-dropout cases improved again** while dropout-on cases still gained (`opt-note.md`); **promoted** before large-width block tiers.
- **Interpretation:** Fewer **host tensor phases** around the kernel—reuse hierarchy story paired with **`attention-cv-pipeline.md`**.

### `28_MultimodalRopePositionComputationWithGridBasedIndexing`

**`opt-round-3` (parent `opt-round-2`)** — theme in `28_MultimodalRopePositionComputationWithGridBasedIndexing/opt-note.md` + `28_MultimodalRopePositionComputationWithGridBasedIndexing/opt-round-3/attempts.md`

- **Kernel / round / parent:** `28_MultimodalRopePositionComputationWithGridBasedIndexing` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** r2 moved **interpolation tables** off NPU; **rotary embedding setup** still hit device hot path (`opt-note.md`).
- **Change:** Move **RoPE coefficient setup** off the NPU hot path (host/precompute strategy per attempts).
- **Evidence:** Correctness passed; **slight improvement vs baseline** (`opt-note.md`); **promoted** toward gather retile rounds.
- **Interpretation:** **Table-driven** kernels should not pay per-launch **device setup** when coefficients are **read-mostly**—keep tables resident and cheap to reach before Triton gather tuning.

### Cross-card note

Other batch-5 operators (`25_NLLLoss`, `26_*`, `27_*`, `28_Interpolate`, `29_*`) emphasize **tiling**, **dispatch**, **PMR**, or **autotune** in their cited `opt-note.md` arcs rather than **extra global phases** alone.

## Gap-fill addendum (inventory alignment, 2026-05-08)

### `11_GroupNorm`

**`opt-round-13` (parent chain in `opt-note.md`)** — `11_GroupNorm/opt-round-13/attempts.md`

- **Kernel / round / parent:** `11_GroupNorm` / `opt-round-13` / parent per `opt-note.md`.
- **Pre-change scenario:** Prior rounds had already reduced scalar/control work; remaining hotspot behavior was dominated by movement/reuse on the normalization path.
- **Change:** Cache/UB-aware reuse adjustments on the hot fused path (per pattern citation in attempts).
- **Evidence:** Correctness passed; branch retained as a validated optimization step in the session arc (`opt-note.md` + attempts).
- **Interpretation:** GroupNorm keeps paying for hierarchy behavior after scalar cleanup; reuse-oriented tuning is a justified follow-up lever.

### `16_Repeat`

**`opt-round-6` (parent chain in `opt-note.md`)** — `16_Repeat/opt-round-6/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-6` / parent per `opt-note.md`.
- **Pre-change scenario:** Initial row/fulltile restructuring left dominant repeat paths sensitive to data movement and staging behavior.
- **Change:** Cache-use-oriented path tuning cited in attempts before later PMR/tiling ladder rounds.
- **Evidence:** Correctness passed and the branch contributed to the promoted trajectory later refined by rounds 9 and 15 (`opt-note.md`).
- **Interpretation:** Repeat is not only a launch-shape problem; staging and reuse choices remain visible in the measured path.
