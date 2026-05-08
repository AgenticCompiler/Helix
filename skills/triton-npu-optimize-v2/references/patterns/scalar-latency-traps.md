# Scalar Latency Trap Removal Pattern

## Summary

Remove scalarizing constructs that make an otherwise vector-friendly Ascend Triton kernel spend time on avoidable scalar control, address arithmetic, or long dependency chains.

Use this as a trap-elimination pattern before larger rewrites. Apply one repair per round, then validate correctness and benchmark evidence through the normal optimize flow.

## Use When

- Runtime values that are shape constants are passed as normal arguments instead of `tl.constexpr`.
- Pointer variables are updated with `+=` inside a loop, creating loop-carried address dependencies.
- Address expressions use modulo addressing (`%`) to wrap tail tiles or index boundaries.
- `tl.where` masks all lanes except a single special position, or has exactly one false lane in a vector.
- Integer elementwise arithmetic is done as scalar-looking `int64` work even though the value range is safely `int32`.
- `tl.cumsum` runs on a long one-dimensional vector and profiling or IR suggests scalar degradation.

## Signals

### Code

- Runtime values that are shape constants are passed as normal arguments instead of `tl.constexpr`.
- Pointer variables are updated with `+=` inside a loop, creating loop-carried address dependencies.
- Address expressions use modulo addressing (`%`) to wrap tail tiles or index boundaries.
- `tl.where` masks all lanes except a single special position, or has exactly one false lane in a vector.
- Integer elementwise arithmetic is done as scalar-looking `int64` work even though the value range is safely `int32`.
- `tl.cumsum` runs on a long one-dimensional vector and profiling or IR suggests scalar degradation.

## Repairs

### Static parameters

Make compile-time constants explicit:

```python
@triton.jit
def kernel(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
```

Prefer `tl.constexpr` for fixed sizes, strides, booleans, mode flags, and architecture-selected knobs. Do not make data-dependent runtime values constexpr.

### Loop pointer recurrences

Avoid pointer updates that depend on the previous iteration:

```python
# Prefer this shape.
for i in tl.range(0, K, BLOCK_K):
    ptrs = base + (i + offs_k) * stride_k + offs_n
    vals = tl.load(ptrs, mask=(i + offs_k) < K)
```

This keeps each iteration's address computation derived from a stable base plus an explicit offset. It is especially useful when loop trip count is large enough for scalar scheduling to matter.

### Modulo removal

Avoid `%` for tail handling when a mask can preserve continuous addresses:

```python
offs = block_start + tl.arange(0, BLOCK)
mask = offs < N
vals = tl.load(x + offs, mask=mask, other=0.0)
```

Use modulo only when wraparound is part of the mathematical semantics, not just a boundary workaround.

### Single-position `tl.where`

When exactly one lane differs, consider replacing a whole-vector `tl.where` with a targeted extract/insert style repair. Only apply this when the one-position condition is proven by shape or index construction. If more than one lane can differ, keep the original vector conditional.

### Int32 vector arithmetic

If index or offset arithmetic is proven to stay within `[-2**31, 2**31 - 1]`, cast once near load or construction and keep the hot vector math in `int32`. Cast back only when the API or pointer expression truly requires it.

Do not use this for values that can overflow `int32`.

### Cumsum axis splitting

For a long one-dimensional `tl.cumsum`, consider reshaping to a two-dimensional tile so cumsum runs on shorter axes, then combine block-local prefix totals. Tune the split size because both axes trade off against each other and can affect UB pressure.

## Risks

- `tl.constexpr` changes specialization behavior and compile-cache cardinality.
- Removing `%` is only safe when masks preserve the original boundary semantics.
- Int32 conversion is a semantic promise about value range.
- Cumsum decomposition must preserve prefix order exactly.

## What To Verify After Applying

- Record the trap and exact code location in `attempts.md`.
- Run correctness before trusting performance.
- Use the project benchmark and `compare-perf` authority for any claimed speedup.
- If the repair changes specialization keys or host call signatures, verify all call sites.

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `1_GELU`
- `1_RotaryMul`
- `10_LayerNorm`
- `12_KvRmsnormRopeCache`
- `12_Permute`
- `13_InterleaveRope`
- `16_Repeat`
- `22_Nonzero`
- `20_Gather`
- `21_Scatter`
- `24_EmbeddingDenseBackward`
- `25_NLLLoss`
- `26_MoeGroupScoreAggregationAndMasking`
- `29_TanhGatedResidualAddBackward`
- `27_MultiMaskAttentionAggregation`
- `28_Interpolate`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `1_GELU`

**`opt-round-1` (parent `baseline`)** — `1_GELU/opt-round-1/attempts.md`

- **Kernel / round / parent:** `1_GELU` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline launch sizing uses dynamic escape logic without disciplined constexpr/static shape story on the hot path (`attempts.md` lists `scalar-latency-traps.md` with `autotune.md` as joint supporting evidence).
- **Change:** Hold **static shape / launch discipline** (fixed launch tier starting point) before larger math rewrites—paired in-session with launch-size selection narrative.
- **Evidence:** Correctness + bench pass; `compare-perf` Geomean **1.02×**, Total **1.01×** vs baseline (`opt-note.md` round 1); **promoted** as first best.
- **Interpretation:** Scalar/latency discipline here is about **avoiding per-element dynamic policy** on the fastest path; deeper launch tiers continue on **`autotune.md`**.

### `1_RotaryMul`

**`opt-round-1` (parent `baseline`)** — `1_RotaryMul/opt-round-1/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-1` / baseline.
- **Pre-change scenario:** Flat `total_pairs` grid with **per-lane** `row = offsets // half_dim`, `pair = offsets % half_dim` decode—classic integer div/mod in the hot indexing path (`attempts.md` citing `scalar-latency-traps.md`).
- **Change:** Replace flat decode with row-blocked mapping (see also **`program-multiple-rows.md`** for the structural batching story).
- **Evidence:** Mean **21265.87→196.117µs** vs baseline in `opt-note.md`; **promoted**.
- **Interpretation:** Removing per-element decode is the scalar-trap win; row batching is how the win is expressed in code.

### `10_LayerNorm`

**`opt-round-5` (parent `opt-round-4`)** — `10_LayerNorm/opt-round-5/attempts.md`

- **Kernel / round / parent:** `10_LayerNorm` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Fused kernel always pays mask/select overhead even when rows and widths are **exact multiples** of tile sizes (`attempts.md` hypothesis).
- **Change:** Add `_layernorm_fused_aligned_kernel` fast path when shapes satisfy divisibility; skip redundant mask control on those launches.
- **Evidence:** vs r4 `compare-perf` **flat** overall; **validated branch** per attempts—correctness OK, no net perf win yet (`opt-note.md` round 5 theme).
- **Interpretation:** Scalar/control-path removal is not sufficient if memory or reduction structure still dominates—anti-signal for “aligned path alone”.

### Other kernels in the eight

`2_SwiGLU`, `10_SwigluQuant`, `11_DequantSwigluQuant`, `2_GroupNormSwish`, and `11_GroupNorm` pilot logs primarily cite **layout**, **PMR**, and **autotune** patterns in their triage headers for the mainline storylines. Map those rounds on those cards unless `attempts.md` explicitly cites this card for a scalar/control-trap mechanism.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `12_KvRmsnormRopeCache`

**`opt-round-1` (parent —)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-1` / first round.
- **Pre-change scenario:** Cache position and head indices used `int64` arithmetic everywhere despite bounded ranges.
- **Change:** Narrowed hot index math to `int32` with explicit bounds checks at tile boundaries only.
- **Evidence:** `attempts.md` value-range proof; profiler ALU mix in `summary.md` companion notes.
- **Interpretation:** KV indexing is a prime int32 vectorization target.

**`opt-round-6` (parent `opt-round-5`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** RoPE angle tables addressed with `% period` inside the innermost vector loop.
- **Change:** Hoisted period wrap into per-tile precomputed angle indices; inner loop uses direct gather without `%`.
- **Evidence:** `attempts.md` modulo removal section; correctness on non-divisible positions.
- **Interpretation:** Table lookup loops should not carry `%` unless wrap is mathematically required every lane.

### `12_Permute`

**`opt-round-1` (parent —)**

- **Kernel / round / parent:** `12_Permute` / `opt-round-1` / first round.
- **Pre-change scenario:** Rank and permutation id were runtime tensors even though JSON cases fixed them per launch.
- **Change:** Promoted permutation metadata to `tl.constexpr` / host dispatch to shrink specialization cardinality.
- **Evidence:** Compile-time log excerpt in `attempts.md`; faster JIT cache hits in `opt-note.md`.
- **Interpretation:** Permute kernels should not pay dynamic rank costs when harness fixes ranks.

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `12_Permute` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Pointer recurrence with `+=` inside the permute tile loop serialized address generation.
- **Change:** Rewrote to base-plus-offset addressing per iteration from stable `make_block_ptr` bases.
- **Evidence:** IR/scalar chain notes in `attempts.md`; `summary.md` medium tensors.
- **Interpretation:** Same “loop pointer recurrence” repair as MLP tiles, applied to permute tiles.

### `13_Cat`

**`opt-round-2` (parent `opt-round-1`)** — `13_Cat/opt-round-2/attempts.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Generic `_cat_copy_kernel` rebuilt **all four** logical coordinates per element even when concat axis makes many indices loop-invariant along a row band (`attempts.md`).
- **Change:** `_cat_row_copy_kernel` computes placement **once per row tile** and uses 2D slice addressing—removes per-element full-rank decode from the hot generic path.
- **Evidence:** Correctness passed; `compare-perf` **+94.1%** avg, **19.21×** geomean vs baseline (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** Div/mod and multi-coordinate decode are scalar-trap symptoms; row tiling is the structural fix.

### `14_AdaptiveInstanceNormalization2DBackward`

**`opt-round-14` (parent `opt-round-13`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-14/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-14` / `opt-round-13`.
- **Pre-change scenario:** Low-row large 2D path still executed **per-lane mask and bounds** work when `n_rows` and `spatial_size` are **exact multiples** of the active tile (`attempts.md`).
- **Change:** `_adain_backward_input_exact_kernel` drops redundant masks on that exact-tile regime; tail-only masking elsewhere unchanged.
- **Evidence:** Correctness passed; **+9.4%** avg, **1.26×** geomean, **2.17×** total vs baseline; **promoted** as session best (`opt-note.md`).
- **Interpretation:** Scalar-trap card overlaps **`layout-store-and-block-pointers`** when the fix is “remove impossible-branch control”.

### `13_InterleaveRope`

**`opt-round-1` (parent —)**

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-1` / first round.
- **Pre-change scenario:** Interleave phase computed from scalar `for`-like patterns lowered to per-lane serial work.
- **Change:** Vectorized pair selection with `tl.where` only where multi-lane masks were required; removed single-lane degenerate masks.
- **Evidence:** `attempts.md` mask pattern audit; `summary.md` short rope lengths.
- **Interpretation:** Rope interleave is easy to accidentally scalarize when mirroring CPU reference.

### `15_Pad`

**`opt-round-1` (parent —)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-1` / first round.
- **Pre-change scenario:** Edge detection recomputed `h`, `w` from linear index with `%` and `/` every inner step.
- **Change:** Computed spatial coordinates once per program tile; reused across vector fill lanes.
- **Evidence:** `attempts.md` index hoisting sketch; `summary.md` 2D pad margins.
- **Interpretation:** Pad edges are small but hot; hoist coordinate decode out of vector bodies.

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** `tl.where` selected between interior fill and edge sentinel with masks that were constant across lanes.
- **Change:** Split kernels / branches on `is_edge` constexpr slices instead of vector `where` on uniform lanes.
- **Evidence:** Correctness unchanged; scalar op reduction in profiler excerpt.
- **Interpretation:** Avoid degenerate `where` when the predicate is tile-uniform.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `16_Repeat`

**`opt-round-1` (parent `baseline`)** — `16_Repeat/opt-round-1/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-1` / baseline.
- **Pre-change scenario:** Prepared baseline decoded **flat output indices** with **`//` / `%`** per element and only inner-block looped to respect launch caps (`attempts.md`; cites **`scalar-latency-traps.md`**).
- **Change:** Replaced elementwise decode with **2D row-tile** fanout so placement math runs at tile granularity.
- **Evidence:** Correctness passed; **+89.8%** avg, **24.37×** geomean vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Repeat exposes div/mod scalar traps until the kernel is reframed as contiguous row work—see also **`layout-store-and-block-pointers.md`**.

### `18_Index`

**`opt-round-1` (parent `baseline`)** — `18_Index/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline **`index_select`** lowered to per-output-element index loads and **up to 4-D coordinate decode** for every store while large cases are contiguous along **`inner_size`** (`attempts.md`).
- **Change:** **Attempt 1** row-copy grid overshot **`coreDim`** on the largest shape; **attempt 2** kept row-copy semantics but **looped `inner_size` blocks inside** each program to cap launches.
- **Evidence:** Correctness passed after repair; **Avg +74.8%**, **11.03×** geomean vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Scalar-trap removal (row copy) must be paired with **launch feasibility**—grid limits are part of the scalar/geometry story on Ascend.

### `19_IndexPut`

**`opt-round-1` (parent `baseline`)** — theme in `19_IndexPut/opt-note.md` + `opt-round-1/attempts.md`

- **Kernel / round / parent:** `19_IndexPut` / `opt-round-1` / baseline.
- **Pre-change scenario:** Host-visible index dtypes and width mismatches amplified scalar-heavy lowering on small updates (`opt-note.md` round 1 theme).
- **Change:** **Normalize kernel indices to int32** before launch on the hot paths (`opt-note.md`).
- **Evidence:** Correctness passed; **+1.6%** avg, **1.02×** geomean vs baseline (`opt-note.md`); **promoted** first trunk.
- **Interpretation:** Index-heavy kernels benefit from **canonical narrow index types** before chasing tile policy.

**`opt-round-5` (parent `opt-round-2`)** — theme in `19_IndexPut/opt-note.md` + `19_IndexPut/opt-round-5/attempts.md`

- **Kernel / round / parent:** `19_IndexPut` / `opt-round-5` / `opt-round-2`.
- **Pre-change scenario:** Profiler and IR still showed a **long scalarized accumulate loop** on the large update path (`opt-note.md` round 5 theme).
- **Change:** **IR-backed** rewrite to shorten the **512-block** accumulate inner loop while preserving uniqueness semantics (`opt-note.md`).
- **Evidence:** Correctness passed; branch reached **1.22×** geomean vs baseline (`opt-note.md`); **promoted** best at that milestone.
- **Interpretation:** IndexPut remains partially **scalar-loop bound**—IR attribution belongs on this card alongside **`tiling.md`** width experiments.

### Other operators in this batch (`16_Batched*`, `17_AdamW`, `17_Embedding*`, `18_FusedAddRmsnorm`, `19_FusedResidual*`)

`17_AdamW` **r1** removes invariant per-element scalar math (see **`loop-invariant-hoisting.md`**). `16_Batched2DRopePositionEncodingBackward` rope backward work is dominated by **PMR / tiling** on other cards once baseline decode is fixed.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `20_Gather`

**`opt-round-3` (parent `opt-round-2`)** — `20_Gather/opt-round-3/attempts.md`

- **Kernel / round / parent:** `20_Gather` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Case-5 profile showed **`_gather_copy_kernel` ~99.6%** of time with **`aiv_scalar_ratio ≈ 0.999`**, **`aiv_vec_ratio = 0`**, **`Block Dim = 34560`** (`attempts.md`).
- **Change:** Rank-2 **`dim=0`** kernel mapping **multiple output rows + wide column block** per program; generic kernel remains fallback.
- **Evidence:** Correctness passed; **Avg +23.1%**, **1.34×** geomean vs baseline (`attempts.md`); **promoted** parent for gather arc.
- **Interpretation:** Dominant gather case was **scalar-launch bound**—fix program geometry before micro-address tweaks.

### `21_Scatter`

**`opt-round-6` (parent `opt-round-4`)** — `21_Scatter/opt-round-6/attempts.md`

- **Kernel / round / parent:** `21_Scatter` / `opt-round-6` / `opt-round-4`.
- **Pre-change scenario:** Remaining host/index path still fed wide indices into kernels without narrowing (`attempts.md` profiler note).
- **Change:** Validate indices then convert **once** to **`torch.int32`** before Triton sees them.
- **Evidence:** Correctness passed; **Avg +38.4%**, **2.20×** geomean vs baseline; **1.05×** geomean vs r4 (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** Same lesson as **`19_IndexPut`**—**narrow index tensors** cut scalar lowering on movement-heavy scatter.

### `24_EmbeddingDenseBackward`

**`opt-round-4` (parent `opt-round-1`)** — theme in `24_EmbeddingDenseBackward/opt-note.md` + `24_EmbeddingDenseBackward/opt-round-4/attempts.md`

- **Kernel / round / parent:** `24_EmbeddingDenseBackward` / `opt-round-4` / `opt-round-1`.
- **Pre-change scenario:** Hot path still carried **padding predicates** even when shapes are **`constexpr`**-friendly (`opt-note.md`).
- **Change:** **`constexpr` no-padding fast path** + redundant guard removal on that branch.
- **Evidence:** Correctness passed; all cases improved; **1.58×** geomean vs baseline (`opt-note.md`); **promoted** over r1.
- **Interpretation:** Scalar traps include **impossible-branch masks**—split kernels/paths when padding is provably inactive.

**`opt-round-9` (parent `opt-round-6`)** — `24_EmbeddingDenseBackward/opt-round-9/attempts.md`

- **Kernel / round / parent:** `24_EmbeddingDenseBackward` / `opt-round-9` / `opt-round-6`.
- **Pre-change scenario:** Profiler showed case-3 still paid **padding-path** masking while **`padding_idx` never appeared** in indices (`attempts.md`).
- **Change:** **Dynamic dispatch** to no-padding kernel when **`padding_idx` absent from actual input** (deterministic replay check).
- **Evidence:** Correctness passed; **Avg +39.8%**, **1.72×** geomean vs baseline (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** Host-visible **semantic guards** beat extra in-kernel scalar work.

### Other operators in this batch (`20_FusedRope*`, `21_Gaussian*`, `22_Hybrid*`, `22_Nonzero`, `23_*`, `24_KvCache*`)

`20_FusedRopeWithQkNormAndKvCacheUpdate` early rounds trimmed loads but **regressed**—session win is **PMR** (`program-multiple-rows.md`). `21_GaussianTopkSparseActivation` **`opt-round-1`** removes **`float32` staging** (see **`loop-invariant-hoisting.md`** / fusion). `22_HybridAttentionMaskPreparation` **`opt-round-8`** simplifies mask value construction after failed bool store (`layout-store-and-block-pointers.md`, **`compile_hint.md`**). `22_Nonzero` **`opt-round-1`** prefix rewrite regressed—dense fast paths live on **`tiling.md`**. `23_*` kernels are **tiling/store-shape** first. `24_KvCacheUpdateWithRopeBackward` is **PMR + grid** (`program-multiple-rows.md`, **`grid-flatten-and-ub-buffering.md`**).

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators: **`25_MaskedSoftmaxWithAttentionDropoutBackward`**, **`25_NLLLoss`**, **`26_AvgPool3d`**, **`26_MoeGroupScoreAggregationAndMasking`**, **`27_MaxPool3d`**, **`27_MultiMaskAttentionAggregation`**, **`28_Interpolate`**, **`28_MultimodalRopePositionComputationWithGridBasedIndexing`**, **`29_DynamicQuant`**, **`29_TanhGatedResidualAddBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `26_MoeGroupScoreAggregationAndMasking`

**`opt-round-1`–`opt-round-5` (parent `baseline`)** — themes in `26_MoeGroupScoreAggregationAndMasking/opt-note.md` + per-round `attempts.md`

- **Kernel / round / parent:** `26_MoeGroupScoreAggregationAndMasking` / **`opt-round-1`–`5`** / baseline.
- **Pre-change scenario:** Fusion / load-trim / **2D reduction** / **profiler-backed `BLOCK_M=4`** / **exact-grid** variants all **regressed** kernel benchmark vs baseline (`opt-note.md`).
- **Change:** Documented **negative structural experiments**—session only turned positive at **tiny-input specialization** (r6).
- **Evidence:** Each round correctness passed but **not promoted** until r6 (`opt-note.md`).
- **Interpretation:** MOE score+mask is **not** a “fuse everything” kernel—scalar/control complexity dominates failed rewrites.

### `29_TanhGatedResidualAddBackward`

**`opt-round-4` (parent `opt-round-3`)** — `29_TanhGatedResidualAddBackward/opt-round-4/attempts.md`

- **Kernel / round / parent:** `29_TanhGatedResidualAddBackward` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Partially aligned workloads still masked **every** block on streaming kernels; bf16 intermediate narrowing **failed** differential (`attempts.md`).
- **Change:** **Dual launch**: **unmasked** kernel over exact-multiple **prefix** with **`USE_MASK=False`**, then **masked** tail launch over remainder with **`USE_MASK=True`** (exact math preserved).
- **Evidence:** Correctness passed; **Avg +12.9%**, **1.17×** geomean, **1.01×** total vs baseline; case **4 −39%**, case **3 −18%** (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** **Tail-only masks** remove degenerate `where` work—classic scalar-trap repair for gated residuals.

### Other operators in this batch (`25_*`, `27_MaxPool3d`, `28_*`, `29_DynamicQuant`)

`25_NLLLoss` **`opt-round-2`** removes **div/mod decode** (see **`program-multiple-rows.md`**). `27_MaxPool3d` interior rewrite targets **MTE/strided loads** more than lane decode (**`tiling.md`**). `28_Interpolate` / `28_MultimodalRopePositionComputationWithGridBasedIndexing` / `29_DynamicQuant` emphasize **tiling / gather / chunking** on other cards.

## Gap-fill addendum (inventory alignment, 2026-05-08)

### `22_Nonzero`

**`opt-round-10` (parent `opt-round-9`)** — `22_Nonzero/opt-round-10/attempts.md`

- **Kernel / round / parent:** `22_Nonzero` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** After probe/count routing wins, compact path still had residual scalar bookkeeping in valid-count handling.
- **Change:** Compact-kernel valid-count simplification (after rejecting unsupported row-split prefix variant).
- **Evidence:** Correctness passed; small direct improvement over r9 and promoted as final best (`opt-note.md`).
- **Interpretation:** Late-stage nonzero tuning still surfaces scalar bookkeeping traps even after larger routing fixes.

### `25_NLLLoss`

**`opt-round-2` (parent `opt-round-1`)** — `25_NLLLoss/opt-round-2/attempts.md`

- **Kernel / round / parent:** `25_NLLLoss` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Mean kernel still paid flattened decode (`// spatial_size`, `% spatial_size`) on the hot unweighted path.
- **Change:** Explicit `batch x spatial-block` mapping removed per-lane decode from the hot reduction.
- **Evidence:** Correctness passed; promoted as final best (`opt-note.md`) with clear baseline-relative win.
- **Interpretation:** Div/mod decode removal is the key scalar-trap fix for this operator's representative workload.

### `27_MultiMaskAttentionAggregation`

**`opt-round-3` (parent `opt-round-2`)** — `27_MultiMaskAttentionAggregation/opt-round-3/attempts.md`

- **Kernel / round / parent:** `27_MultiMaskAttentionAggregation` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Expanding low-precision fused reduction across branches triggered differential sensitivity from reduction-order/scalarized behavior.
- **Change:** Kept only validated path portions and reverted unstable low-precision expansion.
- **Evidence:** Correctness passed but branch regressed versus round 2 (`opt-note.md`); kept as validated branch.
- **Interpretation:** Scalar-latency/card evidence includes correctness-driven anti-signals where scalar/reduction ordering dominates outcome.

### `28_Interpolate`

**`opt-round-6` (parent `opt-round-5`)** — `28_Interpolate/opt-round-6/attempts.md`

- **Kernel / round / parent:** `28_Interpolate` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Tried moving low-precision exact-half bilinear path from built-in op to Triton 2x2 kernel.
- **Change:** Low-precision exact-half bilinear dispatch experiment.
- **Evidence:** Correctness passed but severe regression in representative case 5; not promoted (`opt-note.md`).
- **Interpretation:** Scalar/control simplifications must respect backend implementation quality; replacing a tuned built-in path can backfire.
