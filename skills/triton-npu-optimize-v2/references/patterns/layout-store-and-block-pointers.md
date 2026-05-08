# Layout, Store, And Block Pointer Pattern

## Summary

Improve latency by reshaping memory layout, block-pointer dimensionality, and store granularity so the NPU sees continuous vector-friendly transfers instead of scalarized transpose or many tiny operations.

Use this when profiling or code inspection points to memory layout and transfer shape, not just tile size.

## Use When

- Multiple stores target adjacent addresses but are emitted as separate small `tl.store` operations.
- `tl.store` writes a transposed logical tensor and appears to degrade into scalar element stores.
- A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
- An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.
- A `tl.dot` operand uses `tl.trans(x).to(dtype)` before entering Cube work.
- A matmul epilogue adds bias after `tl.dot` in a way that creates unnecessary broadcast or load ordering overhead.

## Signals

### Code

- Multiple stores target adjacent addresses but are emitted as separate small `tl.store` operations.
- A store writes a transposed logical tensor and appears to degrade into scalar element stores.
- A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
- An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.

## Repairs

### Merge adjacent stores

When store offsets are provably continuous, combine separate small stores into one wider store:

```python
offs = base + tl.arange(0, BLOCK)
vals = compute_contiguous_values(...)
tl.store(out + offs, vals, mask=offs < N)
```

Do not merge stores when the destination addresses are not a continuous interval or when masks differ in a way that changes semantics.

### Avoid store transpose degradation

Shape accumulators and masks so store order matches the output's contiguous memory direction. If the current accumulator is `(N, M)` only to be transposed at store time, consider carrying it as `(M, N)` and adjusting reduction axes.

This is a layout rewrite, so re-check every reduction axis, mask broadcast, and final pointer expression.

### Raise block-pointer dimensionality

For tensors with real multidimensional contiguous layout, prefer a block pointer that models those dimensions directly:

```python
ptr = tl.make_block_ptr(
    base=x,
    shape=(T, H),
    strides=(stride_t, stride_h),
    offsets=(pid_t * BLOCK_T, 0),
    block_shape=(BLOCK_T, BLOCK_H),
    order=(1, 0),
)
tile = tl.load(ptr, boundary_check=(0, 1), padding_option="zero")
```

This is most useful when a flattened 1D pointer causes strided or non-coalesced loads across an inner dimension that is actually contiguous in memory.

### Vectorize an inner dimension

If an inner loop only walks a small dimension, include that dimension in the loaded tile and compute with an extra tensor axis. Update broadcasting and grid mapping together; if the inner dimension was part of grid partitioning, removing that grid axis may be part of the optimization.

### Let Cube handle transpose after dtype conversion

For `tl.dot` operands that currently do:

```python
b = tl.trans(b).to(tl.float16)
acc = tl.dot(a, b)
```

prefer:

```python
b = b.to(tl.float16)
acc = tl.dot(a, tl.trans(b))
```

Only apply this when the transposed tensor is directly consumed by `tl.dot`. Pure Vector code or non-dot uses do not benefit from the Cube load path.

### Bias with matmul

When a matmul always adds bias, load bias with explicit output-column offsets and add it in the epilogue shape that already matches the accumulator. Avoid implicit broadcast patterns that force extra address bookkeeping or late reshaping.

## Risks

- Layout rewrites easily swap axes by accident.
- Store merging requires continuous destinations and compatible masks.
- High-dimensional block pointers need correct `shape`, `strides`, `offsets`, `block_shape`, and `order`; one wrong field can silently benchmark a different access pattern or fail correctness.
- Vec-to-Cube transpose ordering is only a valid optimization when the final consumer is `tl.dot`.

## What To Verify After Applying

- Confirm every changed tensor shape and reduction axis in `attempts.md`.
- Run correctness on tail shapes and non-contiguous stride cases when supported.
- Benchmark against the canonical baseline and record whether the gain comes from fewer stores, better load shape, or removed transpose overhead.

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `1_RotaryMul`
- `10_LayerNorm`
- `10_SwigluQuant`
- `11_GroupNorm`
- `12_Permute`
- `13_InterleaveRope`
- `14_Split`
- `15_AttentionSoftmaxWithSoftcappingAndDropout`
- `15_Pad`
- `16_Repeat`
- `17_EmbeddingWithInitialLayernormBackward`
- `20_Gather`
- `21_Scatter`
- `22_HybridAttentionMaskPreparation`
- `22_Nonzero`
- `23_HyenaFftSizePaddingRfft`
- `23_RepeatInterleave`
- `24_KvCacheUpdateWithRopeBackward`
- `26_MoeGroupScoreAggregationAndMasking`
- `27_MultiMaskAttentionAggregation`
- `28_Interpolate`
- `25_NLLLoss`
- `28_MultimodalRopePositionComputationWithGridBasedIndexing`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `1_RotaryMul`

**`opt-round-2` (parent `opt-round-1`)** — `1_RotaryMul/opt-round-2/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Row batching in r1 still used patterns that forced awkward interleave vectorization on the rotary apply (`attempts.md` interleave discussion).
- **Change:** Contiguous **full-row** loads with `tl.reshape` / `tl.split` interleave; compile repair keeps `split` in the **source dtype** before `float32` cast so wide vector loads stay legal.
- **Evidence:** Mean **196.117→176.033µs** vs r1; **promoted** (`opt-note.md`).
- **Interpretation:** Layout + dtype staging fixes are block-pointer-adjacent: express memory as the compiler expects before widening math precision.

**`opt-round-5` (parent `opt-round-4`)** — `1_RotaryMul/opt-round-5/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Interleave row tile still used manual pointer arithmetic after r4 hotspot widening (`opt-note.md` theme: block pointers).
- **Change:** Express the interleave row tile with **Triton block pointers** instead of scalar offset chains on the hot path.
- **Evidence:** Mean **114.8722→110.7256µs** vs r4; **promoted** (`opt-note.md`).
- **Interpretation:** Canonical `layout-store-and-block-pointers` win once batching is in place—lower address-generation pressure on the interleave store path.

**`opt-round-10` (parent `opt-round-7`)** — `1_RotaryMul/opt-round-10/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-10` / `opt-round-7`.
- **Pre-change scenario:** r7 broadcast-head specialization exists on the primary dtype path; half-precision workloads still use a less specialized layout (`opt-note.md` theme: port specialization to half path).
- **Change:** Port broadcast-head specialization to the **half** execution path (layout/broadcast variant of the r7 structure).
- **Evidence:** Mean **102.3796→121.0916µs** vs r7; **not promoted** (`opt-note.md`).
- **Interpretation:** Layout recipes do not port 1:1 across dtypes—verify contiguity and vector width assumptions before treating a port as a win.

### `2_SwiGLU`

**`opt-round-1` (parent `baseline`)** — `2_SwiGLU/opt-round-1/attempts.md`

- **Kernel / round / parent:** `2_SwiGLU` / `opt-round-1` / baseline.
- **Pre-change scenario:** Wrapper `permute(...).contiguous()` introduces non-canonical strides for the non-last split case while the hot kernel expects a friendlier layout (`attempts.md`).
- **Change:** Remove wrapper transpose for non-last split; route dim-0 work through `_swiglu_split_dim0_kernel` on **contiguous** tensors after explicit layout repair.
- **Evidence:** Dominant large dim-0 case **667.879→261.844µs** vs baseline; `compare-perf` Total **2.14×** vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Implicit transpose removal is a layout-store family move—must pair with a kernel that matches the new stride contract.

### `10_SwigluQuant`

**`opt-round-1` (parent `baseline`)** — `10_SwigluQuant/opt-round-1/attempts.md`

- **Kernel / round / parent:** `10_SwigluQuant` / `opt-round-1` / baseline.
- **Pre-change scenario:** Case 5 dominated by standalone `_pack_int4_to_int8_kernel` pass (~3.7ms) larger than `_swiglu_kernel` + `_quantize_static_kernel` combined (`attempts.md` citing `baseline/perf.txt`).
- **Change:** Fuse static int4 **quantize-and-pack** into `_quantize_static_int4_pack_kernel` / `_quantize_static_int4_pack()` so benchmark `quant_mode==0 && is_int4` skips the extra pack kernel.
- **Evidence:** Case 5 latency **4298.629→3204.277µs** in round perf text; headline **+5.3%** avg, **1.06×** geomean, **1.33×** total vs baseline; **promoted** (`attempts.md` Decision).
- **Interpretation:** Removing a full extra global read/write pass is a layout/pipeline fusion win before PMR on SwiGLU itself.

**`opt-round-2` (parent `opt-round-1`)** — `10_SwigluQuant/opt-round-2/attempts.md`

- **Kernel / round / parent:** `10_SwigluQuant` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Host still runs `Sigmoid`, `Mul`, `Slice` around `_swiglu_kernel` on representative cases even after int4 fusion (`attempts.md`).
- **Change:** Rewrite `_swiglu_kernel` to load halves from the **original row-major** input and compute `gate * tl.sigmoid(gate) * value` in Triton; drop host `torch.sigmoid` / `Mul` / split setup.
- **Evidence:** Host ops disappear from perf stats but case-4 dynamic path regresses (**44.588µs** larger direct kernel); geomean **1.03×** vs baseline—**validated branch, not promoted** over r1 (`attempts.md`).
- **Interpretation:** Collapsing layout/contracts can regress if the new kernel’s load pattern is worse for a different case—still a layout-card lesson with a negative outcome.

### `11_GroupNorm`

**`opt-round-1` (parent `baseline`)** — `11_GroupNorm/opt-round-1/attempts.md`

- **Kernel / round / parent:** `11_GroupNorm` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline separates moments/affine passes with weak reuse of channel–spatial structure (`attempts.md` hypothesis).
- **Change:** **2D channel–spatial tiling** for affine reuse and structured loads/stores on the fused path (cited explicitly in Supporting evidence list with `layout-store-and-block-pointers.md`).
- **Evidence:** `compare-perf` **+91.1%** avg, **12.88×** geomean, **21.64×** total vs baseline (`opt-note.md`); **promoted** as first best.
- **Interpretation:** GroupNorm is a natural 2D tile + block-pointer candidate before PMR widening in later rounds.

### Other kernels in the eight

Later `10_LayerNorm` and `11_DequantSwigluQuant` rounds in this archive are dominated by **row batching**, **aligned fast paths**, and **compile hints** already narrated on **`program-multiple-rows.md`**, **`compile_hint.md`**, and **`autotune.md`**. `1_GELU` and `2_GroupNormSwish` do not cite this card for the pilot-span work captured here. Do not duplicate those rounds on this file without layout-specific evidence.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `12_Permute`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `12_Permute` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Permute was expressed via flattened offsets that hid which axis was contiguous in physical memory.
- **Change:** Replaced 1D decode with `tl.make_block_ptr` on the true contiguous leading dimensions; permute became `advance` on block pointers.
- **Evidence:** `attempts.md` stride/order audit; `summary.md` rank-4 permute cases.
- **Interpretation:** Permute kernels are layout-first; block pointers should model destination contiguity.

**`opt-round-4` (parent `opt-round-3`)**

- **Kernel / round / parent:** `12_Permute` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Multiple narrow `tl.store` sequences wrote logically adjacent lanes after permute.
- **Change:** Merged stores where masks and dtypes matched; carried accumulators in store-major order.
- **Evidence:** Store-count estimate in `attempts.md`; bandwidth win in `summary.md`.
- **Interpretation:** Matches “merge adjacent stores” once permute math is proven contiguous.

### `13_Cat`

**`opt-round-2` (parent `opt-round-1`)** — `13_Cat/opt-round-2/attempts.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Generic path computed **four** output coordinates per element; non-dim-0 concat only needs axes around the concat dimension (`attempts.md`).
- **Change:** Introduced **`_cat_row_copy_kernel`**: contiguous **row-slice** loads/stores along outer×concat axes with placement computed **per row tile** instead of per element.
- **Evidence:** Correctness passed; `compare-perf` **+94.1%** avg, **19.21×** geomean vs baseline (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** Concat “generic” path is a **pointer / stride layout** problem—block-pointer-style row tiles beat full-rank decode.

### `14_AdaptiveInstanceNormalization2DBackward`

**`opt-round-8` (parent `opt-round-6`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-8/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-8` / `opt-round-6`.
- **Pre-change scenario:** Manual tile offsets in the medium/large 2D backward kernel still forced long scalar address chains after PMR wins (`opt-note.md` round 8 theme).
- **Change:** Replaced manual offset arithmetic with **Triton block pointers** on the hot input/output paths while preserving `BLOCK_M`/`BLOCK_N` launch geometry from r6.
- **Evidence:** Correctness passed; faster than baseline overall but **not promoted** over r6 on parent geomean (`opt-note.md`—structural win without winning harness mix).
- **Interpretation:** Layout-store cleanup is a follow-on pass after row batching—measure against **parent**, not only baseline.

**`opt-round-14` (parent `opt-round-13`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-14/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-14` / `opt-round-13`.
- **Pre-change scenario:** Low-row large 2D path still carried **full masks** even when `n_rows` and `spatial_size` divide **`BLOCK_M=8`** and **`BLOCK_N=512`** (`attempts.md`; pattern **`layout-store-and-block-pointers`**).
- **Change:** Added **`_adain_backward_input_exact_kernel`**: unmasked loads/stores for exact-tile low-row large workloads; streaming branches unchanged.
- **Evidence:** Correctness passed; **+9.4%** avg, **1.26×** geomean, **2.17×** total vs baseline; **promoted** as session best (`opt-note.md`).
- **Interpretation:** When divisibility is provable, strip masks—same card family as other exact-tile fast paths.

### `13_InterleaveRope`

**`opt-round-5` (parent `opt-round-4`)** — `13_InterleaveRope/opt-round-5/attempts.md`

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Outputs were materialized through a transpose-shaped scratch tensor to simplify indexing.
- **Change:** Wrote directly in destination layout using 2D `tl.arange` grids aligned to head×pair axes.
- **Evidence:** Axis-swap checklist in `attempts.md`; correctness on non-square heads.
- **Interpretation:** Rope helpers still pay transpose-store costs unless layout is fixed early.

### `14_Split`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `14_Split` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Each split segment used independent pointer arithmetic that prevented merged wide stores.
- **Change:** Built contiguous destination offsets per chunk and issued single-vector stores per segment tile.
- **Evidence:** `summary.md` split-axis benchmark; `attempts.md` pointer diff.
- **Interpretation:** Split output is often contiguous along chunk; exploit that explicitly.

### `15_Pad`

**`opt-round-1` (parent —)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-1` / first round.
- **Pre-change scenario:** N-dimensional pad used nested scalar loops for boundary writes instead of vectorized `tl.store` bands.
- **Change:** Flattened interior fill into one wide `tl.arange` store; kept edge cases masked separately.
- **Evidence:** `attempts.md` interior vs edge decomposition; `summary.md` large tensor fill.
- **Interpretation:** Pad is a store-layout problem on Ascend; treat interior as bulk memcpy shape.

**`opt-round-4` (parent `opt-round-3`)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Reflect-pad reused transposed scratch for symmetric reads from opposite edges.
- **Change:** Recomputed mirror indices into a contiguous staging tile then single-pass store to output.
- **Evidence:** Correctness matrix for reflect vs replicate modes; `summary.md` 2D reflect case.
- **Interpretation:** Symmetric pad modes still benefit from eliminating transpose-at-store when safe.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-12` (parent `opt-round-11`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-12` / `opt-round-11`.
- **Pre-change scenario:** Softmax outputs were accumulated transposed relative to downstream matmul consumer layout.
- **Change:** Carried softmax probabilities in consumer-major order; fused dropout mask application in that layout.
- **Evidence:** Consumer contract referenced in `attempts.md`; `summary.md` end-to-end attention microbench.
- **Interpretation:** Attention fusion chains fail if epilogue layout fights the next `tl.dot` operand layout.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `16_Repeat`

**`opt-round-1` (parent `baseline`)** — `16_Repeat/opt-round-1/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-1` / baseline (`baseline/triton_16_Repeat_prepared.py` per attempts header).
- **Pre-change scenario:** Prepared baseline still decoded **flat output indices** with `//` / `%` per element and looped inner blocks only to respect launch caps (`attempts.md`; cites **`layout-store-and-block-pointers.md`** + **`scalar-latency-traps.md`**).
- **Change:** Replaced flat element kernel with **2D tile** fanout: load contiguous **row tiles** once, fan out across repeat factors at store time.
- **Evidence:** Correctness passed; **+89.8%** avg, **24.37×** geomean, **46.53×** total vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Repeat is a layout problem first—model destination contiguity instead of linearized decode.

**`opt-round-11` (parent `opt-round-10`)** — `16_Repeat/opt-round-11/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-11` / `opt-round-10`.
- **Pre-change scenario:** Exact full-tile hot path still used manual offset recurrence on the fastest branch (`opt-note.md` round 11 theme).
- **Change:** Rewrote exact fulltile hot path with **`tl.make_block_ptr` / `advance`** style addressing (per attempts).
- **Evidence:** Correctness passed; headline stayed near r10 but **did not beat** r10 on geomean (`opt-note.md`); **not promoted**.
- **Interpretation:** Block pointers are not automatic wins—validate against parent when the hot path is already near-optimal.

### `17_EmbeddingWithInitialLayernormBackward`

**`opt-round-7` (parent `opt-round-6`)** — `17_EmbeddingWithInitialLayernormBackward/opt-round-7/attempts.md`

- **Kernel / round / parent:** `17_EmbeddingWithInitialLayernormBackward` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Fixed-width loads/stores on the 4096 hidden fast path still used **manual row pointers** inside the inner tile (`opt-note.md` round 7 theme).
- **Change:** Expressed fixed-width row loads/stores with **block pointers** on the hot path.
- **Evidence:** Correctness passed; large fixed-path case **flat**, average regression **−4.0%** vs baseline (`opt-note.md`); **not promoted**.
- **Interpretation:** Layout-store rewrite after near-baseline tuning can still regress—pair with profiler proof before landing.

### `18_Index`

**`opt-round-1` (parent `baseline`)** — `18_Index/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline `index_select` lowered to non-contiguous gather patterns on large cases (`opt-note.md` / attempts).
- **Change:** **Flatten contiguous `index_select`** into **row-copy** Triton launches with explicit contiguous pointer windows.
- **Evidence:** Correctness passed; **11.03×** geomean vs baseline (`opt-note.md`); **promoted** first major trunk.
- **Interpretation:** When the gather collapses to memcpy-shaped work, block-pointer-friendly row copies dominate.

### `19_FusedResidualRmsNormBackward`

**`opt-round-1` (parent `baseline`)** — `19_FusedResidualRmsNormBackward/opt-round-1/attempts.md`

- **Kernel / round / parent:** `19_FusedResidualRmsNormBackward` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline split **`grad_x`** work across multiple kernels with extra global traffic (`opt-note.md` / attempts).
- **Change:** **Fused** split `grad_x` paths into **one full-row** Triton kernel with contiguous stores.
- **Evidence:** Correctness passed; **+50.6%** avg, **2.08×** geomean vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Residual+RMS backward benefits from fused row stores—layout pressure drops before grad_weight tuning.

### Other operators in this batch (`16_Batched2DRopePositionEncodingBackward`, `17_AdamW`, `19_IndexPut`)

`16_Batched2DRopePositionEncodingBackward` batch-2 narrative is dominated by **program geometry / tile ladders** on **`program-multiple-rows.md`** and **`tiling.md`**. `17_AdamW` and `19_IndexPut` emphasize **LICM**, **scalar/accumulate**, and **compile hints** more than block-pointer rewrites in their cited attempts. Map remaining rounds on those cards unless `attempts.md` foregrounds layout-store evidence.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `20_Gather`

**`opt-round-5` (parent `opt-round-3`)** — `20_Gather/opt-round-5/attempts.md`

- **Kernel / round / parent:** `20_Gather` / `opt-round-5` / `opt-round-3`.
- **Pre-change scenario:** Rank-2 **`dim=0`** specialization from r3 still used **manual offset recurrence** for contiguous index/output slices while case-5 remained scalar-heavy (`attempts.md`).
- **Change:** Replaced contiguous index/output paths with **`tl.make_block_ptr` / advance** style loads and stores; kept dynamic gathered-value loads only where semantics require.
- **Evidence:** Correctness passed; `compare-perf` **Avg +23.4%**, **1.34×** geomean, **1.02×** total vs baseline—session **final best** (`opt-note.md`).
- **Interpretation:** After rank/dim specialization, **block-pointer cleanup** is the low-risk layout win—autotune over row tiles was **abandoned** in the same round for compile/search churn (`attempts.md`).

### `21_Scatter`

**`opt-round-2` (parent `opt-round-1`)** — `21_Scatter/opt-round-2/attempts.md`

- **Kernel / round / parent:** `21_Scatter` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 still computed **`outer_idx = offsets // (scatter_extent * inner_size)` per element** on large `inner_size` (`attempts.md`).
- **Change:** **`_scatter_copy_row_inner_kernel`**: one program owns one **`[outer, axis]` row** and walks **`inner_size`** with `tl.arange` when launch product stays under the **65535** cap; fallback keeps r1 kernel.
- **Evidence:** Correctness passed; **Avg +36.5%**, **2.08×** geomean vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Scatter copies are layout problems—**row+inner** mapping removes rank-decode from the inner loop.

### `22_HybridAttentionMaskPreparation`

**`opt-round-4` (parent `opt-round-3`)** — `22_HybridAttentionMaskPreparation/opt-round-4/attempts.md`

- **Kernel / round / parent:** `22_HybridAttentionMaskPreparation` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Autotuned r3 still rebuilt **`ptrs = offs_m[:, None] * source_length + offs_n[None, :]`** every tile; large cases regressed vs r2 (`attempts.md`).
- **Change:** **Block-pointer store** for the full 2D mask matching **`(target_length, source_length)`** layout.
- **Evidence:** Correctness passed; `compare-perf` vs baseline **Avg +61.6%**, **3.21×** geomean; parent vs r3 recovered large-case tail (`attempts.md`); **validated branch** toward r6.
- **Interpretation:** When stores are naturally **2D contiguous**, block pointers beat flattened pointer tensors even after autotune lands.

### Other operators in this batch (`20_FusedRope*`, `21_Gaussian*`, `22_Nonzero`, `23_*`, `24_*`)

`20_FusedRopeWithQkNormAndKvCacheUpdate` **`opt-round-9`** splits hot paths into **fixed-shape** kernels (`opt-note.md`)—pair with **`program-multiple-rows.md`** + **`compile_hint.md`**. `21_GaussianTopkSparseActivation` and `22_Nonzero` are mostly **tiling / routing / hints** on other cards. `23_RepeatInterleave` store-shape tuning is **`tiling.md`**. `24_KvCacheUpdateWithRopeBackward` **`opt-round-2`** block-pointer store **regressed** vs r1 (`opt-note.md`)—anti-signal for blind pointer refactors without PMR alignment.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`28_MultimodalRopePositionComputationWithGridBasedIndexing`** (see also **`25_NLLLoss`** on **`program-multiple-rows.md`**). Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `28_MultimodalRopePositionComputationWithGridBasedIndexing`

**`opt-round-6` (parent `opt-round-4`)** — theme in `28_MultimodalRopePositionComputationWithGridBasedIndexing/opt-note.md` + `28_MultimodalRopePositionComputationWithGridBasedIndexing/opt-round-6/attempts.md`

- **Kernel / round / parent:** `28_MultimodalRopePositionComputationWithGridBasedIndexing` / `opt-round-6` / `opt-round-4`.
- **Pre-change scenario:** Host/helper overhead reduced (r3–r4); dominant cost returned to **Triton gather** layout on A5 (`opt-note.md`).
- **Change:** **Retile gather kernel** for A5 vector widths and launch shape.
- **Evidence:** Correctness passed; **strong vs baseline** (`opt-note.md`); **promoted** before hidden-128 repair in r7.
- **Interpretation:** Multimodal rope uses **bilinear gather**—tile gather outputs like a layout problem, not only host Python cleanup.

**`opt-round-7`–`opt-round-9` (parents per `opt-note.md`)** — `28_MultimodalRopePositionComputationWithGridBasedIndexing/opt-round-7`–`9`

- **Kernel / round / parent:** same operator / **r7–r9** / prior best.
- **Pre-change scenario:** Outliers at **`hidden=128`** and very large hidden still missed tile tiers (`opt-note.md`).
- **Change:** **Smaller tile** for hidden-128 outlier; add **very-large-hidden** and **`hidden-512`** tiers progressively (`opt-note.md` themes).
- **Evidence:** Correctness passed; each step **beat prior best** on scored mix (`opt-note.md`); set parent for r10 **`BLOCK_SIZE=1024`** tier (**`tiling.md`**).
- **Interpretation:** Gather+bilinear needs **explicit hidden-dimension ladder**—one retile rarely fits all multimodal widths.

### `26_MoeGroupScoreAggregationAndMasking`

**`opt-round-3` (parent `baseline`)** — `26_MoeGroupScoreAggregationAndMasking/opt-round-3/attempts.md`

- **Kernel / round / parent:** `26_MoeGroupScoreAggregationAndMasking` / `opt-round-3` / baseline.
- **Pre-change scenario:** Kernel still used many independent fixed-width loads/reduction chains per token and repeated address-generation structure.
- **Change:** Rewrote reduction around a single 2D `(group, expert)` load and parallel `tl.max`/`tl.argmax`-style reduction shape.
- **Evidence:** Correctness passed, but `compare-perf` regressed all representative cases in `attempts.md`; branch not promoted.
- **Interpretation:** Useful anti-signal for this card: simply regularizing load/store shape is insufficient if the underlying reduction/control mix still mismatches the hardware.

### `27_MultiMaskAttentionAggregation`

**`opt-round-1` (parent `baseline`)** — `27_MultiMaskAttentionAggregation/opt-round-1/attempts.md`

- **Kernel / round / parent:** `27_MultiMaskAttentionAggregation` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline applied mask per class in repeated flattened passes, causing redundant global movement around the real row-by-class reduction.
- **Change:** Added `_masked_attention_reduce_rows_kernel` that consumes `(rows, ref_len)` and emits all class scores per row in one pass, with fallback dispatch retained for low-precision and `mode="max"` correctness.
- **Evidence:** Correctness passed after dispatch narrowing; `compare-perf` vs baseline reported **Avg +30.3%**, **Geomean 1.56x**, **Total 1.05x** in `attempts.md`; promoted.
- **Interpretation:** Layout/store consolidation can deliver first-order wins when class-wise reductions are re-expressed as one row-major fused pass.

### Other operators in this batch (`25_*`, `26_*`, `27_*`, `28_Interpolate`, `29_*`)

`28_Interpolate` uses **2D tile kernels** and **exact-scale specials**—see **`tiling.md`**. `27_MaxPool3d` strip staging is **`tiling.md`**. `29_DynamicQuant` / `29_TanhGatedResidualAddBackward` do not foreground **block-pointer stores** in cited attempts.

## Gap-fill addendum (inventory alignment, 2026-05-08)

### `10_LayerNorm`

**`opt-round-13` (parent chain in `opt-note.md`)** — `10_LayerNorm/opt-round-13/attempts.md`

- **Kernel / round / parent:** `10_LayerNorm` / `opt-round-13` / parent per `opt-note.md`.
- **Pre-change scenario:** LayerNorm path had moved beyond first-order algorithm changes and focused on memory-layout regularity.
- **Change:** Layout/store-oriented refinement step cited in attempts.
- **Evidence:** Correctness passed; retained as validated late-round evidence.
- **Interpretation:** LayerNorm often needs storage-layout cleanup after earlier reduction/tiling milestones.

### `22_Nonzero`

**`opt-round-1` (parent `baseline`)** — `22_Nonzero/opt-round-1/attempts.md`

- **Kernel / round / parent:** `22_Nonzero` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline compaction path was expensive on large inputs.
- **Change:** Replaced full elementwise mask-prefix materialization with tile-count prefix flow.
- **Evidence:** Correctness passed but performance regressed (`opt-note.md`); not promoted.
- **Interpretation:** Serves as anti-signal: layout/prefix rewrites need density-aware routing to pay off.

### `23_HyenaFftSizePaddingRfft`

**`opt-round-1` (parent `baseline`)** — `23_HyenaFftSizePaddingRfft/opt-round-1/attempts.md`

- **Kernel / round / parent:** `23_HyenaFftSizePaddingRfft` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline padding path used less regular row/width traversal.
- **Change:** Row-tiled contiguous padding kernel shape.
- **Evidence:** Correctness passed; strong baseline-relative win (`opt-note.md`); promoted.
- **Interpretation:** FFT-padding setup is highly layout-sensitive from the first round.

### `23_RepeatInterleave`

**`opt-round-9` (parent `opt-round-7`)** — `23_RepeatInterleave/opt-round-9/attempts.md`

- **Kernel / round / parent:** `23_RepeatInterleave` / `opt-round-9` / `opt-round-7`.
- **Pre-change scenario:** Very-large repeat-2 path remained dominated by store-shape movement.
- **Change:** Profile-backed store-shape retuning for the float32 very-large branch.
- **Evidence:** Correctness passed; promoted before final round 10 refinement (`opt-note.md`).
- **Interpretation:** RepeatInterleave improvements remained tightly coupled to store-layout decisions.

### `24_KvCacheUpdateWithRopeBackward`

**`opt-round-2` (parent `opt-round-1`)** — `24_KvCacheUpdateWithRopeBackward/opt-note.md` + `opt-round-2/attempts.md`

- **Kernel / round / parent:** `24_KvCacheUpdateWithRopeBackward` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 PMR baseline was strong; round-2 tested block-pointer store plus small-head tile tweak.
- **Change:** Block-pointer store rewrite on top of row-batched kernel.
- **Evidence:** Correctness passed but regressed versus round 1 (`opt-note.md`); not promoted.
- **Interpretation:** Important anti-signal: pointer-form changes can hurt without matching launch/PMR regime.

### `28_Interpolate`

**`opt-round-7` (parent `opt-round-5`)** — `28_Interpolate/opt-note.md` + `opt-round-7/attempts.md`

- **Kernel / round / parent:** `28_Interpolate` / `opt-round-7` / `opt-round-5`.
- **Pre-change scenario:** Exact 2x downsample wins were in place, but upsample path still needed dedicated shape handling.
- **Change:** Added exact 2x bilinear upsample Triton kernel with layout-regular output traversal.
- **Evidence:** Correctness passed; promoted with strong baseline-relative improvement (`opt-note.md`).
- **Interpretation:** Interpolate performance depends on exact-scale layout-specialized kernels, not only generic formula reuse.

### `25_NLLLoss`

**`opt-round-1` (parent `baseline`)** — `25_NLLLoss/opt-round-1/attempts.md`

- **Kernel / round / parent:** `25_NLLLoss` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline unweighted-mean path carried more generic reduction/materialization overhead.
- **Change:** Partial-reduction layout rewrite for the unweighted mean path.
- **Evidence:** Correctness passed; validated branch with modest geomean win but near-flat total speed (`opt-note.md`).
- **Interpretation:** NLLLoss required layout reshaping first, then explicit batch/spatial mapping in round 2 for the clear promotion.
