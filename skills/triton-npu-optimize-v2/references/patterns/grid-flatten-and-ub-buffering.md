# Grid Flattening And UB Buffering Pattern

## Summary

Change work distribution and UB staging when latency is dominated by too many logical tasks, uneven per-core work, physical-core load balance problems, or tiny row-wise memory transfers after a gather/scatter style rewrite.

This pattern complements `program-multiple-rows`: that pattern widens row-wise work inside a program, while this one focuses on flattening logical work onto physical cores and then batching memory movement inside each core.

## Use When

- The logical grid is much larger than the physical AICore or VectorCore count.
- Work is partitioned by batch or sequence buckets with visible load imbalance.
- Each program processes many tiny rows after grid-to-physical-core mapping.
- Gather-like code has continuous destination rows but still stores one row at a time.
- Scatter-weight-gradient-like code has repeated row loads that can be batched from continuous source rows.

## Signals

### Code

- The logical grid is much larger than the physical AICore or VectorCore count.
- Work is partitioned by batch or sequence buckets that create visible load imbalance.
- Each physical program still processes many tiny rows or row-at-a-time transfers after grid mapping.

### Profile

- Latency is dominated by too many logical tasks, uneven per-core work, or tiny row-wise memory transfers after a gather or scatter style rewrite.

## Repairs

### Flatten logical tasks

Treat all independent logical work items as one continuous stream and split that stream evenly across physical cores:

```python
pid = tl.program_id(0)
task_start = pid * TASKS_PER_CORE
offs = task_start + tl.arange(0, TASKS_PER_CORE)
mask = offs < TOTAL_TASKS
```

Compute `TASKS_PER_CORE` on the host and pass it as `tl.constexpr` when it controls tensor shapes or loop bounds. Prefer uniform per-core loop structure and masks over early returns that create uneven control flow.

### Map logical grid to physical grid

When `logical_tasks > num_cores`, launch `num_cores` programs and loop over logical tasks inside the kernel. Keep `NUM_TASKS` and `NUM_CORES` as explicit constants so the compiler can simplify loop bounds.

This helps only when each physical program has enough work to amortize the internal loop. If the original grid is near the physical core count, the extra loop can be overhead.

### UB aggregate writes

After physical-core mapping, if each program produces several rows with continuous destination addresses, stage a small group of rows in UB and issue a wider 2D store. Choose `SUB_BLOCK_SIZE` so the staged rows plus live compute tensors fit UB.

Do not use aggregate writes for genuinely scattered destinations.

### UB bulk reads

For row-wise reductions or gradient accumulation where several consecutive source rows are loaded one at a time, load a small 2D slab into UB and select rows from the slab. Keep a fallback path if row indices are not continuous enough for a bulk read to be correct.

## Dependencies

- Apply `grid-flatten-and-ub-buffering` only after the kernel's basic access semantics are clear.
- UB aggregate write and UB bulk read usually depend on a prior grid-to-physical-core or flattening rewrite that gives each program multiple rows.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` separately enough to explain the result.

## Risks

- Flattening can erase useful multidimensional locality if offsets are decoded poorly.
- Physical-grid loops can hurt small workloads.
- UB staging can exceed UB capacity or reduce occupancy.
- Gather and scatter have different address-continuity requirements; do not reuse aggregate-write logic for scattered stores.

## What To Verify After Applying

- Record the physical core assumption and how it is discovered or passed.
- Validate edge cases where `TOTAL_TASKS` is not divisible by `NUM_CORES`.
- Benchmark small and large shapes if the operator supports both.
- If the win depends on row continuity, add that condition to the round summary.

## Related Patterns

- Complements `program-multiple-rows`: that pattern widens row-wise work inside a program, while this pattern flattens logical work onto physical cores and batches memory movement inside each core.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` with enough separation to explain the result.

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `16_Repeat`
- `22_Nonzero`
- `21_Scatter`
- `24_KvCacheUpdateWithRopeBackward`
- `29_DynamicQuant`
- `29_TanhGatedResidualAddBackward`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../attempts.md`, `opt-note.md`. Mandatory five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `11_DequantSwigluQuant`

**`opt-round-7` (parent `opt-round-6`)** — `11_DequantSwigluQuant/opt-round-7/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Profiler shows **large launch gaps** between `_swiglu_multiply_kernel` and surrounding wrapper vector ops—too many small programs and hand-offs vs fused work (`attempts.md` diagnosis).
- **Change:** Widen the fused dynamic path across column tiles and simplify quantization so fewer separate launches participate in the hot region (overlaps with **`cache_use.md`** fusion-locality story; grid symptom is launch fragmentation).
- **Evidence:** Geomean **1.84×**, Total **2.07×** vs baseline; **promoted** per `attempts.md` Decision after r6 plateau (`opt-note.md`).
- **Interpretation:** When logs show “gaps” between kernels, flattening/fusing is the occupancy-adjacent fix even if this card was not the literal citation line in `attempts.md`.

### Other kernels among the eight

No `opt-round-*/attempts.md` in this pilot eight cites **`grid-flatten-and-ub-buffering.md`** in the supporting pattern list. Rotary occupancy effects are narrated under **`program-multiple-rows.md`** + **`autotune.md`**. Do not add synthetic grid-flatten five-field entries without a log-backed citation or profiler note that matches this card’s Summary.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `12_KvRmsnormRopeCache`

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Logical programs tracked individual cache slots while physical core count was far smaller, causing imbalance.
- **Change:** Flattened cache-slot work onto `TASKS_PER_CORE` buckets; buffered KV fragments in UB per bucket.
- **Evidence:** Load-balance table in `attempts.md`; `summary.md` long-cache stress.
- **Interpretation:** KV cache kernels hit this card’s “many logical tasks” signal early.

### `13_Cat`

**`opt-round-19` (parent `opt-round-17`)** — `13_Cat/opt-round-19/attempts.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-19` / `opt-round-17`.
- **Pre-change scenario:** After **`BLOCK_SIZE=16384`** flat tiles (r17), refreshed profiling still showed **large Block Dim** on the pure transfer kernel—too many thin programs relative to saturated vector body (`attempts.md` reusing r16/`r18` evidence).
- **Change:** Kept **16384** tile width but **lowered dim-0 flat-path launch cap** from **4096→2048** so each program owns more contiguous copy work per launch.
- **Evidence:** Correctness passed; `compare-perf` **+95.4%** avg, **55.49×** geomean, **300.45×** total vs baseline (`attempts.md`); **final best** per `opt-note.md`.
- **Interpretation:** Pure memcpy-style kernels hit **launch/flatten** limits after width saturates—this card covers “merge programs” once UB width is maxed.

### `13_InterleaveRope`

**`opt-round-6` (parent `opt-round-5`)**

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** After layout fixes, programs still mapped 1:1 to tiny index pairs with poor occupancy.
- **Change:** Packed consecutive interleave pairs per core task; staged rope tables in UB for the whole task window.
- **Evidence:** Core utilization commentary; `summary.md` high-frequency rope pattern.
- **Interpretation:** Tiny row-like units should be flattened before chasing micro-opts.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-13` (parent `opt-round-12`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-13` / `opt-round-12`.
- **Pre-change scenario:** After autotune and pipeline work (r12), sequence tiles were still fine-grained enough that MTE bursts stayed short relative to vector softmax.
- **Change:** Merged logical seq blocks onto fewer programs while using UB double-buffer for score tiles feeding softmax.
- **Evidence:** Timeline gap reduction noted in `attempts.md`; `summary.md` medium seqlen.
- **Interpretation:** Attention benefits from flattening when MTE bursts are too small to amortize startup.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `16_Batched2DRopePositionEncodingBackward`

**`opt-round-4` (parent `opt-round-1`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-4/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-4` / `opt-round-1`.
- **Pre-change scenario:** Single **`BLOCKS_PER_PROGRAM`** choice from r1 improved large tensors but **hurt tiny** cases (`attempts.md` tying r2–r3 lessons).
- **Change:** **Host dispatch** selects **`BLOCKS_PER_PROGRAM=2`** only for **`n_elements <= 1024`**, otherwise **`4`**, keeping **`BLOCK_SIZE=256`**.
- **Evidence:** Correctness passed; **Avg +42.1%**, **1.83×** geomean, **2.50×** total vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** Grid flattening must be **size-gated**—tiny tensors need fewer blocks per program to avoid oversize programs.

### `18_Index`

**`opt-round-1` (parent `baseline`)** — `18_Index/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-1` / baseline.
- **Pre-change scenario:** First row-copy launch shape used **one program per `(selected row, inner block)`**, blowing **`coreDim`** on the largest `inner_size × index_size` (`attempts.md` attempt 1 failure).
- **Change:** Repaired to **one program per selected row** with an **inner `inner_size` loop** so grid width stays within hardware limits.
- **Evidence:** After repair, correctness passed and large cases sped up dramatically (`attempts.md`); **promoted** (see also **`scalar-latency-traps.md`**).
- **Interpretation:** **Launch-cap flattening** is mandatory when widening vector bodies—never treat grid products as unbounded.

### `17_AdamW`

**`opt-round-3` (parent `opt-round-2`)** — `17_AdamW/opt-round-3/attempts.md`

- **Kernel / round / parent:** `17_AdamW` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Profiler showed **transfer-heavy** vector body with modest scalar share; program count still high on large tensors (`attempts.md`).
- **Change:** Each program walks **two adjacent 4096-element chunks** before advancing the grid, preserving r2’s contiguity annotations.
- **Evidence:** Correctness passed; mixed per-case deltas with **Avg +2.0%**, **1.02×** geomean vs baseline (`attempts.md`); **promoted** for large-case wins.
- **Interpretation:** **Chunked flattening** along 1D contiguous spans reduces launch pressure when the inner body is already vectorized.

### Other operators in this batch (`16_Repeat`, `17_Embedding*`, `18_FusedAddRmsnorm`, `19_*`)

`16_Repeat` session best is **`program-multiple-rows.md`** + **`tiling.md`**. `19_IndexPut` **`opt-round-2`** scales launch width with update count (`19_IndexPut/opt-note.md`)—complementary **grid policy** without UB narrative in the cited attempts.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `21_Scatter`

**`opt-round-2` (parent `opt-round-1`)** — `21_Scatter/opt-round-2/attempts.md`

- **Kernel / round / parent:** `21_Scatter` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Row-inner rewrite could exceed **hardware launch cap** on full tensors (`attempts.md`).
- **Change:** Dispatch **`_scatter_copy_row_inner_kernel`** only when **`outer_size * scatter_extent * inner_blocks <= 65535`**; otherwise fall back to r1.
- **Evidence:** Correctness passed; large-case **−6.76%** latency vs parent on scored case (`attempts.md`); **promoted** trunk.
- **Interpretation:** **Conditional flattening**—new geometry must be **launch-feasible** on Ascend before replacing the generic path.

### `24_KvCacheUpdateWithRopeBackward`

**`opt-round-3` (parent `opt-round-1`)** — `24_KvCacheUpdateWithRopeBackward/opt-round-3/attempts.md`

- **Kernel / round / parent:** `24_KvCacheUpdateWithRopeBackward` / `opt-round-3` / `opt-round-1`.
- **Pre-change scenario:** r1 still used **flattened** `(batch, head, row_block) → pid` with **div/mod decode** on the hot path (`attempts.md`).
- **Change:** **Direct 2D grid** **`(batch_head, row_block)`**; removed flattened pid recovery and extra launch argument.
- **Evidence:** Correctness passed; **Avg +55.6%**, **2.46×** geomean vs baseline; modest parent win over r1 (`attempts.md`); **promoted**.
- **Interpretation:** When mapping is naturally **2D**, explicit grids beat decode-from-linear-pid for scalar and scheduling pressure.

### Other operators in this batch (`20_*`, `22_Nonzero`, `23_*`)

`20_Gather` launch shaping is tied to **scalar-dominant dim=0** specialization (`scalar-latency-traps.md`). `22_Nonzero` **`opt-round-9`** removes duplicate **count** launches when probe covers all tiles—see **`cache_use.md`**. `23_*` FFT/repeat sessions are **`tiling.md`**-first.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`29_DynamicQuant`**, **`29_TanhGatedResidualAddBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `29_DynamicQuant`

**`opt-round-4` (parent `opt-round-3`)** — theme in `29_DynamicQuant/opt-note.md` + `29_DynamicQuant/opt-round-4/attempts.md`

- **Kernel / round / parent:** `29_DynamicQuant` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Wide steady-state programs still exited after **one** full-width chunk (`opt-note.md`).
- **Change:** Each program consumes **two adjacent full chunks** on the wide steady-state path before advancing the grid.
- **Evidence:** Correctness passed; **Avg +47.2%**, **1.90×** geomean vs baseline (`opt-note.md`); **promoted** toward r9 row-group ladder (**`program-multiple-rows.md`**).
- **Interpretation:** **Chunked flattening** along width reduces launch count when masks are steady-state heavy.

### `29_TanhGatedResidualAddBackward`

**`opt-round-4` (parent `opt-round-3`)** — `29_TanhGatedResidualAddBackward/opt-round-4/attempts.md`

- **Kernel / round / parent:** `29_TanhGatedResidualAddBackward` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Single launch mixed **fully aligned prefix** with **partial tail** under one mask policy (`attempts.md`).
- **Change:** **Two launches per kernel** when needed: full **`USE_MASK=False`** pass over prefix, **`USE_MASK=True`** pass over tail only.
- **Evidence:** Correctness passed; **Avg +12.9%**, **1.17×** geomean vs baseline (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** **Grid split by alignment class**—prefix/tail is a launch-geometry decision, not only a mask tweak.

### Other operators in this batch (`25_*`–`28_*`)

No other batch-5 operator in this pilot foregrounds **multi-chunk-per-program** or **split-grid** levers in the cited `opt-note.md` arcs as strongly as the two entries above; **`25_MaskedSoftmaxWithAttentionDropoutBackward`** uses **width-tier flat blocks** on **`tiling.md`** instead.
