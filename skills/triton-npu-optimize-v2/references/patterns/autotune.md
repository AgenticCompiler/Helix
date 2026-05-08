# Triton Autotune Pattern

## Summary

Make use of autotune in Triton to optimize parameters automatically. Some analysis is
still needed to set the possible values of parameters to try (limit the number of combinations
to try to at most 20).

## Use When

- The kernel already has several plausible tile or launch parameter choices, and the main structure looks reasonable.
- Manual parameter picking is likely leaving performance on the table, but the search space can still be kept small and bounded.

## Detail

Some examples of using autotune:

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    ...
```

```python
BS_LIST = [32, 64]

@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
```

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `1_GELU`
- `1_RotaryMul`
- `15_AttentionSoftmaxWithSoftcappingAndDropout`
- `20_Gather`
- `22_HybridAttentionMaskPreparation`
- `24_EmbeddingDenseBackward`
- `29_DynamicQuant`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/{1_GELU,1_RotaryMul,2_GroupNormSwish,2_SwiGLU,10_LayerNorm,10_SwigluQuant,11_DequantSwigluQuant,11_GroupNorm}/opt-round-*/attempts.md` and each operator `opt-note.md`. The bench tree lives under gitignored `workspace/`—discover files with `find` or read by absolute path (`skills/triton-npu-kernel-bench-logs/SKILL.md`). Logs cite legacy `.codex/skills/triton-npu-optimize/references/patterns/*.md`; v2 work still maps here by mechanism against `skills/triton-npu-optimize-v2/references/pattern_index.md`. Every **round** below uses the mandatory five-field bullet list from `skills/triton-npu-kernel-bench-logs/SKILL.md`; the closing **`###` “no autotune”** block is intentionally **not** a round entry.*

### `1_GELU`

**`opt-round-1` (parent `baseline`)** — `1_GELU/opt-round-1/attempts.md`

- **Kernel / round / parent:** `1_GELU` / `opt-round-1` / baseline `baseline/baseline_triton_1_GELU.py`.
- **Pre-change scenario:** Baseline only grows `BLOCK_SIZE` on launch-limit escape; `baseline/perf.txt` shows case 5 dominates total latency while smaller cases stay relatively cheap (`attempts.md` hypothesis).
- **Change:** Host/kernel default launch uses fixed `BLOCK_SIZE=4096` instead of dynamic picker for the hot contiguous path.
- **Evidence:** `run-test` + `compare-result` balanced pass; `compare-perf` Geomean **1.02×**, Total **1.01×** vs baseline with per-case deltas listed in attempts (Attempt 1).
- **Interpretation:** First bounded launch-shape move; aligns with `opt-note.md` “fixed 4096-element launch”.

**`opt-round-2` (parent `opt-round-1`)** — `1_GELU/opt-round-2/attempts.md`

- **Kernel / round / parent:** `1_GELU` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 fixed `4096` for the large contiguous path but left smaller tensors on an awkward launch size for the harness mix (`opt-note.md` / attempts).
- **Change:** `_select_block_size()` restores small tensors to `1024`, keeps `4096` for ≥65536 elements, preserves oversize `coreDim` policy from prior tiering.
- **Evidence:** `compare-perf` Geomean **1.10×** vs baseline; Total speedup **1.00×** vs baseline; `opt-note.md` keeps as **validated branch** (not best on total-speed).
- **Interpretation:** Launch heuristics need explicit small-shape escape hatches; a geomean win can coexist with flat total-speed.

**`opt-round-3` (parent `opt-round-2`)** — `1_GELU/opt-round-3/attempts.md`

- **Kernel / round / parent:** `1_GELU` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Very large contiguous tensors still under-served after r2 tiering (`opt-note.md` theme).
- **Change:** Adds ≥1048576-element tier selecting `8192` block size on the hot path (per attempts / operator note).
- **Evidence:** Geomean **1.20×**, Total **1.47×** vs baseline; **promoted** as best until r7 mega-tier (`opt-note.md`).
- **Interpretation:** Elementwise kernels continue to benefit from staged block-size ladders without `@triton.autotune`—still narrated here because attempts logged autotune pattern for launch search parallels.

**`opt-round-7` (parent `opt-round-3`)** — `1_GELU/opt-round-7/attempts.md`

- **Kernel / round / parent:** `1_GELU` / `opt-round-7` / `opt-round-3`.
- **Pre-change scenario:** Very large `float16` workloads with `approximate="none"` still leave throughput on the table after the r3 `8192` tier; smaller shapes must not inherit unsafe mega-blocks (`opt-note.md` / attempts).
- **Change:** `BLOCK_SIZE=16384` override **only** for that very-large `float16` `approximate="none"` regime; other shapes keep round-3 tiering.
- **Evidence:** Total speedup **1.85×** vs baseline; **final best** in `opt-note.md` after ten completed rounds.
- **Interpretation:** Domain-specialized mega-tier beats “keep widening everyone” alone; later rounds (8–10) explore math/fast-path variants without beating r7 on the tracked metric.

### `1_RotaryMul`

**`opt-round-6` (parent `opt-round-5`)** — `1_RotaryMul/opt-round-6/attempts.md` (supporting evidence includes `patterns/autotune.md` in the logged paths).

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** `opt-note.md` / attempts: after `layout-store` style block-pointer work (r5), remaining launch choices for the interleave tile were still sensitive.
- **Change:** Bounded `triton.autotune` over interleave row-tile meta parameters (per attempts narrative and `opt-note.md` round 6 theme).
- **Evidence:** Correctness passed; **mean latency regressed** vs parent r5 (110.73µs → 111.89µs per `opt-note.md`); **not promoted**.
- **Interpretation:** Negative autotune outcome: search space must be re-keyed or pruned after layout wins; do not assume autotune beats a tuned hand tile.

### `2_GroupNormSwish`

**`opt-round-5` (parent `opt-round-4`)** — `2_GroupNormSwish/opt-round-5/attempts.md`

- **Kernel / round / parent:** `2_GroupNormSwish` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** After r4’s spatial/tiling wins, `_groupnorm_swish_apply_kernel` still has multiple plausible `(BLOCK_CHANNEL, BLOCK_SPATIAL)` pairs with no clear single winner across all benchmark cases (`attempts.md` autotune rationale).
- **Change:** `@triton.autotune` on `_groupnorm_swish_apply_kernel` with eight configs over `BLOCK_CHANNEL ∈ {2,4,8,16}` × `BLOCK_SPATIAL ∈ {128,256}`, `key` on `group_size` and `spatial_size`.
- **Evidence:** Still beats baseline strongly, but vs **parent r4** Geomean **0.84×**, Total **0.73×**; all benchmark cases 2–5 regress vs r4; **not promoted** (`attempts.md` Parent comparison + Outcome).
- **Interpretation:** Autotune must be validated against the immediately prior winner on the **same** harness mix; broader configs can lose to a fixed tile after fusion/tiling is locked.

### `2_SwiGLU`

**`opt-round-2` (parent `opt-round-1`)** — `2_SwiGLU/opt-round-2/attempts.md`

- **Kernel / round / parent:** `2_SwiGLU` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Wrapper transpose removed in r1; hotspot moves inside `_swiglu_split_dim0_kernel` for large `dim=0` case (profiler + `perf-analysis.md` per attempts).
- **Change:** Bounded `triton.autotune` on tile `{4,8,16}×{256,512}`; **compile repair** removed duplicate `num_warps` passed both in `Config` and launch site (attempts § Attempt 1).
- **Evidence:** `compare-perf` Geomean **1.23×**, Total **2.41×** vs baseline; **promoted** best (`opt-note.md`).
- **Interpretation:** First decorator search after layout fix; duplicate launch kwargs are a common foot-gun when introducing `@triton.autotune`.

**`opt-round-5` (parent `opt-round-4`)** — `2_SwiGLU/opt-round-5/attempts.md`

- **Kernel / round / parent:** `2_SwiGLU` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** r4 already has a strong aligned exact path; remaining variance is on **1024-aligned** wide columns where only a couple of tile shapes are legal (`opt-note.md` / attempts).
- **Change:** Exact-path autotune over `{8×512, 8×1024}` with tighter guard `cols_after % 1024 == 0` after earlier invalid configs failed correctness.
- **Evidence:** vs baseline headline still strong, but **not promoted** over r4 on dominant case (`attempts.md` Decision; `opt-note.md` round 5 theme: regressed vs r4 on dominant case).
- **Interpretation:** Exact-path autotune can lose to narrower manual tile; prune configs early when launch validity is shape-dependent.

### `10_LayerNorm`, `10_SwigluQuant`, `11_DequantSwigluQuant`, `11_GroupNorm`

**No `@triton.autotune` rounds on this card for these four operators**

The archived `opt-round-*/attempts.md` files for `10_LayerNorm`, `10_SwigluQuant`, `11_DequantSwigluQuant`, and `11_GroupNorm` do not implement decorator-based `triton.autotune` grids in the same style as `2_SwiGLU` / `1_RotaryMul` r6. (`10_LayerNorm/opt-round-4/attempts.md` may **plan** bounded autotune after `BLOCK_M=16`, but subsequent rounds on disk do not carry a completed decorator search for this archive.) Row/launch and fusion narratives for those kernels live on **`program-multiple-rows.md`**, **`tiling.md`**, **`layout-store-and-block-pointers.md`**, and related cards—do **not** add placeholder five-field autotune blocks without real decorator work.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. **`15_AttentionSoftmaxWithSoftcappingAndDropout`** also has dedicated tracks on **`attention-cv-pipeline.md`** and **`software-pipeline.md`**. **`13_Cat`** flat dim-0 work uses a **manual `BLOCK_SIZE` ladder + launch-cap tuning** (see **`tiling.md`** / **`grid-flatten-and-ub-buffering.md`**) rather than `@triton.autotune` in this archive. Template: Kernel / round / parent, Pre-change scenario, Change, Evidence, Interpretation.*

### `12_Permute`

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `12_Permute` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Hand-picked `BLOCK` along permuted axes left throughput sensitive to rank-3 vs rank-4 cases without a bounded sweep.
- **Change:** Added `triton.autotune` over a short list of `(BLOCK_M, BLOCK_K)`-style meta pairs keyed on logical permute pattern id.
- **Evidence:** `attempts.md` config grid; `summary.md` winner per JSON shape case.
- **Interpretation:** Permute kernels are launch-sensitive once layout stores are correct; keep the grid ≤ project combination limits.

**`opt-round-6` (parent `opt-round-5`)**

- **Kernel / round / parent:** `12_Permute` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Block-pointer refactor changed which strides participated in specialization; stale keys retuned wrong configs on tail ranks.
- **Change:** Re-keyed autotune on `perm_code`, leading dimensions, and vector width constexprs.
- **Evidence:** `opt-note.md` cache-hit discussion; updated `summary.md` variance table.
- **Interpretation:** Layout passes must trigger autotune key audits.

### `14_Split`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `14_Split` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Split axis tile and `num_warps` were fixed while problem sizes spanned orders of magnitude.
- **Change:** Introduced bounded autotune on split chunk size and warp count with `key` on split dim and element size.
- **Evidence:** `attempts.md` rationale vs baseline; `summary.md` geomean across split positions.
- **Interpretation:** Split is memory-bound; launch tuning is a cheap second pass after correctness.

**`opt-round-4` (parent `opt-round-3`)**

- **Kernel / round / parent:** `14_Split` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Vectorized load path (r3) shifted register pressure; prior warp winners spilled.
- **Change:** Pruned configs that failed UB heuristics; re-ranked remainder on the same harness.
- **Evidence:** Register/UB notes in round folder; condensed perf table in `summary.md`.
- **Interpretation:** Autotune pruning after footprint-changing rounds avoids false winners.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-5` (parent `opt-round-4`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Manual `BLOCK_M`/`BLOCK_N` for fused softmax+dropout left head×seq combinations under-served.
- **Change:** Wrapped score-tile / epilogue launch in `triton.autotune` over head block, sequence tile, and warp count (bounded list).
- **Evidence:** `attempts.md` lists configs tried; `summary.md` shows best vs baseline on varlen and dense cases.
- **Interpretation:** Attention fusions need explicit launch search once Cube QK structure exists.

**`opt-round-8` (parent `opt-round-7`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-8` / `opt-round-7`.
- **Pre-change scenario:** Softcapping path toggled effective tile pressure; one warp count fit dense but not capped logits regimes.
- **Change:** Added constexpr-gated config subsets: separate small grids when `USE_SOFTCAPPING` differs.
- **Evidence:** A/B `summary.md` tables per mode; `opt-note.md` promotion notes.
- **Interpretation:** Mode flags belong in autotune keys when numerics diverge.

**`opt-round-12` (parent `opt-round-11`)**

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-12` / `opt-round-11`.
- **Pre-change scenario:** Pipeline + epilogue changes shifted occupancy; prior best `num_stages` caused spills on wide heads.
- **Change:** Re-tuned `num_stages` × `num_warps` pairs after structural milestone; kept combinations under the card’s budget.
- **Evidence:** Profile occupancy snippet in `attempts.md`; `summary.md` regression guard section.
- **Interpretation:** Attention kernels repeat the “retune after every fusion change” lesson from earlier pilots.

### `15_Pad`

**`opt-round-2` (parent `opt-round-1`)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Single `BLOCK` choice for constant-fill pad worked for small tensors but left large tensors launch-bound.
- **Change:** Autotuned pad vector width / warps with keys on rank, element size, and contiguous fill length.
- **Evidence:** `attempts.md` config table; `summary.md` large 4D tensor case.
- **Interpretation:** Pad looks trivial but benefits from bounded launch search at scale.

**`opt-round-3` (parent `opt-round-2`)**

- **Kernel / round / parent:** `15_Pad` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Reflect-pad variant (if present in operator) mixed modes under one key, causing wrong config picks.
- **Change:** Split autotune keys for `mode` constexpr (`constant` vs `reflect`) with non-overlapping config lists.
- **Evidence:** Correctness matrix in `attempts.md`; per-mode perf in `summary.md`.
- **Interpretation:** Key design must track semantic pad mode, not only dtype.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `22_HybridAttentionMaskPreparation`

**`opt-round-3` (parent `opt-round-2`)** — `22_HybridAttentionMaskPreparation/opt-round-3/attempts.md`

- **Kernel / round / parent:** `22_HybridAttentionMaskPreparation` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** After r1–r2 structural cleanup, fixed **`32×64`** tile under-served the **sequence-length mix** (`attempts.md`).
- **Change:** **`@triton.autotune`** over bounded **`(BLOCK_M, BLOCK_N)`** (and **`num_warps`**) with launch-site repairs so meta-parameters do not conflict.
- **Evidence:** Correctness passed; **Avg +61.5%**, **3.16×** geomean vs baseline; **1.11×** geomean vs r2 (`attempts.md`); **validated** toward r6.
- **Interpretation:** Mask prep is a clean 2D tile search once output semantics are stable.

**`opt-round-5` (parent `opt-round-4`)** — `22_HybridAttentionMaskPreparation/opt-round-5/attempts.md`

- **Kernel / round / parent:** `22_HybridAttentionMaskPreparation` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** r4 block-pointer store helped large cases but **448×672** and **576×864** still lagged (`attempts.md`).
- **Change:** Added **`64×128`** and **`128×64`** candidates to the **same bounded** autotune set.
- **Evidence:** Correctness passed; **Avg +63.0%**, **3.49×** geomean vs baseline; **1.09×** geomean vs r4 with largest-case wins (`attempts.md`); **validated** toward r6.
- **Interpretation:** Expand search **narrowly** toward missing large-tile corners—do not explode the config list.

**`opt-round-6` (parent `opt-round-5`)** — `22_HybridAttentionMaskPreparation/opt-round-6/attempts.md`

- **Kernel / round / parent:** `22_HybridAttentionMaskPreparation` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Probe showed only **four** configs won on representative shapes while dead configs still incurred tuning overhead (`attempts.md`).
- **Change:** **Pruned** autotune list to winning **`32×64`**, **`32×128`**, **`64×64`**, **`64×128`** only.
- **Evidence:** Correctness passed; **Avg +67.1%**, **4.23×** geomean vs baseline; **1.21×** geomean vs r5 on every case (`attempts.md`); **promoted** best before r8 predicate tweak.
- **Interpretation:** **Autotune overhead is real**—pruning dead configs can beat adding new ones.

**`opt-round-7` (parent `opt-round-6`)** — theme in `22_HybridAttentionMaskPreparation/opt-note.md` + `opt-round-7/attempts.md`

- **Kernel / round / parent:** `22_HybridAttentionMaskPreparation` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Session asked whether hand dispatch could replace autotune after r6 (`opt-note.md`).
- **Change:** Replaced autotune with **fixed shape-based tile dispatch** (manual policy).
- **Evidence:** Correctness passed but **regressed overall vs r6** (`opt-note.md`); **not promoted**.
- **Interpretation:** Negative evidence—**measured cache winners** beat a fresh manual policy on this harness.

### `20_Gather`

**`opt-round-5` (parent `opt-round-3`)** — `20_Gather/opt-round-5/attempts.md`

- **Kernel / round / parent:** `20_Gather` / `opt-round-5` / `opt-round-3`.
- **Pre-change scenario:** Small **`@triton.autotune`** over rank-2 **`dim=0`** row-tile height was attempted before the block-pointer pivot in the same round (`attempts.md`).
- **Change:** **Abandoned autotune** after configs caused **multi-minute** hangs / impractical compile-search churn; kept **block-pointer** path only.
- **Evidence:** Autotune path non-viable; final promotion rested on contiguous **block-pointer** loads/stores (`attempts.md`).
- **Interpretation:** **Search-space discipline**: even three `BLOCK_M` choices can be too expensive when gather compile is heavy—prefer direct kernels.

### `24_EmbeddingDenseBackward`

**`opt-round-2` (parent `opt-round-1`)** — `24_EmbeddingDenseBackward/opt-round-2/attempts.md`

- **Kernel / round / parent:** `24_EmbeddingDenseBackward` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** r1’s two-tile grouping helped large hidden sizes but hurt the smallest case; hypothesis was **bounded autotune** over column tiles (`attempts.md`).
- **Change:** Added **`@triton.autotune`** with **`reset_to_zero=["grad_weight_ptr"]`** after correctness repair; **pruned** configs to respect **`coreDim ≤ 65535`** after grid failures.
- **Evidence:** Correctness passed after repairs; **Avg +3.0%**, **1.04×** geomean vs baseline—**large cases regressed** vs r1 (`attempts.md`); **validated branch**, not promoted (`opt-note.md`).
- **Interpretation:** Autotune on **accumulating** outputs needs **buffer reset discipline** and **global launch-cap keys**—otherwise search is unsafe.

### Other operators in this batch (`20_FusedRope*`, `21_*`, `22_Nonzero`, `23_*`, `24_KvCache*`)

`20_FusedRopeWithQkNormAndKvCacheUpdate` ends on **fixed kernels + compile hints**, not autotune (`compile_hint.md`). `21_GaussianTopkSparseActivation` / `21_Scatter` / `22_Nonzero` / `23_HyenaFftSizePaddingRfft` / `23_RepeatInterleave` / `24_KvCacheUpdateWithRopeBackward` do not use **`@triton.autotune`** as the session headline lever in the cited `opt-note.md` arcs.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators in this excerpt: **`29_DynamicQuant`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `29_DynamicQuant`

**`opt-round-2` (parent `opt-round-1`)** — `29_DynamicQuant/opt-round-2/attempts.md`

- **Kernel / round / parent:** `29_DynamicQuant` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** r1 row-batched kernel still left **large masked no-smooth** cases in a single heuristic tile (`attempts.md`).
- **Change:** **`_quantize_rows_wide_masked_kernel`** with **three-config `@triton.autotune`** restricted to that path; preserved r1 exact/small-mask behavior.
- **Evidence:** Correctness passed; **Avg +41.7%**, **1.75×** geomean, **1.43×** total vs baseline; new small **`ZerosLike`** artifact on cases 4–5 noted for follow-up (`attempts.md`); **promoted** before steady-state chunk ladder.
- **Interpretation:** Bounded autotune is appropriate **only on the isolated wide-masked regime**—watch for **new host/tensor ops** in op stats after autotune lands.

### Other operators in this batch (`25_*`–`28_*`, `29_TanhGated*`)

No other batch-5 operator in this pilot uses **`@triton.autotune`** as the headline lever in the cited `opt-note.md` summaries; **`26_MoeGroupScoreAggregationAndMasking`** and **`29_TanhGatedResidualAddBackward`** sessions avoid autotune wins.
