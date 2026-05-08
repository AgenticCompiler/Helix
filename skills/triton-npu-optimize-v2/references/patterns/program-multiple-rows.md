# Program Multiple Rows Pattern

## Summary

Amortize per-program fixed costs and improve vector-friendly batching for **row-reduction or row-wise fused kernels** by mapping **multiple rows** to one Triton `program_id` via `BLOCK_M > 1`, instead of one row per program.

## Use When

- The kernel is **naturally row-wise**: each output row depends mainly on one row of input (e.g. row-wise LogSumExp, row norms, row softmax statistics).
- Profiling or timeline views suggest **high scalar/control overhead**, **under-filled vector work per program**, or **many tiny programs** relative to problem size `B` (batch / number of rows).
- The row-wise math already uses **tile loops along `N`** (`BLOCK_N`); increasing **`BLOCK_M`** does not force an extra full pass over global memory if you keep a **single streaming pass** over `N` per program.

## Signals

### Code

- `program_id(0)` indexes **rows 1:1** (`pid_m` is the row index), and the inner loop only tiles **`N`**.
- Scalar helpers (`program_id`, pointer arithmetic per row) run once **per row**; vector units see **narrow** tensors (e.g. `(1, BLOCK_N)` loads).

### Profile

- **`aiv_scalar_ratio`** or scalar-related time is **disproportionately high** compared to useful vector math, for workloads where `B` is large enough that vector throughput should dominate.
- **`op_statistic`** (per-kernel): **Avg** latency improves when the same logical work uses **fewer launches** (compare with care: **Count** and input shapes must be comparable across runs).
- If **`aiv_mte2_ratio`** is **not** the sole dominant bucket, pure “double-buffer the loads” may be the wrong first lever; **program batching** can still help by making each program’s inner loop **wider** along rows.
- Frequent **barrier / wait** patterns tied to **many short programs** or **thin** vector blocks.
- **Note:** High **`BAR`** cycle counts alone are **not** a success metric; correlate with **wall time**, **op_statistic Avg**, and correctness.

## Implementation sketch (Triton)

1. Add **`BLOCK_M: tl.constexpr`** and treat **`pid_m`** as a **block of rows**:
   - `rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`
   - `row_mask = rows < B`
2. Build **2D tiles**: `vals` shape **`(BLOCK_M, BLOCK_N)`**; row-wise reductions use **`axis=1`** (max, sum) so running state **`m`, `s`** are **`(BLOCK_M,)`**.
3. Adjust **grid**: `grid = (triton.cdiv(B, BLOCK_M),)`.
4. **Stores**: one scalar per row → `tl.store(y_ptr + rows * stride_ym, x, mask=row_mask)`.
5. **Tune `BLOCK_M`**: start with a small power of two (e.g. 4–16). Too large **`BLOCK_M * BLOCK_N`** may hurt UB/register pressure; validate with benchmark + profile.

### Example reference (row-wise LSE + fused activations)

See the operator workspace pattern: row pointers `row_ptrs = x_ptr + rows[:, None] * stride_xm`, masked load to `(BLOCK_M, BLOCK_N)`, streaming LSE with `tl.max` / `tl.sum` on `axis=1`, then fused elementwise ops and masked store.

## Avoid When

- **Second full pass** over `x` for the same row (e.g. two-pass LSE) usually **increases global reads**; msprof often shows **more MTE / wait** unless the algorithm truly requires it. Prefer **single-pass streaming LSE** when numerically stable.
- **Ping-pong / multibuffer** without evidence of **MTE–vector overlap** can add **sync and UB** cost; treat as a **separate hypothesis** to validate.
- Do not conclude from **one** metric (e.g. `BAR` cycles) without **end-to-end** timing and comparable workload.

## What To Verify After Applying

1. **Correctness**: same dtypes, masks for `rows >= B`, and numerically stable reductions (e.g. LSE max-shift) unchanged in meaning.
2. **Benchmark**: compare **mean / geomean** with the same harness; use project **`compare-perf`** flow when available—avoid hand-computed speedups from raw logs.
3. **Profiler**: compare **`op_statistic` Avg** for the same op; note **Count** and tensor shapes. Optionally re-check **`op_summary`** vector/scalar/MTE mix.
4. **Sanity**: if `B` is tiny, **`BLOCK_M > 1`** may help little; if `B` is huge, launching **`cdiv(B, BLOCK_M)`** programs should visibly reduce launch/program overhead.

## Related Patterns

- Complements **`parallel`**: `BLOCK_M` widens work **within** one program; `tl.parallel` splits **independent** subgraphs across vector cores—orthogonal when dependencies allow.
- Differs from **`software-pipeline`**: multibuffer targets **load/compute overlap** along a tile loop; **`program-multiple-rows`** targets **program granularity** and **row batching**.

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `1_RotaryMul`
- `10_LayerNorm`
- `10_SwigluQuant`
- `11_DequantSwigluQuant`
- `11_GroupNorm`
- `12_KvRmsnormRopeCache`
- `13_Cat`
- `13_InterleaveRope`
- `14_AdaptiveInstanceNormalization2DBackward`
- `14_Split`
- `15_AttentionSoftmaxWithSoftcappingAndDropout`
- `15_Pad`
- `16_Repeat`
- `17_EmbeddingWithInitialLayernormBackward`
- `18_FusedAddRmsnorm`
- `18_Index`
- `19_FusedResidualRmsNormBackward`
- `21_GaussianTopkSparseActivation`
- `23_HyenaFftSizePaddingRfft`
- `23_RepeatInterleave`
- `24_KvCacheUpdateWithRopeBackward`
- `20_FusedRopeWithQkNormAndKvCacheUpdate`
- `24_EmbeddingDenseBackward`
- `25_NLLLoss`
- `29_DynamicQuant`
- `26_MoeGroupScoreAggregationAndMasking`
- `27_MaxPool3d`
- `27_MultiMaskAttentionAggregation`
- `16_Batched2DRopePositionEncodingBackward`
- `16_Repeat`
- `17_AdamW`
- `17_EmbeddingWithInitialLayernormBackward`
- `18_FusedAddRmsnorm`
- `18_Index`
- `19_FusedResidualRmsNormBackward`

## NPUKernelBench round narratives (pilot: eight kernels, 2026-05-08, log-backed)

*Sources: `workspace/NPUKernelBench_level_1_2_triton/.../opt-round-*/attempts.md` and `opt-note.md` (gitignored `workspace/`—use `find` / absolute reads). Every round below uses the mandatory five-field template from `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `1_RotaryMul`

**`opt-round-1` (parent `baseline`)** — `1_RotaryMul/opt-round-1/attempts.md`

- **Kernel / round / parent:** `1_RotaryMul` / `opt-round-1` / baseline (`baseline/baseline_triton_1_RotaryMul.py`).
- **Pre-change scenario:** Flat `total_pairs` grid with per-element `row = offsets // half_dim`, `pair = offsets % half_dim` scalar decode; high launch and indexing overhead on row-rich workloads.
- **Change:** Row-blocked launch over `BLOCK_ROWS × HALF_DIM` tiles with `HALF_DIM` / `FULL_DIM` as `tl.constexpr`; `BLOCK_ROWS = 256 // half_dim` so each program covers a band of rows before handling interleaved pairs.
- **Evidence:** `compare-perf` vs baseline **Avg +98.9%**; dominant case **103484µs → 954µs** in `attempts.md`; `opt-note.md` marks round 1 as current best after correctness + bench pass.
- **Interpretation:** Canonical `program-multiple-rows` win—batch logical rows per program before smaller layout/autotune refinements.

### `2_SwiGLU`

**`opt-round-6` (parent `opt-round-4`)** — `2_SwiGLU/opt-round-6/attempts.md`

- **Kernel / round / parent:** `2_SwiGLU` / `opt-round-6` / `opt-round-4` (`opt-round-4/opt_triton_2_SwiGLU.py` parent candidate in attempts header).
- **Pre-change scenario:** Dim-0 and aligned fast paths already improved case 5 (`opt-note.md` rounds 1–4), but cases 2–4 still used a **one-row-per-program** last-dimension kernel while benchmark JSON shows row-rich last-dim shapes.
- **Change:** Added `_swiglu_last_dim_grouped_kernel` with bounded autotune over row/column tiles; `_launch_last_dim_kernel` routes to it when `rows >= 16` and `out_cols <= 256`, keeping dim-0 kernels unchanged.
- **Evidence:** `run-test` + `compare-result` passed; vs baseline `compare-perf` case deltas `+7.34%`, `-69.28%`, `-23.15%`, `-70.45%`, `-84.59%`, headline **Avg +48.0%**, **Geomean 2.44×**, **Total 5.63×**; vs parent r4 case 5 delta **+1.23%** with large wins on cases 2–4; **promoted** as session best per `opt-note.md` / attempts Decision.
- **Interpretation:** After non-last-dim structure is fixed, last-dim row grouping is the next structural lever; small regression on the already-fast dominant case can be acceptable when the harness is multi-case.

### `10_LayerNorm`

**`opt-round-2` (parent `opt-round-1`)** — `10_LayerNorm/opt-round-2/attempts.md`

- **Kernel / round / parent:** `10_LayerNorm` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 fused kernel still launched **one row per program** while benchmark invocations use hundreds–thousands of rows (`attempts.md` hypothesis).
- **Change:** `BLOCK_M=4` row batching inside `_layernorm_fused_kernel`; body rewritten for `(BLOCK_M, BLOCK_STATS)` and `(BLOCK_M, BLOCK_APPLY)` tiles while preserving reduction-width safety from r1.
- **Evidence:** Correctness passed; vs baseline Geomean **4.00×**, Total **4.54×**; vs r1 Geomean **3.78×** (`attempts.md` perf table); **promoted** per `opt-note.md`.
- **Interpretation:** Direct application of this card after a fusion-first round—row batching is the obvious second move without waiting for new profiles.

**`opt-round-3` (parent `opt-round-2`)** — `10_LayerNorm/opt-note.md` + `opt-round-3/summary.md`

- **Kernel / round / parent:** `10_LayerNorm` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** `BLOCK_M=4` already wins; remaining headroom on wide-row cases from widening rows per program (`opt-note.md` theme).
- **Change:** Widen fused row batch from **4 → 8** rows per program (`_ROWS_PER_PROGRAM` / equivalent control in the fused kernel per round theme).
- **Evidence:** Correctness passed; `opt-note.md` reports **+81.1%** avg vs baseline and **5.63×** geomean speedup; **promoted** as best over r2.
- **Interpretation:** Diminishing returns appear later (r5+), but one step of row widening after r2 is still a clear net win on this harness.

**`opt-round-4` (parent `opt-round-3`)** — `10_LayerNorm/opt-note.md` + `opt-round-4/summary.md`

- **Kernel / round / parent:** `10_LayerNorm` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Row batch of 8 is strong; large-row cases still amortize poorly vs an even wider batch (`opt-note.md` theme).
- **Change:** Widen fused row batch from **8 → 16** rows per program on the same fused kernel structure.
- **Evidence:** Correctness passed; `opt-note.md` reports **+85.1%** avg vs baseline and **7.98×** geomean speedup; **promoted** as best over r3.
- **Interpretation:** Validates repeated widening until later rounds (aligned paths, UB, scalar traps) take over as the dominant risk.

### `10_SwigluQuant`

**`opt-round-3` (parent `opt-round-2`)** — `10_SwigluQuant/opt-round-3/attempts.md`

- **Kernel / round / parent:** `10_SwigluQuant` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** SwiGLU moved into Triton in r2 but `_swiglu_kernel` still behaved row-wise enough that case 4–5 latency dominated traces cited in attempts.
- **Change:** Row-batch `BLOCK_M` in `_swiglu_kernel` (`2` when very wide else `4`) so each program covers multiple rows on the direct SwiGLU path.
- **Evidence:** Correctness passed; attempts cite `_swiglu_kernel` case4 **44.6→21.4µs**, case5 **230→180µs**; headline **+15.2%** avg, **1.20×** geomean, **1.38×** total vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Same row-batch playbook as LayerNorm/SwiGLU family kernels once math is already in Triton.

**`opt-round-4` (parent `opt-round-3`)** — `10_SwigluQuant/opt-note.md` + `opt-round-4/attempts.md`

- **Kernel / round / parent:** `10_SwigluQuant` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Row-batch SwiGLU wins; dynamic quantization kernel still row-poor on the same suite (`opt-note.md` theme: batch rows in dynamic quant kernel).
- **Change:** Extend multi-row batching into the **dynamic quantization** kernel (same workload family as r3, different launcher path).
- **Evidence:** Correctness passed; `opt-note.md` case-2..5 improvements (e.g. case-2 **-40.15%**, case-4 **-41.77%**); **1.47×** geomean, **1.40×** total vs baseline; **promoted** over r3.
- **Interpretation:** Row batching must follow **all** hot launchers in a fused pipeline, not only the first kernel touched.

**`opt-round-5` (parent `opt-round-4`)** — `10_SwigluQuant/opt-note.md` + `opt-round-5/attempts.md`

- **Kernel / round / parent:** `10_SwigluQuant` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Dynamic path batched in r4; static **int4 quantize-and-pack** path still left case-5 sensitive (`opt-note.md`: batch rows in static int4 kernel).
- **Change:** Row-batch the static int4 quantize-and-pack kernel to match suite shapes.
- **Evidence:** Correctness passed; strong dynamic-case wins retained but case-5 slipped to **-26.67%**, geomean **1.45×**, total **1.37×** vs baseline; `opt-note.md` marks **validated branch** (not best vs r4 on all headline metrics).
- **Interpretation:** PMR-style batching can trade one case for others when the static path gains extra register/host pressure—validate against parent, not only baseline.

### `11_DequantSwigluQuant`

**`opt-round-1` (parent `baseline`)** — `11_DequantSwigluQuant/opt-round-1/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline `perf.txt` shows many small wrapper ops around a single Triton multiply; benchmark cases are `int32`, `quant_mode=1`, single-group dynamic (`attempts.md`).
- **Change:** Single-group dynamic fast path fusing dequant, SwiGLU, optional smooth-scale multiply, per-row scale/max work, and int8 quantization in Triton; generic fallback preserved.
- **Evidence:** `run-test` + `compare-result` passed; `compare-perf` Geomean **1.07×**, Total **1.09×** vs baseline (`attempts.md`); `opt-note.md` lists round 1 as **validated branch** with the same headline metrics.
- **Interpretation:** Row-batched fusion is the structural match for this card; modest geomean can still be the right first promoted trunk when wrappers dominated.

**`opt-round-4` (parent `opt-round-3`)** — `11_DequantSwigluQuant/opt-note.md` + `opt-round-4/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** Medium row-batch tuning on the fused dynamic kernel (r3) still left a separate fallback multiply kernel row-poor on large cases (`opt-note.md` theme).
- **Change:** Batch the **fallback multiply** kernel across multiple rows so launch count drops on large matrices.
- **Evidence:** Correctness passed; `opt-note.md` reports case-2 **-42.28%**, case-5 **-44.61%**, **+27.4%** avg, **1.40×** geomean, **1.47×** total vs baseline; **validated branch** (session trunk advanced in later rounds).
- **Interpretation:** Any remaining PyTorch-side or fallback kernel with one-row launches should be checked for the same PMR move as the fused path.

**`opt-round-7` (parent `opt-round-6`)** — `11_DequantSwigluQuant/opt-note.md` + `opt-round-7/attempts.md`

- **Kernel / round / parent:** `11_DequantSwigluQuant` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** r6 already widened wide-row batch on fallback; fused dynamic path still processed columns too narrowly for largest shapes (`opt-note.md` theme: widen fused path across column tiles, simplify kernel-side quantization).
- **Change:** Widen fused dynamic work across column tiles and simplify quantization bookkeeping inside the fused kernel (per round theme and attempts Decision).
- **Evidence:** Correctness passed; `opt-note.md` **1.84×** geomean, **2.07×** total vs baseline; **promoted** as best at that stage of the session.
- **Interpretation:** Row batching plus **column blocking** inside the same fused kernel is still a PMR-family story: fewer programs and more amortized control on 2D tiles.

### `11_GroupNorm`

**`opt-round-11` (parent `opt-round-10`)** — `11_GroupNorm/opt-note.md` + `opt-round-11/attempts.md`

- **Kernel / round / parent:** `11_GroupNorm` / `opt-round-11` / `opt-round-10`.
- **Pre-change scenario:** Through r10, generic moments path improved but still launched thin row programs on large spatial grids (`opt-note.md` theme).
- **Change:** Batch **multiple rows** inside the generic **moments** kernel (stats pass) while preserving r10’s fused stats+reduction structure.
- **Evidence:** Correctness passed; `opt-note.md` **+92.4%** avg, **15.74×** geomean, **36.42×** total vs baseline; **promoted** over r10.
- **Interpretation:** GroupNorm’s moments kernel is a direct PMR target once channel/spatial tiling is stable.

**`opt-round-15` (parent `opt-round-12`)** — `11_GroupNorm/opt-note.md` + `opt-round-15/attempts.md`

- **Kernel / round / parent:** `11_GroupNorm` / `opt-round-15` / `opt-round-12`.
- **Pre-change scenario:** r12 batched spatial tiles in the **apply** kernel; stats kernel still left headroom on multi-row reuse (`opt-note.md` theme).
- **Change:** Batch multiple rows inside the **stats** kernel (distinct from r11’s generic moments batching—here building on the r12 apply-side trunk).
- **Evidence:** Correctness passed; `opt-note.md` **+93.2%** avg, **18.85×** geomean, **50.35×** total vs baseline; **promoted** as best over r12.
- **Interpretation:** Apply-side and stats-side PMR passes may alternate; both need explicit row widening passes.

**`opt-round-18` (parent `opt-round-16`)** — `11_GroupNorm/opt-note.md` + `opt-round-18/attempts.md`

- **Kernel / round / parent:** `11_GroupNorm` / `opt-round-18` / `opt-round-16`.
- **Pre-change scenario:** r16 is session trunk; experiment widens generic multi-row batch again to chase case-2 (`opt-note.md` theme).
- **Change:** Increase generic multi-row `BLOCK_M` **4→8** when `rows>=16`, mirrored to stats launch configuration (per attempts narrative).
- **Evidence:** Correctness passed; case 2 improved but geomean **did not beat r16**; `opt-note.md` **validated branch, not promoted**.
- **Interpretation:** PMR widening is not monotonic—register/L2 pressure from wider stats+apply coupling can erase gains; parent comparisons matter more than baseline-only headlines.

### `1_GELU`, `2_GroupNormSwish`

No rounds in the archived `attempts.md` headers for these two operators cite **`program-multiple-rows.md`** or describe an equivalent structural “multiple logical rows per program” rewrite in the same sense as the kernels above (`1_GELU` is launch-tier / elementwise work narrated on **`autotune.md`**; `2_GroupNormSwish` early fusion of stats into apply is narrated on **`loop-invariant-hoisting.md`**). Do not fabricate five-field PMR entries here.

## NPUKernelBench round narratives (pilot: eight kernels `12_*`–`15_*`, 2026-05-08, log-backed)

*Operators: **`12_KvRmsnormRopeCache`**, **`12_Permute`**, **`13_Cat`**, **`13_InterleaveRope`**, **`14_AdaptiveInstanceNormalization2DBackward`**, **`14_Split`**, **`15_AttentionSoftmaxWithSoftcappingAndDropout`**, **`15_Pad`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/<Operator>/` (gitignored—use `find` or absolute reads per `skills/triton-npu-kernel-bench-logs/SKILL.md`). Every **round** entry uses the mandatory five-field template.*

### `12_KvRmsnormRopeCache`

**`opt-round-4` (parent `opt-round-3`)** — `workspace/.../12_KvRmsnormRopeCache/opt-round-4/attempts.md`

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-4` / `opt-round-3`.
- **Pre-change scenario:** One cache row (or head slice) per program produced thin tiles and high launch counts on long KV lengths.
- **Change:** Batched multiple logical rows / head bands per `program_id` with `(BLOCK_ROWS, BLOCK_HIDDEN)` masks.
- **Evidence:** `attempts.md` grid change; `summary.md` op_statistic Count drop vs baseline; `opt-note.md` promotion arc.
- **Interpretation:** KV+RMS+rope fusion is row-structured; widening rows matches this card’s core move.

**`opt-round-5` (parent `opt-round-4`)** — `12_KvRmsnormRopeCache/opt-round-5/attempts.md`

- **Kernel / round / parent:** `12_KvRmsnormRopeCache` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Widened rows increased per-program work but left inner reductions axis-1 sized too narrow without paired `BLOCK_N`.
- **Change:** Co-tuned `BLOCK_M` with inner `BLOCK_N` so each program still performs a single streaming pass per batched row band.
- **Evidence:** `opt-note.md` “no second full pass” assertion; benchmark parity on varlen shapes in `summary.md`.
- **Interpretation:** Validates the card’s guardrail against extra global traversals.

### `12_Permute`

**No PMR-primary rounds for this operator in batch 2**

Archived `13`-batch work on **`12_Permute`** improves rank-3/4 specialization, **block-pointer transpose** (`opt-round-3`/`opt-round-7` themes in `opt-note.md`), and **`autotune.md`** re-keying (`opt-round-6`). None of the recorded `attempts.md` headers for this operator cite **`program-multiple-rows.md`** as the primary lever for “multiple logical rows per program.” Map `12_Permute` on **`layout-store-and-block-pointers.md`** and **`autotune.md`** instead of inventing PMR five-field entries here.

### `13_Cat`

**`opt-round-2` (parent `opt-round-1`)** — `13_Cat/opt-round-2/attempts.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Generic `_cat_copy_kernel` still decoded **four** full-rank coordinates per element; round-1 perf showed case-3 at **186.01µs** on that path while dim-0 flat copy was already fast (`attempts.md`).
- **Change:** Replaced generic per-element decode with **`_cat_row_copy_kernel`**: copy **outer × concat** rows as 2D slices and compute output row placement **once per row tile** instead of per element.
- **Evidence:** Correctness passed; `compare-perf` **+94.1%** avg, **19.21×** geomean, **36.11×** total vs baseline (`attempts.md`); **promoted** best over r1 (`opt-note.md`).
- **Interpretation:** Concat’s generic path is a row-structured PMR target—amortize coordinate decode across row bands.

**`opt-round-7` (parent `opt-round-6`)** — `13_Cat/opt-round-7/attempts.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-7` / `opt-round-6`.
- **Pre-change scenario:** Profiler on case-3 `_cat_row_copy_kernel` showed **scalar-heavy** behavior with **Block Dim 448** and tiny task time—thin programs on the row-copy path (`perf-analysis.md` cited in attempts).
- **Change:** Increased **`BLOCK_ROWS` from 4 to 8** while keeping `BLOCK_COLS` unchanged on `_cat_row_copy_kernel`.
- **Evidence:** Correctness passed; `compare-perf` **+94.5%** avg, **21.26×** geomean, **39.70×** total vs baseline (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** Same PMR widening playbook as pad/SwiGLU once profiling proves one-row-per-program overhead.

**`opt-round-11` (parent `opt-round-10`)** — `13_Cat/opt-round-11/attempts.md`

- **Kernel / round / parent:** `13_Cat` / `opt-round-11` / `opt-round-10`.
- **Pre-change scenario:** Round-9 validated **`BLOCK_ROWS=32`** on the row kernel; round-10 widened the **flat** dim-0 copy tile independently—both paths needed integration without cross-talk (`attempts.md` hypothesis).
- **Change:** Merged validated **row-path `BLOCK_ROWS=32`** batching into the **round-10 flat-path** winner so two disjoint optimizations compose.
- **Evidence:** Correctness passed; `compare-perf` **+95.3%** avg, **28.40×** geomean, **72.98×** total vs baseline (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** PMR on row concat and transfer tiling on flat concat are orthogonal until dispatch integrates them—integration is its own gated round.

### `13_InterleaveRope`

**`opt-round-2` (parent `opt-round-1`)** — `13_InterleaveRope/opt-round-2/attempts.md`

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Each program handled a single interleaved pair index; scalar overhead dominated short ropes.
- **Change:** Mapped `BLOCK_PAIR` interleave units per program with vectorized cos/sin application.
- **Evidence:** `attempts.md` pair indexing math; `summary.md` short-sequence cases; `opt-note.md` promotion.
- **Interpretation:** Interleave rope is naturally batchable along pair index.

**`opt-round-3` (parent `opt-round-2`)** — `13_InterleaveRope/opt-round-3/attempts.md`

- **Kernel / round / parent:** `13_InterleaveRope` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** After widening, tail pairs left partially empty programs.
- **Change:** Masked tail handling without shrinking `BLOCK_PAIR` on interior programs; host grid uses `cdiv`.
- **Evidence:** Correctness cases in `attempts.md`; stable perf in `summary.md`.
- **Interpretation:** Same tail discipline as other `BLOCK_M` wideners.

### `14_AdaptiveInstanceNormalization2DBackward`

**`opt-round-3` (parent `opt-round-1`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-3/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-3` / `opt-round-1`.
- **Pre-change scenario:** Round-1 row kernel and round-2 linear decode still launched **32 programs** on tiny spatial case `[32,4,2,2]`; per-element row decode remained expensive (`attempts.md`; pattern reference **`program-multiple-rows`**).
- **Change:** Small-spatial dispatch: keep row-wise structure but use **`BLOCK_M=32`**, **`BLOCK_N=8`** when `spatial_size <= 64`.
- **Evidence:** Correctness passed; still below baseline on headline mix but **best branch vs r1/r2** with `Avg -26.9%`, **Geomean 0.79×**, **Total 0.76×** vs baseline (`attempts.md`); **promoted** as best validated Triton branch per Decision.
- **Interpretation:** Tiny spatial + many rows is a PMR-first regime before spatial widening pays.

**`opt-round-6` (parent `opt-round-5`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-6/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Round-5 large-spatial `BLOCK_N=512` win left **medium** spatial cases (128–2048) on default launch shape, dragging geomean (`attempts.md`; pattern **`program-multiple-rows`**).
- **Change:** Medium-spatial branch with **`BLOCK_M=16`**, **`BLOCK_N=256`** for `128 <= spatial_size < 2048`.
- **Evidence:** Correctness passed; headline **+3.9%** avg, **1.10×** geomean, **1.60×** total vs baseline; parent vs r5 improves cases 2–3 (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** Separate **launch geometry per spatial regime** is still PMR—row programs must match each regime’s parallelism.

**`opt-round-10` (parent `opt-round-6`)** — `14_AdaptiveInstanceNormalization2DBackward/opt-round-10/attempts.md`

- **Kernel / round / parent:** `14_AdaptiveInstanceNormalization2DBackward` / `opt-round-10` / `opt-round-6`.
- **Pre-change scenario:** Round-9 full streaming rewrite helped only the largest case and regressed others; launch collapse needs **high row count** to amortize (`attempts.md`; pattern **`program-multiple-rows`**).
- **Change:** Keep round-6 **2D** kernel as default; route **`spatial_size >= 2048` and `n_rows >= 128`** only to a **streaming row-batch** kernel reusing r9’s inner `BLOCK_N` loop.
- **Evidence:** Correctness passed; **+6.6%** avg, **1.20×** geomean, **2.00×** total vs baseline; strong case-5 win vs r6 with bounded regressions on other cases (`attempts.md`); **promoted** (`opt-note.md`).
- **Interpretation:** **Workload-gated** alternate program geometry—PMR card covers the gating discipline; see also **`tiling.md`** for inner spatial streaming.

### `14_Split`

**`opt-round-1` (parent `baseline`)** — `14_Split/opt-round-1/attempts.md`

- **Kernel / round / parent:** `14_Split` / `opt-round-1` / baseline.
- **Pre-change scenario:** Split dimension iterated with one output row per program along the slow split axis.
- **Change:** Batched consecutive split chunks per program when chunks are contiguous in output layout.
- **Evidence:** `attempts.md` contiguous chunk predicate; `summary.md` wide-split-axis tensors.
- **Interpretation:** Split is not strictly row-reduction, but chunk batching still reduces launch overhead when layout allows.

**`opt-round-3` (parent `opt-round-2`)** — `14_Split/opt-round-3/attempts.md`

- **Kernel / round / parent:** `14_Split` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Vectorized loads (r2) widened inner tile without widening program batching along split axis.
- **Change:** Increased `BLOCK_SPLIT` so more elements emit per program along the split axis.
- **Evidence:** Scalar-ratio improvement in profiler notes; `summary.md` delta vs r2.
- **Interpretation:** Vector width and program batching are orthogonal levers; apply both deliberately.

### `15_AttentionSoftmaxWithSoftcappingAndDropout`

**`opt-round-2` (parent `opt-round-1`)** — `15_AttentionSoftmaxWithSoftcappingAndDropout/opt-round-2/attempts.md`

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** One head per program left softmax statistics kernels launch-heavy on many-head models.
- **Change:** Batched multiple heads per program for vector-side softmax/dropout while keeping QK tiles consistent with Cube schedule.
- **Evidence:** `attempts.md` head-band masks; `summary.md` multi-head benchmark.
- **Interpretation:** Vector-heavy attention epilogues benefit from `BLOCK_HEAD` style batching once Cube path is stable.

**`opt-round-6` (parent `opt-round-5`)** — `15_AttentionSoftmaxWithSoftcappingAndDropout/opt-round-6/attempts.md`

- **Kernel / round / parent:** `15_AttentionSoftmaxWithSoftcappingAndDropout` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Sequence tiling alone under-filled programs on short seqlen but wide batch.
- **Change:** Combined batch and head widening so each program carries meaningful vector work at small `T`.
- **Evidence:** Short-seqlen case table in `summary.md`; profiler program duration notes in `attempts.md`.
- **Interpretation:** Attention launch shaping must co-optimize batch, head, and seq axes.

### `15_Pad`

**`opt-round-2` (parent `opt-round-1`)** — `15_Pad/opt-round-2/attempts.md`

- **Kernel / round / parent:** `15_Pad` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 row-major rewrite sped large cases but **regressed case-2**; manual `msprof` on case-2 showed **`aiv_scalar_ratio ≈ 0.498`** and **Block Dim 4608**—one program per **9-column** output row (`attempts.md` profile narrative).
- **Change:** Generalized `_pad_constant_kernel` to **`BLOCK_ROWS > 1`** so short rows batch into fewer programs while preserving the wide-row path from r1.
- **Evidence:** Correctness passed; case-2 **120.936→49.141µs** in round summary path; **+28.5%** avg, **1.50×** geomean vs baseline (`opt-note.md`); **validated branch** then superseded by later rounds.
- **Interpretation:** Pad is row-major in this harness; PMR fixes thin-row scalar traps before widening column tiles.

## NPUKernelBench round narratives (pilot: eight kernels `16_*`–`19_*`, 2026-05-08, log-backed)

*Operators: **`16_Batched2DRopePositionEncodingBackward`**, **`16_Repeat`**, **`17_AdamW`**, **`17_EmbeddingWithInitialLayernormBackward`**, **`18_FusedAddRmsnorm`**, **`18_Index`**, **`19_FusedResidualRmsNormBackward`**, **`19_IndexPut`**. Sources: `workspace/NPUKernelBench_level_1_2_triton/<Operator>/opt-note.md` and `opt-round-*/attempts.md` (gitignored `workspace/`—discover with `find` or absolute reads). Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `16_Batched2DRopePositionEncodingBackward`

**`opt-round-1` (parent `baseline`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-1/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-1` / baseline.
- **Pre-change scenario:** Flat elementwise backward kernel paid high launch/program overhead; a sin/cos fusion attempt failed balanced numerics and was **rejected** (`attempts.md` pivot).
- **Change:** Pivot kept baseline math but let each program walk **multiple contiguous sub-blocks** with `tl.static_range` so contiguous angle tensors amortize launch work.
- **Evidence:** Correctness passed; `compare-perf` **+36.8%** avg in attempts transcript; `opt-note.md` reports strong case 2–5 wins and **2.51×** total speedup vs baseline; **promoted** best.
- **Interpretation:** Even “pure elementwise” rope work benefits from **more contiguous work per program** before tile-shape ladders.

**`opt-round-4` (parent `opt-round-1`)** — `16_Batched2DRopePositionEncodingBackward/opt-round-4/attempts.md`

- **Kernel / round / parent:** `16_Batched2DRopePositionEncodingBackward` / `opt-round-4` / `opt-round-1`.
- **Pre-change scenario:** Single geometry from r1 helped large tensors but **hurt tiny** cases (`opt-note.md` rounds 2–3 validated branches).
- **Change:** **Size-based dispatch** for blocks-per-program / tile template so tiny vs non-tiny tensors use different static kernels.
- **Evidence:** Correctness passed; **1.83×** geomean vs baseline (`opt-note.md`); **promoted** new best over r1.
- **Interpretation:** PMR here is **regime dispatch**, not a single global `BLOCK_M`.

### `16_Repeat`

**`opt-round-9` (parent `opt-round-8`)** — `16_Repeat/opt-round-9/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Exact full-tile last-dim path and wide-row `REP3>1` optimizations existed separately (`opt-note.md` arc through r8).
- **Change:** **Merged** full-tile routing with **row batching** on the wide-row last-dimension path so both mechanisms apply on the same branch where safe.
- **Evidence:** Correctness passed; **+91.7%** avg, **57.72×** geomean, **245.00×** total vs baseline (`opt-note.md`); **promoted** over r8 on geomean.
- **Interpretation:** Repeat is a **two-lever** problem (fulltile + row bands); compose only after each lever is validated in isolation.

**`opt-round-15` (parent `opt-round-14`)** — `16_Repeat/opt-round-15/attempts.md`

- **Kernel / round / parent:** `16_Repeat` / `opt-round-15` / `opt-round-14`.
- **Pre-change scenario:** Width-224 exact fulltile branch shipped in r14 still left inner-row launch slack on scored cases (`opt-note.md` round 15 theme).
- **Change:** **Increased row batching inside** the width-224 fulltile micro-branch while keeping exact-divisibility predicates.
- **Evidence:** Correctness passed; **+92.0%** avg, **61.16×** geomean, **246.36×** total vs baseline—**session best** (`opt-note.md` final summary).
- **Interpretation:** After scalar/layout wins, incremental `BLOCK_ROWS` tuning inside a proven exact path is still PMR.

### `17_AdamW`

**`opt-round-3` (parent `opt-round-2`)** — `17_AdamW/opt-round-3/attempts.md`

- **Kernel / round / parent:** `17_AdamW` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** 1D fused update path still issued **one contiguous chunk per program** on the transfer-heavy vector regime (`opt-note.md` round 3 theme).
- **Change:** Process **two contiguous chunks per program** on that path while preserving host-precomputed invariants from earlier rounds.
- **Evidence:** Correctness passed; **+2.0%** avg vs baseline (`opt-note.md`); **validated branch** in the long AdamW arc.
- **Interpretation:** Memory-bound optimizers should treat “chunk pairs per program” as PMR-like widening when rows are not the natural unit.

### `17_EmbeddingWithInitialLayernormBackward`

**`opt-round-8` (parent `opt-round-6`)** — `17_EmbeddingWithInitialLayernormBackward/opt-round-8/attempts.md`

- **Kernel / round / parent:** `17_EmbeddingWithInitialLayernormBackward` / `opt-round-8` / `opt-round-6`.
- **Pre-change scenario:** Fixed **4096** hidden fast path still used **one row per program** despite long row counts (`opt-note.md` / attempts).
- **Change:** **`BLOCK_M=2`** on the fixed-width branch so **`norm_weight`** loads amortize across two embedding rows.
- **Evidence:** Correctness passed; average kernel latency **≈ baseline** (**−0.1%** per `opt-note.md`); **validated** bridge toward r9.
- **Interpretation:** Embedding+LN backward is row-structured; PMR applies even when hidden is constexpr-specialized.

**`opt-round-20` (parent `opt-round-19`)** — `17_EmbeddingWithInitialLayernormBackward/opt-round-20/attempts.md`

- **Kernel / round / parent:** `17_EmbeddingWithInitialLayernormBackward` / `opt-round-20` / `opt-round-19`.
- **Pre-change scenario:** Three-regime dispatch (small/medium vs **largest-only partial-sum** kernel) still needed more row-level work on the largest slice alone (`opt-note.md` round 20 theme).
- **Change:** Raised largest-only partial-sum path from **`BLOCK_M=2` to `BLOCK_M=3`** under the strict largest-only gate.
- **Evidence:** Correctness passed; **+3.2%** avg, **1.03×** geomean, **1.07×** total vs baseline (`opt-note.md`); **final best** after 20 rounds.
- **Interpretation:** Partial-sum offload + PMR on the extreme tail can be the session closer once medium rows are protected by thresholds.

### `18_FusedAddRmsnorm`

**`opt-round-3` (parent `opt-round-2`)** — `18_FusedAddRmsnorm/opt-round-3/attempts.md`

- **Kernel / round / parent:** `18_FusedAddRmsnorm` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** After inverse-RMS was collapsed into the apply kernel (r1–r2), the streaming path still launched **one row per program** on large `B` (`opt-note.md`).
- **Change:** **`BLOCK_M=4`** rows per fused streaming program.
- **Evidence:** Correctness passed; **+63.4%** avg, **2.75×** geomean vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Fused RMS+add is a row-wise kernel; widen rows after structural fusion stabilizes numerics.

**`opt-round-5` (parent `opt-round-3`)** — `18_FusedAddRmsnorm/opt-round-5/attempts.md`

- **Kernel / round / parent:** `18_FusedAddRmsnorm` / `opt-round-5` / `opt-round-3`.
- **Pre-change scenario:** `BLOCK_M=8` from a sibling experiment regressed mid cases (`opt-note.md` round 4 validated branch); need **gated** widening.
- **Change:** Apply **`BLOCK_M=8` only when `rows >= 128`**, keep `BLOCK_M=4` otherwise on the same streaming kernel template.
- **Evidence:** Correctness passed; **+64.0%** avg, **2.80×** geomean vs baseline (`opt-note.md`); **promoted** over r3.
- **Interpretation:** Row batching must be **threshold-gated** when UB/register pressure flips mid-suite cases.

### `18_Index`

**`opt-round-6` (parent `opt-round-3`)** — `18_Index/opt-round-6/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-6` / `opt-round-3`.
- **Pre-change scenario:** Round-3 winner used efficient row-copy launches; hypothesis tested **multiple selected rows per program** (`opt-note.md` round 6 theme).
- **Change:** Batched multiple gathered rows per program on the fast path (per attempts plan).
- **Evidence:** Correctness passed; geomean fell to **7.79×** vs baseline (`opt-note.md`); **not promoted**.
- **Interpretation:** Negative PMR evidence—index_select-style gathers can become **more scalar-bound** when batching complicates addressing.

### `19_FusedResidualRmsNormBackward`

**`opt-round-10` (parent `opt-round-7`)** — `19_FusedResidualRmsNormBackward/opt-round-10/attempts.md`

- **Kernel / round / parent:** `19_FusedResidualRmsNormBackward` / `opt-round-10` / `opt-round-7`.
- **Pre-change scenario:** Scored **`_fused_residual_rmsnorm_backward_fused_kernel`** still spent ~half of the large-case time on the **4096**-wide fused path with **one row per program** (`attempts.md` cites `program-multiple-rows.md` + r9 profiler).
- **Change:** Selector returns **`BLOCK_M=2`** for the **`hidden_size==4096`** fused fast path when **`rows >= 4096`**, preserving smaller-width selectors from r7.
- **Evidence:** Correctness passed; **+58.9%** avg, **2.48×** geomean, **2.44×** total vs baseline; **+5.3%** avg vs parent r7 (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** Large-row fused RMS backward matches this card’s “wide hidden + many rows” sweet spot.

### `19_IndexPut`

**No PMR-primary narrative for this operator**

The `19_IndexPut` session in `opt-note.md` centers on **index dtype normalization**, **launch width scaling**, **IR-backed accumulate loop shortening**, and **`compile_hint.md`** passes (`opt-round-2`, `opt-round-5`, `opt-round-8`, `opt-round-10` themes). No `attempts.md` header in this archive cites **`program-multiple-rows.md`** as the primary lever. Map `19_IndexPut` on **`scalar-latency-traps.md`**, **`tiling.md`**, and **`compile_hint.md`** instead of duplicating PMR five-field entries here.

## NPUKernelBench round narratives (pilot: ten kernels `20_*`–`24_*`, batch 4, 2026-05-08, log-backed)

*Operators: **`20_FusedRopeWithQkNormAndKvCacheUpdate`**, **`20_Gather`**, **`21_GaussianTopkSparseActivation`**, **`21_Scatter`**, **`22_HybridAttentionMaskPreparation`**, **`22_Nonzero`**, **`23_HyenaFftSizePaddingRfft`**, **`23_RepeatInterleave`**, **`24_EmbeddingDenseBackward`**, **`24_KvCacheUpdateWithRopeBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `20_FusedRopeWithQkNormAndKvCacheUpdate`

**`opt-round-4` (parent `baseline`)** — `20_FusedRopeWithQkNormAndKvCacheUpdate/opt-round-4/attempts.md`

- **Kernel / round / parent:** `20_FusedRopeWithQkNormAndKvCacheUpdate` / `opt-round-4` / baseline.
- **Pre-change scenario:** Rounds 1–3 load-trimming experiments **regressed**; benchmark mix is dominated by **`head_dim=64`** with **one row per program** (`attempts.md`).
- **Change:** **Batch multiple rows per program** on the dominant 64-dim path; keep a generic single-row fallback for the 128-dim path.
- **Evidence:** Correctness passed; `compare-perf` **Avg +48.0%**, **2.14×** geomean, **1.77×** total vs baseline (`attempts.md`); **promoted**.
- **Interpretation:** First lever that cuts real kernel time after failed scalar/load pivots—classic PMR on row-rich rope+norm work.

**`opt-round-6` (parent `opt-round-5`)** — theme in `20_FusedRopeWithQkNormAndKvCacheUpdate/opt-note.md` + `opt-round-6/attempts.md`

- **Kernel / round / parent:** `20_FusedRopeWithQkNormAndKvCacheUpdate` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** 64-dim path still left throughput on the table after r4–r5 row widening (`opt-note.md`).
- **Change:** Raised **64-dim** row batch from **4→8** rows per program (session arc).
- **Evidence:** Correctness passed; further gains vs r5 and baseline (`opt-note.md`); **promoted** through ladder.
- **Interpretation:** Head-dim-specialized PMR needs explicit **upper bound** search—8-wide was next safe step after 4-wide.

**`opt-round-8` (parent `opt-round-6`)** — theme in `opt-note.md` + `opt-round-8/attempts.md`

- **Kernel / round / parent:** `20_FusedRopeWithQkNormAndKvCacheUpdate` / `opt-round-8` / `opt-round-6`.
- **Pre-change scenario:** 128-dim path still used smaller row batch than 64-dim after r6 (`opt-note.md`).
- **Change:** Increased **128-dim** row batch from **2→4** rows per program.
- **Evidence:** Correctness passed; improved vs r6 and baseline (`opt-note.md`); **promoted** before fixed-shape split in r9.
- **Interpretation:** **Per-head_dim batching policy**—do not assume one `BLOCK_ROWS` across all rope widths.

### `21_GaussianTopkSparseActivation`

**`opt-round-2` (parent `opt-round-1`)** — `21_GaussianTopkSparseActivation/opt-round-2/attempts.md`

- **Kernel / round / parent:** `21_GaussianTopkSparseActivation` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** `_gaussian_row_mean_kernel` stayed **one program per row** on `rows ∈ {500,640,2048}` while r1 already cut pipeline cost (`attempts.md`; cites **`program-multiple-rows.md`**).
- **Change:** Batched **8 rows** per mean program (`attempts.md` attempt 2).
- **Evidence:** Correctness passed but **Avg −20.2%**, **0.93×** geomean vs baseline—shape-sensitive regressions on cases 2/4/5 (`attempts.md`); **not promoted**.
- **Interpretation:** **Negative PMR evidence**—isolated row widening on one reduction stage hurts without co-tuning variance/output stages (session recovered via r3–r10 on other cards).

### `24_EmbeddingDenseBackward`

**`opt-round-1` (parent `baseline`)** — theme in `24_EmbeddingDenseBackward/opt-note.md` + `24_EmbeddingDenseBackward/opt-round-1/attempts.md`

- **Kernel / round / parent:** `24_EmbeddingDenseBackward` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline launched **one hidden tile per program** on the dominant backward path (`opt-note.md`).
- **Change:** **Group two hidden-dimension tiles per program** to amortize launch and pointer setup.
- **Evidence:** Correctness passed; **4/5** cases improved; `compare-perf` **1.50×** geomean vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Embedding backward is column-band structured; modest **hidden-axis program widening** is PMR-shaped even when rows dominate in other embeddings.

### `24_KvCacheUpdateWithRopeBackward`

**`opt-round-1` (parent `baseline`)** — theme in `24_KvCacheUpdateWithRopeBackward/opt-note.md` + `opt-round-1/attempts.md`

- **Kernel / round / parent:** `24_KvCacheUpdateWithRopeBackward` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline mapped **one sequence row per program** on the fused KV-cache+rope backward path (`opt-note.md`).
- **Change:** **Batch multiple sequence rows per Triton program** on the hot path.
- **Evidence:** Correctness passed; improved **all five** representative cases vs baseline (`opt-note.md`); **promoted**.
- **Interpretation:** KV rope backward matches attention-adjacent PMR—row bands before head-dim micro-tiles.

**`opt-round-5` (parent `opt-round-3`)** — theme in `opt-note.md` + `24_KvCacheUpdateWithRopeBackward/opt-round-5/attempts.md`

- **Kernel / round / parent:** `24_KvCacheUpdateWithRopeBackward` / `opt-round-5` / `opt-round-3`.
- **Pre-change scenario:** Profiler-backed movement pressure for **`head_dim ≤ 128`** still left row tiles conservative after r3’s 2D grid (`opt-note.md`).
- **Change:** **Wider row tile** for `head_dim` up to **128** (profile-driven `BLOCK_ROWS` increase).
- **Evidence:** Correctness passed; material gain vs r3 and baseline (`opt-note.md`); **promoted**.
- **Interpretation:** Small-head regimes are **row-tile sensitive** once inner stride path is stable.

**`opt-round-8` (parent `opt-round-7`)** — `24_KvCacheUpdateWithRopeBackward/opt-round-8/attempts.md`

- **Kernel / round / parent:** `24_KvCacheUpdateWithRopeBackward` / `opt-round-8` / `opt-round-7`.
- **Pre-change scenario:** Unconditional **`BLOCK_ROWS=16`** on small-head path **regressed** `seq_len ∈ {4,16}` while helping heavy cases (`attempts.md`).
- **Change:** **Gate** `BLOCK_ROWS=16` on **`seq_len >= 32`**; keep **8-row** fallback for shorter sequences when `head_dim ≤ 128`.
- **Evidence:** Correctness passed; **Avg +59.8%**, **3.09×** geomean vs baseline; modest parent win over r7 (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** PMR widening must be **sequence-length gated**—tiny-seq cases are launch-latency limited.

### Other operators in this batch (`20_Gather`, `21_Scatter`, `22_HybridAttentionMaskPreparation`, `22_Nonzero`, `23_HyenaFftSizePaddingRfft`, `23_RepeatInterleave`)

`20_Gather` PMR is **not** the session winner—**rank-2 dim-0 specialization** lives on **`gather-load.md`**. `21_Scatter` emphasizes **row+inner mapping** under **`layout-store-and-block-pointers.md`** + **`grid-flatten-and-ub-buffering.md`**. `22_HybridAttentionMaskPreparation` is **tile/autotune**-first (`tiling.md`, **`autotune.md`**). `22_Nonzero` is **routing / dense-tile** (`tiling.md`, **`cache_use.md`**). `23_HyenaFftSizePaddingRfft` and `23_RepeatInterleave` are **inner-axis tiling / store-shape** stories on **`tiling.md`** and **`compile_hint.md`**.

## NPUKernelBench round narratives (pilot: ten kernels `25_*`–`29_*`, batch 5 final, 2026-05-08, log-backed)

*Operators: **`25_MaskedSoftmaxWithAttentionDropoutBackward`**, **`25_NLLLoss`**, **`26_AvgPool3d`**, **`26_MoeGroupScoreAggregationAndMasking`**, **`27_MaxPool3d`**, **`27_MultiMaskAttentionAggregation`**, **`28_Interpolate`**, **`28_MultimodalRopePositionComputationWithGridBasedIndexing`**, **`29_DynamicQuant`**, **`29_TanhGatedResidualAddBackward`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `25_NLLLoss`

**`opt-round-2` (parent `opt-round-1`)** — `25_NLLLoss/opt-round-2/attempts.md`

- **Kernel / round / parent:** `25_NLLLoss` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Mean kernel still used **flattened-position decode** (`// spatial_size`, `% spatial_size`) on the unweighted mean hot path (`attempts.md`; cites **`layout-store-and-block-pointers.md`**).
- **Change:** **`batch × spatial-block`** partial-reduction grid; removed per-lane **`batch_idx` / `spatial_idx`** reconstruction on that path.
- **Evidence:** Correctness passed; **Avg +14.9%**, **1.27×** geomean, **1.13×** total vs baseline (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** PMR-shaped **launch remap**—2D workloads need a 2D program grid before row-only batching experiments.

**`opt-round-3` (parent `opt-round-2`)** — theme in `25_NLLLoss/opt-note.md` + `25_NLLLoss/opt-round-3/attempts.md`

- **Kernel / round / parent:** `25_NLLLoss` / `opt-round-3` / `opt-round-2`.
- **Pre-change scenario:** Hypothesis to add **row batching** for **`spatial_size == 1`** subcase (`opt-note.md`).
- **Change:** Row-batched variant for that subcase.
- **Evidence:** Correctness passed but **larger 2D row counts regressed** enough to lose vs r2 overall (`opt-note.md`); **validated branch**, not promoted.
- **Interpretation:** **Negative PMR** when batching widens programs without recovering spatial parallelism—gate tiny-shape batching carefully.

### `26_MoeGroupScoreAggregationAndMasking`

**`opt-round-6` (parent `baseline`)** — theme in `26_MoeGroupScoreAggregationAndMasking/opt-note.md` + `26_MoeGroupScoreAggregationAndMasking/opt-round-6/attempts.md`

- **Kernel / round / parent:** `26_MoeGroupScoreAggregationAndMasking` / `opt-round-6` / baseline.
- **Pre-change scenario:** Rounds **1–5** structural/fusion/PMR attempts **regressed** kernel benchmark vs baseline (`opt-note.md`).
- **Change:** **Tiny-input specialization** with baseline fallback for larger shapes.
- **Evidence:** Correctness passed; first **net win** vs baseline (`opt-note.md`); **promoted** anchor for r7–r9 ladder.
- **Interpretation:** This operator’s PMR wins are **regime-gated**—default row batching hurts the **512-row** path (see r10).

**`opt-round-9` (parent `opt-round-8`)** — theme in `opt-note.md` + `26_MoeGroupScoreAggregationAndMasking/opt-round-9/attempts.md`

- **Kernel / round / parent:** `26_MoeGroupScoreAggregationAndMasking` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Specialization validated through **20-** and **75-row** regimes; **512×256** still open (`attempts.md` narrative).
- **Change:** Extended specialization to **`75 × 256`** representative case while keeping larger cases near baseline (`opt-note.md`).
- **Evidence:** Correctness passed; **cases 1–4 improved**, 512-row case **slight regression** accepted for aggregate win (`opt-note.md`); **final best** before r10.
- **Interpretation:** PMR ladder stops before the **512-row** bulk path—see r10 anti-result.

**`opt-round-10` (parent `opt-round-9`)** — `26_MoeGroupScoreAggregationAndMasking/opt-round-10/attempts.md`

- **Kernel / round / parent:** `26_MoeGroupScoreAggregationAndMasking` / `opt-round-10` / `opt-round-9`.
- **Pre-change scenario:** Hypothesis: conservative **`BLOCK_M=2`** on **`num_tokens > 75`** exact-grid prefix (`attempts.md`; cites **`program-multiple-rows.md`**).
- **Change:** Even-row prefix with **`BLOCK_M=2`** + one-row tail fallback.
- **Evidence:** Correctness passed but **case 5 regressed** **`47.419 → 59.179 µs`**; **Total 0.92×** vs r9 (`attempts.md`); **not promoted**.
- **Interpretation:** **Hard anti-PMR** on large-token MOE score paths—keep r9-style specialization without wide-row batching on the 512 regime.

### `29_DynamicQuant`

**`opt-round-6` (parent `opt-round-5`)** — theme in `29_DynamicQuant/opt-note.md` + `29_DynamicQuant/opt-round-6/attempts.md`

- **Kernel / round / parent:** `29_DynamicQuant` / `opt-round-6` / `opt-round-5`.
- **Pre-change scenario:** Very largest **wide steady-state** path still left row coverage on the table after r5 absorbed lone chunks (`opt-note.md`).
- **Change:** **Higher row coverage** (`wide_block_m` / row-group tuning) on the dominant very-large-width regime.
- **Evidence:** Correctness passed; **Avg +54.0%**, **2.27×** geomean, **3.06×** total vs baseline (`opt-note.md`); **promoted** toward r9.
- **Interpretation:** Wide-row quant kernels are **PMR + chunk-width** coupled—raise row groups only where profiler says steady-state dominates.

**`opt-round-9` (parent `opt-round-8`)** — `29_DynamicQuant/opt-round-9/attempts.md`

- **Kernel / round / parent:** `29_DynamicQuant` / `opt-round-9` / `opt-round-8`.
- **Pre-change scenario:** Final bounded experiment: **`wide_block_m` 32→48** only when **`cols >= 24576`** (`attempts.md`).
- **Change:** Increased row group on that guard; preserved r8 chunk geometry elsewhere.
- **Evidence:** Correctness passed; **Avg +54.1%**, **2.29×** geomean, **3.18×** total vs baseline (`attempts.md`); **final best** (`opt-note.md`).
- **Interpretation:** Last row-group step helped dominant case enough to offset a small case-4 slip—**stop when r10 confirms saturation** (`opt-note.md`).

### Other operators in this batch (`25_MaskedSoftmax*`, `26_AvgPool3d`, `27_*`, `28_*`, `29_TanhGated*`)

`25_MaskedSoftmaxWithAttentionDropoutBackward` is **kernel split + block-tier tuning** on **`attention-cv-pipeline.md`** and **`tiling.md`**, not row batching. `26_AvgPool3d` / `27_MaxPool3d` / `28_Interpolate` / `28_MultimodalRopePositionComputationWithGridBasedIndexing` emphasize **tiling**, **dispatch**, and **gather layout** on **`tiling.md`** / **`layout-store-and-block-pointers.md`**. `27_MultiMaskAttentionAggregation` adds **LICM** on **`loop-invariant-hoisting.md`**. `29_TanhGatedResidualAddBackward` **`opt-round-4`** is **split-launch / tail masking** on **`grid-flatten-and-ub-buffering.md`** + **`scalar-latency-traps.md`**.

## Gap-fill addendum (inventory alignment, 2026-05-08)

### `23_HyenaFftSizePaddingRfft`

**`opt-round-1` (parent `baseline`)** — `23_HyenaFftSizePaddingRfft/opt-round-1/attempts.md`

- **Kernel / round / parent:** `23_HyenaFftSizePaddingRfft` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline path under-amortized row work on wide FFT-padding outputs.
- **Change:** Introduced row-tiled contiguous kernel organization (foundation for later width-ladder tuning).
- **Evidence:** Correctness passed; promoted as first strong branch (`opt-note.md`).
- **Interpretation:** Even before later tile-width ladders, this operator used PMR-style row grouping as an initial lever.

### `23_RepeatInterleave`

**`opt-round-1` (parent `baseline`)** — `23_RepeatInterleave/opt-round-1/attempts.md`

- **Kernel / round / parent:** `23_RepeatInterleave` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline repeated/interleaved path launched too fine-grained work on large contiguous slices.
- **Change:** Input-oriented row-tile replication baseline rewrite.
- **Evidence:** Correctness passed; promoted as first best (`opt-note.md`) and became the base for later repeat-2 specializations.
- **Interpretation:** RepeatInterleave begins with PMR-like row-tile grouping before compile/store-shape refinements.

### `27_MaxPool3d`

**`opt-round-2` (parent `opt-round-1`)** — `27_MaxPool3d/opt-round-2/attempts.md`

- **Kernel / round / parent:** `27_MaxPool3d` / `opt-round-2` / `opt-round-1`.
- **Pre-change scenario:** Round-1 row-tiled no-index kernel still underfit narrow output-width regimes.
- **Change:** Specialized no-index fast path for narrow widths by adjusting row/width work partition.
- **Evidence:** Correctness passed; promoted over round 1 (`opt-note.md`).
- **Interpretation:** MaxPool3d progression includes explicit per-program row/work partition tuning before later full-window and strip-staging wins.

### `27_MultiMaskAttentionAggregation`

**`opt-round-5` (parent `opt-round-4`)** — `27_MultiMaskAttentionAggregation/opt-note.md` + `opt-round-5/attempts.md`

- **Kernel / round / parent:** `27_MultiMaskAttentionAggregation` / `opt-round-5` / `opt-round-4`.
- **Pre-change scenario:** Fused float32 mean path improved, but tiny-shape and medium-shape batching trade-offs remained.
- **Change:** Kept medium-shape row batching while restoring tiny-shape block settings on the validated fused path.
- **Evidence:** Correctness passed; promoted over round 4 with improved aggregate metrics (`opt-note.md`).
- **Interpretation:** MultiMask attention combines reduction fusion with PMR-style row-block tuning in the profitable float32 mean regime.
