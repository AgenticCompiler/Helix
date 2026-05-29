# Static Range to Range Conversion Pattern (Loop Unrolling → Software Pipeline)

## Summary

Replace `tl.static_range` with `tl.range` in hot loops where iteration bodies are lightweight and iterations have no cross-iteration data dependencies. This allows the compiler to preserve loop structure and apply software pipelining, overlapping memory transfers and compute across iterations — instead of fully unrolling the loop into a flat instruction sequence that prevents inter-iteration overlap.

On Ascend NPU, this pattern converts a serialized "load→compute→store per iteration" execution into a pipelined cascade where row N's store and row N+1's load run concurrently with row N's compute.

## Use When

- Hot loop uses `tl.static_range` (or `tl.range` with a `tl.constexpr` bound that triggers full unrolling).
- Loop iterations are **independent** — no loop-carried data dependency between iterations.
- Loop body is lightweight (simple elementwise ops, few intermediates).
- Profiling shows MTE2 (DMA) ratio is disproportionately low relative to VECTOR, and SCALAR/MTE2 overlap is poor.

## Avoid When

- Loop body is compute-heavy with many intermediates — `tl.range` adds register rename pressure that may cause spills.
- `BLOCK_SIZE` is very large (≥4096) **and** loop body is complex — each iteration's register footprint is already high.
- `num_warps` is already large (≥8) — multi-warp register doubling plus range rename overhead may overflow.
- Iteration count is very small (≤4) — unrolling cost is negligible, pipeline depth is insufficient for meaningful overlap.
- Cross-iteration dependencies exist (e.g., reduction accumulators that must complete before next iteration begins).

## Signals

### Code

- `tl.static_range(BLOCK_M)` or `tl.range(BLOCK_M)` where `BLOCK_M` is `tl.constexpr` in a hot loop.
- Loop body contains load → compute → store with no dependency on previous iteration's output.
- Simple per-iteration work: elementwise activation, copy, or light arithmetic.

### Profile

- **MTE2 ratio far below VECTOR ratio.** For memory-bound operators, DMA engine utilization should approach or exceed compute engine utilization. When MTE2 dur% is only ~60–70% of VECTOR dur%, DMA is likely waiting on scalar address generation — pipeline bubbles from fully unrolled iterations.
- **SCALAR & MTE2 overlap < 50%.** `%(SCALAR&MTE2/SCALAR)` describes the fraction of SCALAR's total runtime that overlaps with MTE2. When this ratio is below 50%, over half of SCALAR time runs without MTE2 overlap, meaning address generation and DMA are serialized rather than parallel. The reverse view `%(SCALAR&MTE2/MTE2)` (typically ~15–25% in affected kernels) further confirms that only a small portion of MTE2's time benefits from concurrent scalar work.
- **MTE2 & VECTOR overlap is asymmetric.** While `%(MTE2&VECTOR/MTE2)` may appear reasonable (~70–80%), `%(MTE2&VECTOR/VECTOR)` at ~50% or below reveals that VECTOR spends a large fraction of its time without MTE2 overlap — compute is starved of incoming data for long stretches.
- **SCALAR & VECTOR overlap is negligible.** `%(SCALAR&VECTOR/VECTOR)` typically <5% and `%(SCALAR&VECTOR/SCALAR)` <15% indicate scalar address generation and vector compute are almost entirely serialized — a hallmark of fully unrolled loops where each iteration's address setup blocks the next compute.
- **Pipeline flows are unidirectional only.** Only `MTE2→VECTOR` and `VECTOR→MTE3` flows appear, with no reverse flows (`MTE3→MTE2`, `VECTOR→MTE2`). Missing reverse flows mean the compiler is not cascading iterations — each iteration completes fully before the next begins.
- **High WAIT_FLAG and BAR counts.** Elevated `WAIT_FLAG` and `BAR` counts (e.g., WAIT_FLAG >10000, BAR >20000) with heavy VECTOR-side synchronization indicate serialization barriers between pipeline stages, consistent with iteration-by-iteration execution where each stage must complete before the next begins.

### Profile Case: SwiGLU Kernel (BLOCK_M=128, BLOCK_N=4096, elementwise body)

Before optimization (using `tl.static_range`), the profiling data showed:

| Metric | Value | Interpretation |
|---|---|---|
| SCALAR dur% | 13.19% | Moderate scalar overhead |
| VECTOR dur% | 46.23% | Dominant pipe |
| MTE2 dur% | 29.14% | Starved — only 63% of VECTOR |
| MTE3 dur% | 20.65% | |
| %(SCALAR&MTE2/SCALAR) | 43.20% | Over half of SCALAR runs without MTE2 overlap |
| %(SCALAR&MTE2/MTE2) | 19.56% | MTE2 barely benefits from concurrent SCALAR |
| %(MTE2&VECTOR/MTE2) | 77.16% | MTE2 mostly overlaps VECTOR |
| %(MTE2&VECTOR/VECTOR) | 48.65% | But VECTOR is starved half the time |
| %(SCALAR&VECTOR/VECTOR) | 2.69% | SCALAR and VECTOR nearly fully serialized |
| %(SCALAR&VECTOR/SCALAR) | 9.43% | |
| Pipeline Flows | MTE2→VECTOR, VECTOR→MTE3 (2 unidirectional) | No inter-iteration cascade |
| WAIT_FLAG total | 12300 (VECTOR: 8200, MTE3: 4100) | Heavy sync barriers |
| BAR total | 25420 (VECTOR: 21320) | |

After replacing `tl.static_range` with `tl.range`, key improvements:

| Metric | Before | After | Change |
|---|---|---|---|
| MTE2 dur% | 29.14% | 47.19% | ↑**18pp (+62%)** |
| MTE2/VECTOR ratio | 63% | 95% | MTE2 no longer starved |
| %(SCALAR&MTE2/SCALAR) | 43.20% | 55.11% | ↑**11.9pp** — better address/DMA overlap |
| %(MTE2&VECTOR/MTE2) | 77.16% | 90.49% | ↑**13.3pp** |
| %(MTE2&VECTOR/VECTOR) | 48.65% | 85.94% | ↑**37.3pp** — VECTOR no longer starved |
| Pipeline Flows | 2 unidirectional | 4 (with reverse) | Cascaded pipeline activated |
| Latency | 129μs | 117μs | ↓**9.3%** |

## Optimization Strategy

1. **Identify candidate loop**: find `tl.static_range` in the hot path where iterations are independent.
2. **Replace `tl.static_range(N)` with `tl.range(N)`**: this tells the compiler to keep the loop structure rather than fully unroll.
3. **Verify the compiler applies software pipelining**: check that pipeline flows now include reverse edges (`MTE3→MTE2`, `VECTOR→MTE2`), indicating inter-iteration overlap.
4. **Validate register pressure**: ensure no spill or UB overflow from the rename overhead `tl.range` introduces.

### Before

```python
@triton.jit
def kernel(x_ptr, y_ptr, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    for i in tl.static_range(BLOCK_M):  # fully unrolled — no pipeline opportunity
        row_offs = offs_m[i] * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = mask_m[i] & (tl.arange(0, BLOCK_N) < BLOCK_N)
        x_row = tl.load(x_ptr + row_offs, mask=row_mask, other=0.0)
        y_row = x_row * tl.sigmoid(x_row)  # elementwise compute
        tl.store(y_ptr + row_offs, y_row, mask=row_mask)
```

### After

```python
@triton.jit
def kernel(x_ptr, y_ptr, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    for i in tl.range(BLOCK_M):  # loop preserved — compiler can pipeline
        row_offs = offs_m[i] * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = mask_m[i] & (tl.arange(0, BLOCK_N) < BLOCK_N)
        x_row = tl.load(x_ptr + row_offs, mask=row_mask, other=0.0)
        y_row = x_row * tl.sigmoid(x_row)
        tl.store(y_ptr + row_offs, y_row, mask=row_mask)
```

The compiler now schedules iterations as a pipeline cascade:

```
[Row N MTE2 load] → [Row N Vector compute] → [Row N MTE3 store]
          ↘ [Row N+1 MTE2 load] → [Row N+1 Vector compute] → ...
```

## Performance Expectations

Expected gain range: **5–15%** for lightweight loop bodies with independent iterations and clear MTE2 starvation signals. Gains are larger when MTE2 underutilization is more severe.

Key improvement patterns to expect after applying `tl.range`:

- **MTE2 dur% rises sharply** (often +15–20pp), as DMA no longer waits on serialized address generation.
- **%(SCALAR&MTE2/SCALAR) improves** (typically +10–15pp), indicating better overlap between address computation and DMA transfer.
- **%(MTE2&VECTOR/VECTOR) improves dramatically** (often +30–40pp), as compute no longer starves while waiting for data.
- **Pipeline flows gain reverse edges** (`MTE3→MTE2`, `VECTOR→MTE2`), confirming inter-iteration cascading is active.
- **SCALAR dur% decreases slightly**, as loop-control overhead is reduced and register renaming is more efficient than full unrolling.

## Practical Notes

- This pattern is a **single-line change** (`tl.static_range` → `tl.range`) but its effect depends entirely on the compiler's ability to pipeline the resulting loop structure.
- The pattern is most effective when the loop body is simple (few intermediates, no reduction) — complex bodies may not pipeline well and the register rename cost can dominate.
- `tl.range` with a `tl.constexpr` bound may still be unrolled by the compiler in some cases; verify via profiling that pipeline flows actually change.
- This pattern is complementary to `software-pipeline` (explicit prefetch/advance staging): `tl.range` enables *implicit* compiler-driven pipelining, while `software-pipeline` enables *explicit* manual staging. Try `tl.range` first as it requires no structural changes; escalate to explicit pipelining if compiler-driven results are insufficient.
- If `tl.range` causes regressions (spills, longer compile, no pipeline improvement), the loop body may be too heavy — revert and consider explicit software pipelining or loop-invariant hoisting to reduce per-iteration cost first.

## Risks

- **Register pressure increase**: `tl.range` requires the compiler to rename/reuse registers across iterations, which can cause spills if the loop body already has high register demand.
- **Compile time increase**: the compiler must analyze and schedule the loop for pipelining rather than simply emitting unrolled copies.
- **No guaranteed pipeline**: the compiler may not always apply software pipelining; always verify via profiling.
- **Interaction with `num_warps`/`num_stages`**: higher warp counts multiply register pressure; if `tl.range` causes issues at `num_warps=8`, try reducing warps or increasing `num_stages` to give the scheduler more room.

## What To Verify After Applying

1. **Correctness**: output matches reference across full and boundary tiles.
2. **Pipeline activation**: profiling shows reverse pipeline flows (`MTE3→MTE2`, `VECTOR→MTE2`) and improved MTE2/VECTOR overlap.
3. **No register spills**: SCALAR ratio should decrease, not increase (spills would raise it).
4. **Parent-vs-child benchmark**: confirm real latency improvement on representative shapes.
5. **Stability across configs**: verify the change does not regress at different `num_warps`/`num_stages` settings.

## Related Patterns

- `software-pipeline` — explicit prefetch/advance staging for cases where compiler-driven pipelining is insufficient.
- `loop-invariant-hoisting` — reduce per-iteration cost before applying this pattern if loop body is heavy.
- `scalar-latency-traps` — related scalar/address-generation bottlenecks that may co-occur.
- `tiling` — ensure loop is already well-tiled before attempting pipeline conversion.
