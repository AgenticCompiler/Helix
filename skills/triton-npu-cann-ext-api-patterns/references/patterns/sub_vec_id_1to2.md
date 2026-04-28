# 1:2 `sub_vec_id` Rewrite Pattern

Use this reference when a Triton Ascend NPU kernel should split vector work across the two `sub_vec_id()` lanes while preserving full-tile cube math.

## When To Use

- the kernel mixes vector work and `tl.dot` work
- cube math should remain full-tile
- the kernel can stage half-tile vector results into a full-tile cube handoff

Do not use this pattern for:

- purely elementwise kernels
- kernels that have no real vector/cube handoff
- rewrites that depend on lane-0-only compute fallback

Choose this pattern when the main problem is:

- vector utilization is likely below what the A5 mixed vector-plus-cube structure can support

## Quick Start

1. Copy the vanilla kernel module to a sibling `_sub_vec` module and keep the original unchanged.
2. Use `vec_id = tle.sub_vec_id()` to assign disjoint half-tile vector ownership.
3. Keep `tl.dot` full-tile unless mathematical proof and numerical validation justify otherwise.
4. Make vector-to-cube handoff explicit with shared staging plus `tl.debug_barrier()`.
5. Prefer `tle.fixpipe(..., NZ2ND, ROW_SPLIT)` for cube-to-vector handoff when applicable.
6. Do not fake correctness by routing real compute through lane 0 only.
7. Validate against the same golden as baseline, then benchmark and record before/after tile choices.
8. Keep `_sub_vec` diffs reviewable and structure-preserving.

## Rewrite Goal

Split vector work so vector units process half of the selected tile axis while cube keeps authoritative full-tile `tl.dot` math.

## Why It Helps

On A5, each AI Core exposes AI Vector and AI Cube compute resources in an effective `1:2` ratio. For mixed vector-plus-cube kernels, the vector side can become the underutilized or limiting side even when the cube path still has headroom. A 1:2 vector split is therefore a practical way to raise vector utilization without shrinking the authoritative full-tile cube math.

In practice this can:

- better match vector work distribution to the A5 AI Core vector:cube ratio
- improve vector utilization without shrinking cube math
- unlock larger viable tiles
- improve vector-heavy tail behavior

As a rule of thumb, if a kernel is genuinely mixed vector-plus-cube, and the structure matches this handoff pattern, this rewrite is worth trying as an optimization candidate.

## Mandatory Rules

1. Vector does vector work and cube does `tl.dot`.
2. Both lanes must do real disjoint work.
3. Handoff must be explicit and synchronized.
4. Do not replace full-dot semantics with partial-dot shortcuts without proof.
5. Keep rollout reviewable through `_sub_vec` duplication, stable signatures, and minimal unrelated refactors.

## Lane Setup

```python
import triton.language.extra.cann.extension as tle

HALF_BT: tl.constexpr = BT // 2
vec_id = tle.sub_vec_id()
half_offset = (vec_id * HALF_BT).to(tl.int32)
```

Use `half_offset` in block pointer origins so the two lanes cover disjoint halves.

## Vector To Cube Handoff

- Each lane writes its own half into shared scratch.
- Synchronize with `tl.debug_barrier()`.
- Reload the full tile for cube consumption.

Example:

```python
HALF_BT: tl.constexpr = BT // 2
vec_id = tle.sub_vec_id().to(tl.int32)
half_off = (vec_id * HALF_BT).to(tl.int32)

p_k_half = tl.make_block_ptr(
    k_ptr,
    (K, T),
    (1, H * K),
    ((i_k * BK).to(tl.int32), (i_t * BT + half_off).to(tl.int32)),
    (BK, HALF_BT),
    (0, 1),
)
b_k_half = tl.load(p_k_half, boundary_check=(0, 1))

p_tmp_half = tl.make_block_ptr(
    temp_k + scratch_base,
    (BK, BT),
    (BT, 1),
    (0, half_off),
    (BK, HALF_BT),
    (1, 0),
)
tl.store(p_tmp_half, b_k_half, boundary_check=(0, 1))
tl.debug_barrier()
```

## Cube To Vector Handoff

Prefer:

```python
b_half = tle.fixpipe(
    b_cube,
    dma_mode=tle.FixpipeDMAMode.NZ2ND,
    dual_dst_mode=tle.FixpipeDualDstMode.ROW_SPLIT,
)
```

If fixpipe does not apply, use explicit full-tile store plus half-tile reload with synchronization.

## Common Mistakes

### Lane-0-only fallback

Wrong direction:

- lane 0 does the real computation
- lane 1 stays idle

Correct direction:

- both lanes own disjoint halves
- lane guards are used only for race-free write-once shared outputs when required

### Missing barrier before full-tile cube load

Wrong direction:

- one lane loads the full tile before the other lane has finished its half store

Correct direction:

- store lane halves
- call `tl.debug_barrier()`
- then reload the full tile for cube

### Assuming partial dot equals full-dot slice

Wrong direction:

- replacing the authoritative full `tl.dot` with a reduced-shape approximation

Correct direction:

- keep the mathematically authoritative full dot
- split afterward with `fixpipe` or explicit staged half-loads

## Validation Checklist

- confirm the kernel is genuinely mixed vector-plus-cube and not just a mostly-vector kernel with incidental dot usage
- confirm lane ownership is disjoint and symmetric
- confirm no lane-0-only computational fallback exists
- confirm full-tile `tl.dot` semantics are preserved where required
- confirm vector/cube handoff synchronization is explicit
- confirm scratch allocation is sized for the chosen tile sweep
- confirm correctness passes against the same golden as baseline
- confirm before/after benchmark data is recorded with tile parameters
- confirm vanilla and `_sub_vec` paths remain clearly separable
