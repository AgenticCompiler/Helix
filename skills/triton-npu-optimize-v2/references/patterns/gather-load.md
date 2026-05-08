# NPU Gather Load Optimization Pattern

## Summary

Optimize gather-like kernels by transforming index-heavy scattered reads into load shapes that are closer to contiguous copy work. On Ascend NPU, gather performance usually improves when the hot path reduces per-element index decoding and minimizes high-width index traffic.

This pattern is about **gather-specific load shaping** (index dtype, axis specialization, contiguous row/span mapping), not generic tuning.

## Use When

- The operation is semantically gather/index-select, and profiling shows gather loads dominate latency.
- The dominant cases have contiguous structure on at least one axis, even if API semantics are indexed.
- The kernel is scalar-heavy from index decode and address reconstruction.

## Avoid When

- Access is already contiguous and gather logic is not the bottleneck.
- Source/value movement is tiny and launch/setup overhead dominates.
- The main issue is dot/reduction structure (use `classic-matmul`) or broad tiling/launch geometry first.

## Signals

### Code

- Direct global loads using index vectors on the hot path.
- Repeated per-lane coordinate decode for rank handling.
- High-width index tensors (for example `int64`) used where narrower indices are valid.

### Profile

- Gather kernel consumes most time on one representative case.
- Scalar ratio remains high after simple address cleanup.

## Optimization Strategy

1. **Normalize index width where safe**: add `int32` fast paths when axis bounds fit.
2. **Specialize by dominant axis/rank regime**: split generic gather into shape-aware kernels.
3. **Map work to contiguous spans**: prefer row/span ownership over per-element indexed loads where semantics allow.
4. **Keep fallback correctness paths** for noncontiguous or unsupported regimes.
5. **Validate parent-vs-parent** after each specialization stage.

## Common Repairs

### Index dtype narrowing

If gather axis bounds fit signed 32-bit, narrow index tensors for the fast path to reduce bandwidth and cast overhead in-kernel.

### Rank/axis specialization

When one rank/axis case dominates, create a dedicated kernel for that case instead of forcing one generic gather kernel for all shapes.

### Row-copy style remap

When selected values align to contiguous inner spans, switch from per-element gather semantics to contiguous row/span movement plus local indexing logic.

### Launch-shape repair

After contiguous remap, adjust grid/program mapping to respect launch limits and keep per-program work balanced.

### Simplified code sketch

```python
# Fast path: int32 indices + contiguous inner span ownership.
idx = tl.load(index_ptr + row_offs).to(tl.int32)
src = tl.load(x_ptr + idx[:, None] * inner_size + tl.arange(0, INNER_BLOCK)[None, :])
tl.store(out_ptr + row_offs[:, None] * inner_size + tl.arange(0, INNER_BLOCK)[None, :], src)
```

## Failure Modes And Anti-signals

- **Narrowing without guardrails**: invalid dtype narrowing can break correctness on large-axis cases.
- **Generic-only persistence**: keeping one generic gather kernel after evidence of axis-specific dominance often leaves major wins unrealized.
- **Overfitting one case**: a specialization can improve the dominant case while harming the broader mix; keep fallback dispatch and parent checks.
- **Confusing cards**: if the main gain now comes from store layout cleanup, move follow-up work to `layout-store-and-block-pointers`.

## Risks

- Additional dispatch logic increases maintenance complexity.
- Specialized paths can drift in behavior if fallback parity tests are weak.
- Index width conversions can add host-side helper ops if applied too broadly.

## What To Verify After Applying

- Correctness across all indexed shapes, including boundary axis sizes.
- Parent-vs-child performance on the same benchmark mix.
- Dominant gather case latency improves without unacceptable regressions elsewhere.
- Index dtype/path dispatch guards are explicit and correct.
- Profile confirms reduced scalar index/decode pressure or lower gather-load cost.

## Related Patterns

- `discrete_memory_access`: broader staging-oriented conversion from scattered loads to contiguous-plus-local selection.
- `layout-store-and-block-pointers`: follow-up when gather load shape is fixed but store/layout addressing remains inefficient.
- `scalar-latency-traps`: use when index decode arithmetic is still the limiting factor.
- `program-multiple-rows`: combine when gather specialization benefits from wider per-program row ownership.
