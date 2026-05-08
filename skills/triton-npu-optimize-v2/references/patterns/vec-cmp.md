# Comparison Vectorization Pattern

## Summary

Use this pattern when explicit integer comparison logic becomes a scalar bottleneck on Ascend NPU.

The goal is to keep hot-path comparison and masking in vector-friendly form so conditional selection does not collapse into scalar-heavy control.

## Use When

- Explicit `i64`/`i32` comparisons drive hot-path masks for `tl.where` or similar conditional logic.
- Comparison-heavy logic appears outside compiler-optimized load/store mask fast paths.
- Profiling indicates scalar/control pressure tied to compare-and-mask sections.
- Semantics allow safe conversion to vector-friendly compare dtypes.

## Avoid When

- Comparisons are already in compiler-fused `tl.load`/`tl.store` mask expressions and perform well.
- Values are outside safe representable range for the chosen comparison cast strategy.
- Comparison path is cold and not performance relevant.
- Rewriting comparisons would add extra casts/conversions that outweigh benefits.

## Signals

### Code

- Integer compare masks are constructed explicitly and reused in hot control flow.
- Repeated compare/cast/mask scaffolding appears in inner loops.
- Integer compare results gate large vector operations through `tl.where`.

### Profile

- Scalar instruction share remains high in otherwise vector-friendly kernels.
- Throughput improves little from tiling or overlap tuning until compare path is cleaned up.

## Repairs

### Convert compare operands to vector-friendly dtype where safe

Cast operands to a compare-friendly dtype before hot comparisons when value range guarantees semantic equivalence.

### Keep compare path simple and localized

Compute mask once per logical region and reuse it, rather than rebuilding equivalent masks repeatedly.

### Preserve fast-path mask usage

When load/store mask expressions are already optimal, avoid unnecessary external mask rewrites.

### Guard precision and range assumptions

Document or enforce value-range constraints required by the dtype conversion strategy.

### Simplified code sketch

```python
# Before: explicit integer compare in hot path.
valid = offsets_i64 < limit_i64
out = tl.where(valid, x, 0.0)

# After: compare with vector-friendly dtype when range is safe.
offsets_f32 = tl.cast(offsets_i64, tl.float32)
limit_f32 = tl.cast(limit_i64, tl.float32)
valid = offsets_f32 < limit_f32
out = tl.where(valid, x, 0.0)
```

## Synthesized Guidance

- Apply this pattern when compare logic, not math throughput, limits performance.
- Start with the smallest safe compare-path rewrite and validate correctness before broad conversion.
- Combine with scalar-control cleanup (`scalar-latency-traps`) when compare logic is part of wider control overhead.
- If compare optimization yields minimal gain, re-evaluate layout/tiling/launch bottlenecks next.

## Related Patterns

- `scalar-latency-traps`
- `loop-invariant-hoisting`
- `tiling`

## What To Verify After Applying

- Boolean semantics are identical for all representative and boundary cases.
- Value-range assumptions for converted compare operands are valid.
- Hot-path scalar pressure is reduced in profile after rewrite.
- End-to-end benchmark shows net gain after including cast overhead.
