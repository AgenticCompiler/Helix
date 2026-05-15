# Scalar Latency Trap Removal Pattern

## Summary

Remove scalarizing constructs that block vector hardware utilization on Ascend NPU, including unnecessary scalar control flow, loop-carried pointer recurrences, modulo addressing, narrow masks, and int64 arithmetic on vector paths.

## Use When

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
