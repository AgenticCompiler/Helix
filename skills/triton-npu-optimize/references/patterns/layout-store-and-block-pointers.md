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
