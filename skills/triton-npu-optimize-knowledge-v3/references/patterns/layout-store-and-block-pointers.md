# Layout, Store, And Block Pointer Pattern

## Summary

Use this pattern when latency is limited by memory layout expression and transfer shape, not arithmetic complexity. The goal is to present movement as contiguous vector-friendly tiles instead of flattened scalarized address chains, transpose-at-store paths, or many tiny stores.

Common levers:
- rewrite pointer math to match true multidimensional layout,
- raise block-pointer dimensionality when dimensions/strides are known,
- merge adjacent stores/loads and remove avoidable transpose overhead.

## Use When

- Adjacent destinations are written by many narrow `tl.store` operations.
- Store order is effectively transposed relative to destination contiguity.
- Contiguous multidimensional tensors are accessed through flattened 1D decode chains.
- Inner dimensions are looped/pid-decoded even though they can be encoded in tile shape.
- Dot paths use avoidable transpose/cast ordering.

## Avoid When

- Main bottleneck is still launch geometry, scalar control, or algorithm shape.
- Continuity assumptions are weak or shape-conditional without dispatch guards.
- Block-pointer metadata cannot be expressed robustly for all active regimes.

## Signals

### Code

- Repeated scalar index arithmetic around otherwise simple movement.
- Accumulators or intermediates are transposed only at final store.
- Many small loads/stores where contiguous vectors are possible.

### Profile

- Transfer overhead remains high after first-order tiling.
- Gains plateau until layout/store representation changes.

## Repairs

### Merge adjacent stores

When destination offsets are contiguous and mask-compatible, issue one wider store.

```python
offs = base + tl.arange(0, BLOCK)
vals = compute_contiguous_values(...)
tl.store(out + offs, vals, mask=offs < N)
```

Never merge stores when destination intervals are disjoint or masks represent different semantic regions (for example mixed interior/tail conditions).

### Avoid store transpose degradation

Carry data in store-major layout earlier so final store follows contiguous direction directly.

### Raise block-pointer dimensionality

For genuinely multidimensional contiguous layouts, encode dimensions explicitly.

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

### Vectorize an inner dimension

Promote small inner loops into tensor axes/block shape when that dimension is naturally contiguous.

### Let Cube handle transpose after dtype conversion

For dot consumers, prefer:

```python
b = b.to(tl.float16)
acc = tl.dot(a, tl.trans(b))
```

instead of:

```python
b = tl.trans(b).to(tl.float16)
acc = tl.dot(a, b)
```

This reorder is only intended for `tl.dot`-consumed operands. Do not apply it to pure vector pipelines where transpose/materialization behavior follows different lowering rules.

### Bias with matmul

Load bias with explicit output-column offsets and add in the epilogue shape already matching accumulator layout.

## Risks

- Axis/order mistakes can silently change semantics.
- Store merging on noncontiguous destinations breaks correctness.
- Incorrect block-pointer fields (`shape/strides/offsets/block_shape/order`) can benchmark wrong access patterns.
- More aggressive vectorized transfers may increase register pressure.

## What To Verify After Applying

- Correctness on tail and noncontiguous-stride cases.
- Parent-vs-child benchmark gains on same harness.
- Evidence that wins come from transfer-shape improvements (fewer tiny stores, better contiguous access, less transpose overhead).
- Block-pointer metadata and masks stay valid across dispatched regimes.
- Record changed shape/axis assumptions in `attempts.md` when running optimization rounds so follow-up passes preserve the same layout contract.

## Related Patterns

- `tiling`
- `program-multiple-rows`
- `scalar-latency-traps`
- `compile_hint`
