# Layout, Store, And Block Pointer Pattern

## Summary

Use this pattern when latency is limited by **memory layout expression** and **store/load shape**, not by arithmetic complexity. The goal is to present memory movement to the NPU as contiguous, vector-friendly tiles instead of flattened scalarized address chains, transposed store paths, or many tiny stores.

Typical levers:
- rewrite pointer math to match real multidimensional layout,
- use block pointers where dimensions/strides are known,
- merge adjacent stores and avoid transpose-at-store degradation.

## Use When

- Stores target adjacent addresses but are emitted as multiple small `tl.store` ops.
- Store order is effectively transposed relative to destination contiguity.
- A contiguous multidimensional tensor is accessed through flattened 1D offsets with heavy decode overhead.
- Inner dimensions are looped or pid-decoded even though they can be represented in tile/block shape.
- Dot paths use avoidable transpose/cast ordering that hurts load/store shape.

## Avoid When

- Main bottleneck is still launch geometry, scalar traps, or algorithm structure.
- Destination/source continuity assumptions are weak or shape-dependent without dispatch guards.
- Block-pointer metadata (`shape/strides/offsets/order`) cannot be made correct and stable.

## Signals

### Code

- Repeated scalar address arithmetic (`div/mod`, manual offset chains) around otherwise simple data movement.
- Transpose-shaped accumulators only to transpose again at store.
- Repeated narrow loads/stores where contiguous vectors are possible.

### Profile

- High transfer overhead despite moderate arithmetic.
- Improvements from row/tiling passes plateau until layout/store expression changes.

## Optimization Strategy

1. **Fix layout contract first**: make logical tile axes match physical contiguous directions.
2. **Regularize pointers**: replace flattened/manual chains with block-pointer or structured offset forms.
3. **Widen transfer granularity**: merge adjacent stores/loads when masks and destinations are compatible.
4. **Keep transpose handling close to consumer**: especially for dot paths, avoid unnecessary intermediate transpose/store forms.
5. **Validate against parent**: keep only changes that win on immediate parent comparison.

## Common Repairs

### Merge adjacent stores

If destination addresses are contiguous and mask-compatible, issue one wider store instead of many narrow stores.

### Remove transpose-at-store patterns

Carry accumulator/data in store-major layout earlier so final store follows destination contiguity directly.

### Raise block-pointer dimensionality

For truly multidimensional contiguous layouts, encode those dimensions explicitly in block pointers instead of flattening then re-decoding.

### Vectorize inner dimensions

Promote inner loops/dimensions into tensor axes and tile shapes when that dimension is naturally contiguous.

### Dot operand ordering cleanup

When transpose exists only for `tl.dot` consumption, prefer ordering that lets Cube path consume transposed view directly without extra layout churn.

### Simplified code sketch

```python
# Before: flattened offsets with transpose-at-store behavior.
offs = pid * BLOCK + tl.arange(0, BLOCK)
vals = tl.load(x_ptr + idx0 * stride0 + idx1 * stride1)
tl.store(y_ptr + offs, vals)

# After: explicit 2D tile layout via block pointers.
x_blk = tl.make_block_ptr(
    base=x_ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(m0, n0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
)
y_blk = tl.make_block_ptr(
    base=y_ptr, shape=(M, N), strides=(stride_m, stride_n),
    offsets=(m0, n0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
)
tile = tl.load(x_blk)
tl.store(y_blk, tile)
```

## Failure Modes And Anti-signals

- **Blind block-pointer rewrites regress** when applied without matching launch/program mapping.
- **Layout ports across dtype/path can fail**: a layout specialization that wins in one dtype may regress in another.
- **Store merge misuse**: merging noncontiguous or mask-incompatible stores breaks semantics.
- **Structural mismatch persists**: regularizing load/store shape alone can regress if reduction/control structure remains unsuitable.
- **Block-pointer store dtype mismatch**: when arithmetic is done in `tl.float32` but output tensors use `float16`/`bfloat16` storage, `tl.store` via block pointers enforces strict dtype matching that flat pointer stores do not. The compiler will reject a `tl.store(out_blk_ptr, float32_val)` call. **Fix**: explicitly cast the stored value before the store — `tl.store(out_blk_ptr, val.to(out_blk_ptr.dtype.element_ty), boundary_check=(0,))`. The `.to()` conversion is efficiently lowered by the compiler and the performance gain from block pointers typically outweighs its overhead. **Do not abandon block pointers because of this error** — it is a one-line fix, not a fundamental limitation.

## Risks

- Axis/order mistakes can silently change semantics.
- Higher-dimensional pointer metadata is error-prone.
- More aggressive vectorized store/load shapes may increase register pressure.

## What To Verify After Applying

- Correctness on boundary/tail and noncontiguous stride cases.
- Parent-vs-child benchmark improvements on the same harness.
- Evidence that gains come from better transfer shape (fewer tiny stores, improved contiguous access, reduced transpose overhead).
- Block-pointer fields and mask behavior are consistent across dispatched regimes.
- **Testing methodology:** When the kernel dispatches to multiple code paths (e.g., masked vs no-mask, single-pass vs multi-pass), apply block pointers to the **highest-overhead variant first** — the one with masks, boundary checks, and explicit address computation. Do NOT test on an already-optimized no-mask fast path and then generalize a null result. The no-mask path has minimal scalar overhead, so block pointer gains there are the smallest; if block pointers show no improvement on the highest-overhead variant, only then can they be rejected. On Ascend NPU, block pointers reduce scalar address generation via DMA descriptors even without `tl.advance` loops — they are not exclusively a loop-level optimization.

## Related Patterns

- `tiling`: choose tile sizes after layout/store representation is correct.
- `program-multiple-rows`: improve per-program batching once store/load shape is fixed.
- `scalar-latency-traps`: address remaining decode/control overhead.
- `compile_hint`: apply hints only after layout/store structure is stable.
