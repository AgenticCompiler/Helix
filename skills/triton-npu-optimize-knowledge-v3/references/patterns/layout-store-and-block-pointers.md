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

Use this profile gate when `extracted_bin_data/report.txt` exists under `opt-round-*` or at the operator workspace root. Parse it as text blocks headed by `core*.veccore*:`. Metric lines may be indented.

```text
core0.veccore0:
    OverlapRatio(VECTOR/CUBE & MTE2): 0.00%
    OverlapRatio(VECTOR/CUBE & MTE3): 0.00%
    OverlapRatio(MTE2 & MTE3): 99.90%
    Ratio(VECTOR/CUBE): 0.35%
```

Interpret the fields as:

- `OverlapRatio(VECTOR/CUBE & MTE2)`: compute/MTE2-load overlap. Near-zero values mean compute and MTE2 transfer are completely serialized.
- `OverlapRatio(VECTOR/CUBE & MTE3)`: compute/MTE3-store overlap. Near-zero values mean compute and MTE3 transfer are completely serialized.
- `OverlapRatio(MTE2 & MTE3)`: MTE2/MTE3 transfer overlap. Near-maximum values mean both DMA engines are active but contending — characteristic of scattered 1D access that prevents efficient pipelining.
- `Ratio(VECTOR/CUBE)`: Vector/Cube compute share of total execution or total active time. This metric is not a primary trigger for this pattern; focus on the three overlap ratios above.

Use the distribution across active cores, not a single core.

- Very low `OverlapRatio(VECTOR/CUBE & MTE2)` and `OverlapRatio(VECTOR/CUBE & MTE3)` — compute and DMA are fully serialized.
- Very high `OverlapRatio(MTE2 & MTE3)` — DMA engines contend with each other, characteristic of flattened 1D scalar access preventing contiguous data pipelining.
- Transfer overhead remains high after first-order tiling.
- Gains plateau until layout/store representation changes.

## Dependency Features

1. [profiling] Compute-DMA serialization.

   Read `OverlapRatio(VECTOR/CUBE & MTE2)` and `OverlapRatio(VECTOR/CUBE & MTE3)` from each active `core*.veccore*` block. Very low values across most active cores mean compute and both DMA engines are fully serialized — data movement completes before computation begins.

2. [profiling] DMA-engine contention.

   Read `OverlapRatio(MTE2 & MTE3)` from each active `core*.veccore*` block. Very high values mean both DMA engines are active simultaneously but contending — characteristic of scattered 1D access where neither engine can transfer efficiently, and the signature of flattened scalar address decode chains.

3. [code] Flattened 1D scalar address decode chains.

   The kernel reconstructs multi-dimensional coordinates from a flat `program_id` using integer division `//` and modulo `%` (for example `row = offsets // half_dim`, `pair = offsets % half_dim`). These scalar operations produce scattered memory access patterns that the DMA engines cannot pipeline efficiently.

4. [code] Contiguous inner dimension.

   The tensor has at least one contiguous dimension (stride=1) that can be encoded as a block-pointer axis with `order=(1, 0)`. Verify the inner dimension is contiguous in memory before raising block-pointer dimensionality.

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
