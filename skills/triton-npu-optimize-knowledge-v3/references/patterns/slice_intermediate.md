# Intermediate Slice Processing Pattern

## Summary

Use staged slice processing when intermediate tensors (not algorithm shape itself) push a kernel over UB capacity.

Process bounded sub-slices that fit on-chip, apply identical math per slice, then reassemble results deterministically.

## Use When

- Intermediate tensors are the main source of UB pressure.
- Formula is sound, but full-size temporaries cannot coexist in UB.
- Elementwise/fused updates create multiple same-shape live tensors.
- One or more axes can be partitioned into independent chunks.

## Avoid When

- UB pressure is low and slicing only adds loop/control cost.
- Cross-slice dependencies would alter reduction order or semantics.
- Layout/launch/tiling rewrites can remove pressure more directly.

## Signals

### Code

- Broadcasted scales/masks/updates remain live with full-size accumulators.
- Larger tiles trigger UB overflow or instability.
- Arithmetic is simple, but live-footprint count is too high.

### Profile

- Performance cliffs appear at larger tiles due to memory pressure.
- Throughput does not scale with larger blocks as expected.

## Optimization Strategy

1. Choose a slice axis with independent chunk semantics.
2. Keep per-slice math exactly equivalent to unsliced formulation.
3. Minimize slice bookkeeping and reuse offsets/masks.
4. Reinsert each slice into deterministic non-overlapping destination ranges.

## Example

```python
# Unsliced intent: result = acc * alpha[:, None] + update
num_slices = 4
slice_size = BLOCK_M // num_slices

for i in range(num_slices):
    offset = i * slice_size
    acc_slice = tl.extract_slice(acc, (offset, 0), (slice_size, HEAD_DIM), (1, 1))
    alpha_slice = tl.extract_slice(alpha, [offset], [slice_size], [1])
    update_slice = tl.extract_slice(update, (offset, 0), (slice_size, HEAD_DIM), (1, 1))
    result_slice = acc_slice * alpha_slice[:, None] + update_slice
    acc = tl.insert_slice(acc, result_slice, (offset, 0), (slice_size, HEAD_DIM), (1, 1))
```

## Practical Notes

- Ascend UB is fixed-size per core (commonly around 192KB class), so intermediate live-range control is often mandatory.
- Slice ops are view-style structural tools; they are most effective when they prevent large temporary materialization.
- Start with the smallest slice change that removes UB overflow, then retune tile/pipeline knobs.

## What To Verify After Applying

- Numerical equivalence on full and boundary slices.
- No UB overflow on largest representative cases.
- Parent-vs-child benchmark confirms slicing overhead is acceptable.
- Profile shows the previous memory-pressure cliff is removed.

## Related Patterns

- `tiling`
- `software-pipeline`
- `slice_coalesce`
- `scalar-latency-traps`
