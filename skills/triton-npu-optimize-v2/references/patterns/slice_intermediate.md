# Intermediate Slice Processing Pattern

## Summary

Use this pattern when intermediate tensors, not core algorithm shape, are the main reason a kernel exceeds Unified Buffer (UB) capacity.

The strategy is to process computation in bounded slices that fit UB, apply the same arithmetic per slice, and reassemble results without changing end-to-end semantics.

## Use When

- Intermediate tensors, rather than just inputs or outputs, are the main source of UB pressure.
- The overall algorithm is still reasonable, but staged slice processing is needed to keep temporary values within on-chip memory limits.
- The kernel repeatedly performs elementwise or fused updates where temporaries have the same shape as the main accumulator.
- You can partition one or more axes into independent chunks with predictable boundaries.

## Avoid When

- UB pressure is low and slicing would only add loop/control overhead.
- The operation has strong cross-slice dependencies that would require complex synchronization or change reduction order.
- The main bottleneck is transfer layout, launch geometry, or scalar index control rather than temporary footprint.
- A simpler structural rewrite (for example better tiling or fusion split) removes UB pressure more directly.

## Signals

### Code

- Full-tensor temporaries (broadcasted scales, masks, updates) stay live alongside inputs/outputs in the hot path.
- UB-related failures or near-limit configurations appear when block size grows.
- Arithmetic itself is straightforward, but live tensor count per program is too high.

### Profile

- Performance is unstable across tile sizes due to memory-pressure cliffs.
- Candidate kernels regress sharply when enabling larger blocks that should otherwise help arithmetic efficiency.

## Repairs

### Slice the temporary-heavy axis

Choose an axis where slices are independent and process one slice at a time so live intermediates stay bounded.

### Keep per-slice math identical

Apply exactly the same formula on each slice as the unsliced version to preserve semantics and simplify validation.

### Minimize slice bookkeeping

Precompute slice offsets and reuse shape/mask metadata so slicing does not become a new scalar bottleneck.

### Reassemble with deterministic placement

Use predictable `insert_slice` placement back into the destination tensor so each slice writes a disjoint region.

### Simplified code sketch

```python
# Unsliced intent: acc = acc * alpha[:, None] + update
num_slices = tl.cdiv(BLOCK_M, SUB_M)
for s in range(num_slices):
    off_m = s * SUB_M
    acc_s = tl.extract_slice(acc, [off_m, 0], [SUB_M, BLOCK_N], [1, 1])
    upd_s = tl.extract_slice(update, [off_m, 0], [SUB_M, BLOCK_N], [1, 1])
    alp_s = tl.extract_slice(alpha, [off_m], [SUB_M], [1])
    out_s = acc_s * alp_s[:, None] + upd_s
    acc = tl.insert_slice(acc, out_s, [off_m, 0], [SUB_M, BLOCK_N], [1, 1])
```

## Synthesized Guidance

- Use this pattern when UB capacity, not compute throughput, is the immediate blocker.
- Start with the smallest slicing change that eliminates overflow, then retune tile size and pipeline depth.
- Prefer stable, coarse slices over too many tiny slices; over-slicing often trades UB pressure for control overhead.
- If slicing fixes capacity but not latency, move next to transfer-shape or scalar-control patterns.

## Related Patterns

- `tiling`
- `software-pipeline`
- `slice_coalesce`
- `scalar-latency-traps`

## What To Verify After Applying

- Numerical equivalence with the unsliced formulation (including boundary slices).
- No UB overflow or memory-pressure failures on representative largest cases.
- Parent-vs-child benchmark comparison to ensure slicing overhead is acceptable.
- Profile confirmation that the previous memory-pressure cliff is removed.
