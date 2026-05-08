# Slice Coalescing Pattern

## Summary

Use this pattern when scatter/gather-like kernels are dominated by random global-memory access and poor transfer coalescing.

The core idea is to move data in contiguous blocks whenever possible, use Unified Buffer (UB) slices as short-lived staging, and pay scattered access only on the unavoidable side (either read or write, but not both in the hot path).

## Use When

- Scatter or gather style data movement dominates, and batching work in UB could replace many random global accesses with fewer contiguous transfers.
- The kernel resembles token rearrangement, sparse reordering, or other index-based movement where access direction determines whether reads or writes should be coalesced.
- The operation has a stable block structure where contiguous chunk staging can be repeated predictably.
- Profiling suggests data movement shape (not arithmetic complexity) is the primary bottleneck.

## Avoid When

- Accesses are already mostly contiguous and coalesced; extra slicing would add overhead without reducing random traffic.
- UB pressure is already near limit and additional staging would force overly small tiles or expensive synchronization.
- Index paths are highly irregular with weak locality, so coalescing opportunities are minimal.
- The dominant issue is scalar control, launch geometry, or reduction structure rather than transfer shape.

## Signals

### Code

- Tight loops perform repeated elementwise scattered loads and scattered stores in the same path.
- The kernel can naturally batch contiguous rows/chunks but currently handles tokens one by one.
- UB-local assembly/disassembly could shift randomness to only one side of the transfer.

### Profile

- Transfer-heavy hotspots dominate while compute utilization remains modest.
- Performance scales poorly as index randomness increases, even when arithmetic work stays similar.
- Improvements appear when contiguous chunk size increases, indicating coalescing sensitivity.

## Repairs

### Choose one-sided randomness

Reframe the kernel so one side is coalesced:

- contiguous load + scattered store (`extract_slice`-style decomposition), or
- scattered load + contiguous store (`insert_slice`-style assembly).

### Stage in UB by chunk

Process block-sized slices in UB, then emit fewer larger transfers. Keep staging lifetime short to avoid UB congestion.

### Match slice granularity to locality

Tune chunk dimensions so each staged slice has meaningful contiguous transfer while still fitting UB with required masks/index vectors.

### Keep index math outside the innermost transfer loops

Precompute reusable offsets where possible so the hot copy path emphasizes data movement rather than repeated coordinate reconstruction.

### Simplified code sketch

```python
# Case A: contiguous read, scattered write.
tile = tl.load(src_ptr + src_base + tl.arange(0, BLOCK_D)[None, :])
for i in range(BLOCK_ROWS):
    token = tl.extract_slice(tile, [i, 0], [1, BLOCK_D], [1, 1])
    dst_row = tl.load(index_ptr + row_base + i)
    tl.store(dst_ptr + dst_row * stride_d + tl.arange(0, BLOCK_D), token)

# Case B: scattered read, contiguous write.
buf = tl.zeros((BLOCK_ROWS, BLOCK_D), dtype=tl.float32)
for i in range(BLOCK_ROWS):
    src_row = tl.load(index_ptr + row_base + i)
    row = tl.load(src_ptr + src_row * stride_d + tl.arange(0, BLOCK_D))
    buf = tl.insert_slice(buf, row, [i, 0], [1, BLOCK_D], [1, 1])
tl.store(dst_ptr + dst_base + tl.arange(0, BLOCK_ROWS)[:, None] * stride_d + tl.arange(0, BLOCK_D)[None, :], buf)
```

## Synthesized Guidance

- Try this pattern early for movement-dominated gather/scatter kernels after basic correctness is stable.
- First identify which side (read or write) has better contiguity, then coalesce that side aggressively and tolerate randomness on the other.
- Treat UB staging as a transfer-shape optimization, not a generic “more buffering is always better” rule.
- If gains plateau quickly or regress with larger slices, switch focus to launch mapping or scalar/index cleanup patterns.

## Related Patterns

- `discrete_memory_access`
- `gather-load`
- `grid-flatten-and-ub-buffering`
- `tiling`
- `scalar-latency-traps`

## What To Verify After Applying

- Correctness across repeated indices, sparse/empty segments, and boundary chunks.
- Benchmark delta versus parent branch on representative random and semi-contiguous cases.
- Profile confirmation that random global transfers were reduced on the targeted side.
- UB usage and tile sizing remain stable without introducing new memory pressure regressions.
