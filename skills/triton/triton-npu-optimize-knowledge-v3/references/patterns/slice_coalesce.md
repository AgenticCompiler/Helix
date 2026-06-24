# Slice Coalescing Pattern

## Summary

Use UB slice operations (`tl.extract_slice`, `tl.insert_slice`) to reshape scatter/gather-like transfers so random global-memory traffic is paid on only one side of the path.

This pattern is most useful when data movement dominates and the kernel can stage contiguous chunks in UB before emitting fewer larger transfers.

## Use When

- Scatter/gather movement dominates runtime and random global accesses are expensive.
- Work resembles token rearrangement, sparse index remap, or other index-directed copy paths.
- Access direction implies either reads or writes can be coalesced (even if not both).
- A stable block/chunk structure exists for UB staging.

## Avoid When

- Accesses are already mostly contiguous/coalesced.
- UB is too constrained for useful staging.
- Index locality is too weak for chunked staging to help.
- The primary bottleneck is not transfer shape.

## Signals

### Code

- Hot loops issue repeated scattered loads and scattered stores together.
- Per-token movement is handled one-by-one although chunk staging is possible.
- Index arithmetic is mixed directly into every transfer step.

### Profile

- Transfer-heavy timeline with modest arithmetic pressure.
- Strong sensitivity to randomness/locality despite similar compute counts.

## Optimization Strategy

1. Decide which side should remain random (read or write), and coalesce the other side.
2. Stage block-sized data in UB.
3. Use slice ops to assemble/disassemble in UB.
4. Emit fewer larger global transfers where possible.

## Reference patterns

```python
# Pattern A: contiguous read + scattered write.
data = tl.load(x_ptr + block_start + data_offset)
for i in range(BLOCK_SIZE):
    token = tl.extract_slice(data, [i, 0], [1, D], [1, 1])
    output_offset = D * tl.get_element(indices, (i,)) + tl.arange(0, D)[None, :]
    tl.store(output_ptr + output_offset, token)

# Pattern B: scattered read + contiguous write.
output_buffer = tl.full((BLOCK_SIZE, D), 0, dtype=x_ptr.type.element_ty)
for i in range(BLOCK_SIZE):
    token_idx = tl.get_element(indices, (i,))
    data_offset = token_idx * D + tl.arange(0, D)[None, :]
    token_data = tl.load(x_ptr + data_offset)
    output_buffer = tl.insert_slice(output_buffer, token_data, [i, 0], [1, D], [1, 1])
tl.store(output_ptr + block_offset, output_buffer)
```

## Practical Notes

- Ascend NPUs have fewer cores than large GPUs, so reducing launch/transfer fragmentation often matters more than increasing tiny-kernel count.
- Prefer coarse, stable chunking over many tiny slices; over-slicing adds control overhead.
- Keep index/offset setup reusable so transfer loops stay movement-focused.

## What To Verify After Applying

- Correctness on repeated indices, sparse groups, and boundary chunks.
- Parent-vs-child benchmark improvements on representative locality regimes.
- Profile evidence that random global traffic dropped on the intended side.
- UB usage remains stable without introducing new pressure cliffs.

## Related Patterns

- `discrete_memory_access`
- `gather-load`
- `grid-flatten-and-ub-buffering`
- `tiling`
- `scalar-latency-traps`
