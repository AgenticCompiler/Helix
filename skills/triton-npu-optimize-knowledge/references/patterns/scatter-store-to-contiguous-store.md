# Scatter Store To Contiguous Store Pattern

## Summary

When the kernel performs a scattered store through index-based addressing (e.g. `out[indices] = values`, `out[row_base + idx_tensor] = values`), replace the kernel-side scattered store with a contiguous sequential store, then perform the final rearrangement on the host side with a single library call (`scatter_`, `index_copy_`, or equivalent). This converts random global-memory writes into coalesced DMA writes in the kernel, plus one efficient host-side reorder operation.

## Use When

- The kernel writes output through a permutation/reorder index tensor (e.g. sorted indices, routing maps, shuffle indices, gather-to positions).
- Profiling shows **high MTE2 cycles%** in `[Pipe Distribution]` (e.g. MTE2 > 20%) relative to VECTOR, and **low MTE2–VECTOR overlap** (e.g. `%(MTE2&VECTOR/MTE2) < 10%`), indicating write latency is serializing execution.
- The scatter indices have high randomness — no regular stride, affine pattern, or block-uniform structure that the compiler could vectorize.
- The host-side scatter (`torch.scatter_`, `torch.index_copy_`) is a well-optimized library call that runs in negligible time relative to the kernel.
- The kernel is **store-bound**, not load-bound or compute-bound.
- The intermediate contiguous output buffer is already needed or can be allocated without doubling peak memory.

## Avoid When

- The scatter indices follow a regular affine pattern (e.g. reverse, stride, transpose) that could instead be handled by adjusting pointer math or block-pointer metadata — prefer `layout-materialization-elision` or `block-pointer-dimensionality` first.
- The host-side scatter dominates end-to-end time, making kernel-side scatter the lesser cost.
- The kernel already uses UB staging (`extract_slice` / `insert_slice`) for scatter/gather, and the UB-resident scatter approach is already performing well.
- The output tensor is too large for an intermediate contiguous buffer without exceeding memory budget.
- The scatter indices are computed inside the kernel and cannot be returned to the host without an additional kernel.

## Signals

### Code

Look for these patterns in the Triton kernel source:

- `tl.store(output_ptr + base + index_tensor, values, mask=mask)` — store address contains a dynamically loaded index tensor rather than a sequential offset.
- The kernel accepts an index/permutation tensor as input whose **sole purpose** is to compute the store destination address.
- The host wrapper allocates an intermediate output buffer, then applies `scatter_` / `index_copy_` / `__setitem__` with an index tensor after the kernel returns — or could do so if refactored.
- The store offset formula involves more than `program_id * stride + arange` — it involves a loaded tensor.

### Profile (from `report.txt`)

| Section | Signal | Threshold | Meaning |
|---------|--------|-----------|---------|
| `[Pipe Distribution]` | MTE2 cycles% | > VECTOR cycles% or > 20% | Memory write path is the dominant cost |
| `[Pipe Overlap Ratio]` | `%(VECTOR&MTE2/MTE2)` | < 10% | Compute and memory write are serialized |
| `[Pipe Overlap Ratio]` | `%(SCALAR&MTE2/MTE2)` | < 5% | Address generation blocks memory write |
| `[Pipe Overlap Ratio]` | `%(SCALAR&MTE2/SCALAR)` | < 5% | Address gen and MTE are not concurrent |
| `[Pipe Overlap Ratio]` | `%(SCALAR&VECTOR/SCALAR)` | < 5% | Address gen and compute are serialized |
| `[VECTOR Unit]` | Utilization avg | < 30% | VECTOR stalled waiting for scattered stores |

These signals together indicate a **store-bound kernel with serialized address generation, memory write, and compute** — the classic profile of scattered store overhead.

## Related Patterns

- `discrete_memory_access` — addresses the **load** side of the same problem (scattered reads → contiguous staging + gather).
- `slice_coalesce` — addresses scatter/gather within the kernel using `extract_slice`/`insert_slice`; this pattern is the alternative when UB-resident scatter is viable.
- `merge-adjacent-stores` — complementary: after converting to contiguous stores, ensure adjacent writes are merged.
- `compile_hint` — after converting to contiguous stores, add `tl.max_contiguous` / `tl.multiple_of` hints for further lowering gains.
- `layout-materialization-elision` — if the scatter is just a layout transform, consider folding it into the consumer rather than moving it to host.
- `kernel-to-host-offload` — the broader pattern; this is the store-specific specialization.

## What To Verify After Applying

- Verify the host-side scatter produces identical output values for all edge cases (masked lanes, boundary tiles, sentinel values like `-inf` or `0`).
- Verify the intermediate contiguous buffer does not cause OOM at the largest benchmark shape.
- Verify the kernel run time decrease is larger than the host-side scatter time increase.
- Verify `report.txt` shows reduced MTE2 cycles% and improved pipe overlap ratios after the rewrite.

---

## Detail

### Key Principle

On Ascend NPU, a scattered store such as:

```python
tl.store(output_ptr + row_base + index_tensor, values, mask=mask)
```

degrades into individual per-lane DMA transactions because each lane writes to a different, unpredictable address. The MTE2 engine cannot coalesce these writes, leading to serialized store completion and poor MTE2–VECTOR overlap.

The fix is to move the permutation out of the kernel entirely:
1. Kernel writes **contiguously** (sequential offset: `row_base + col_offset`).
2. Host side performs one library call (`scatter_`, `index_copy_`, etc.) to rearrange into the final layout.

This works because host-side scatter routines are heavily optimized — they may use a dedicated scatter DMA engine, a well-tuned tiled kernel, or parallelized CPU threads — and their cost is amortized over the entire tensor rather than paid per-lane inside the kernel.

### Recognition Flow

When reviewing a kernel, ask these questions in order to determine if this pattern applies:

1. **Does the kernel write output through an index tensor?**
   ```python
   # Signs: store address includes a loaded tensor, not just program_id + arange
   tl.store(out + base + idx_tensor, val)  # YES — candidate
   tl.store(out + row * stride + arange, val)  # NO — already contiguous
   ```

2. **Are the indices random (non-affine)?**
   - Affine indices (reverse, stride, broadcast) → use `block-pointer-dimensionality` or pointer math
   - Random indices (shuffle, sort, token routing) → continue to step 3

3. **Can the indices be passed out of the kernel to the host?**
   - If indices are loaded from memory (not computed inside kernel) → YES
   - If indices are computed inside kernel and cannot be stored out → NO, try `slice_coalesce` instead

4. **Is the host scatter cheaper than the kernel scatter?**
   - Large row/column dimensions (>512) → likely YES (more scattered stores eliminated)
   - Small dimensions → benchmark to confirm

5. **Does the intermediate buffer fit in memory?**
   - Same size as output → YES in most cases
   - If memory-constrained → consider in-place variants or UB staging

### General Template

```python
# === BEFORE: Scattered store in kernel ===

@triton.jit
def kernel_before(input_ptr, index_ptr, output_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    col = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    offset = pid * n_cols + col

    # Load permutation index (extra MTE2 transaction)
    indices = tl.load(index_ptr + offset, mask=mask)

    # Compute values
    values = compute(tl.load(input_ptr + offset, mask=mask))

    # Scattered store — MTE2 bottleneck
    dst = pid * n_cols + indices
    tl.store(output_ptr + dst, values, mask=mask)


# === AFTER: Contiguous store in kernel + host scatter ===

@triton.jit
def kernel_after(input_ptr, output_intermediate_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    col = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    offset = pid * n_cols + col

    values = compute(tl.load(input_ptr + offset, mask=mask))

    # Contiguous store — coalesced MTE2 write
    tl.store(output_intermediate_ptr + offset, values, mask=mask)


def host_wrapper(input_tensor, index_tensor):
    intermediate = torch.empty_like(input_tensor)
    grid = (input_tensor.shape[0], triton.cdiv(input_tensor.shape[1], BLOCK_SIZE))
    kernel_after[grid](input_tensor, intermediate, input_tensor.shape[1], BLOCK_SIZE=BLOCK_SIZE)

    # Host-side permutation
    output = torch.full_like(input_tensor, fill_value)
    output.scatter_(dim, index_tensor, intermediate)
    return output
```

### Code Transformation Checklist

When applying this transformation, follow these steps:

1. **Remove the index tensor input** from the kernel signature.
2. **Remove the `tl.load` of the index tensor** from the kernel body.
3. **Simplify the store offset** from `row_base + index_tensor` to `row_base + col_offset` (sequential).
4. **Add `tl.max_contiguous` / `tl.multiple_of`** hints on the simplified offset to signal alignment to the compiler.
5. **Rename the output pointer** to reflect it is now an intermediate buffer (e.g. `output_intermediate_ptr`).
6. **Add host-side scatter** after the kernel launch using the index tensor.
7. **Adjust grid and launch parameters** — the simplified kernel can often use larger BLOCK_SIZE since contiguous stores benefit more from wider tiles.

### Variant: Scattered Store on a Per-Program Single Value

A common special case: each program computes a single scalar and writes it to a position determined by a per-program index.

```python
# Before: scalar scatter
idx = tl.load(index_ptr + pid)
tl.store(output_ptr + idx, scalar_value)

# After: gather into contiguous buffer, then host scatter
# Option A: Write to contiguous location, host scatter
tl.store(output_intermediate_ptr + pid, scalar_value)
# Host: output.scatter_(0, indices, intermediate)
```

### When Scatter Order Does Not Matter

If the output has no ordering constraints (e.g. all non-masked positions receive valid values, masked positions receive a fill value), the host-side scatter can be replaced by a simpler operation:

```python
# Instead of scatter_:
output = torch.full_like(input_tensor, fill_value)
output.scatter_(dim, indices, intermediate)

# If fill_value is the same for all masked positions, consider:
output = intermediate.new_full(input_tensor.shape, fill_value)
output.view(-1)[indices.view(-1)] = intermediate.view(-1)  # May be faster
```

### Combined With Other Patterns

After applying this pattern, the kernel now has contiguous stores. This unlocks further optimizations:

1. **Compiler alignment hints** (`compile_hint`):
   ```python
   offset = tl.max_contiguous(tl.multiple_of(offset, BLOCK_SIZE), BLOCK_SIZE)
   ```

2. **Larger BLOCK_SIZE** (`tiling` + `autotune`):
   Contiguous stores benefit more from wider tiles than scattered stores because wider tiles mean wider coalesced DMA transactions.

3. **Fewer warps** (`autotune`):
   With less work per program (no index load, no address computation), fewer warps may suffice.

4. **Grid flattening** (`grid-flatten-and-ub-buffering`):
   Simplified contiguous kernels are easier to batch across rows.

5. **Remove boundary mask** (`exact-tile-no-boundary-fast-path`):
   If the simplified kernel's BLOCK_SIZE exactly divides the dimension, masks can be eliminated.

### Detection Signals In report.txt

When evaluating whether to apply this pattern, check these signals in `report.txt` in order:

| Priority | Signal | What It Tells You |
|----------|--------|-------------------|
| 1 | `[Pipe Distribution]` MTE2 cycles% is the largest bucket | Memory write is the bottleneck |
| 2 | `[Pipe Overlap Ratio]` `%(VECTOR&MTE2/MTE2) < 10%` | Compute cannot overlap with memory writes — serialized execution |
| 3 | `[Pipe Overlap Ratio]` `%(SCALAR&MTE2/MTE2) < 5%` | Address generation blocks memory write |
| 4 | `[Pipe Distribution]` SCALAR instr% > VECTOR instr% | More address computation than actual compute |
| 5 | `[VECTOR Unit]` Utilization avg < 30% | VECTOR stalled waiting for memory |
| 6 | `[SCALAR Instr Types]` dominated by ADD, MOV, CMP, JUMPCMP | Index arithmetic inflating scalar pipe |

If signals 1–3 are all present and the kernel code contains a scattered store through an index tensor, this pattern is highly likely to help.
