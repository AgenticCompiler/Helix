# Kernel To Host Offload Pattern

## Summary

Move complex, non-contiguous, or permutation-heavy operations out of the Triton kernel and into host-side library calls (e.g. `torch.scatter_`, `torch.index_copy_`, `torch.gather`). Keep the kernel focused on contiguous, vector-friendly work that maps well to NPU hardware. This trades a small host-side cost for a large reduction in kernel complexity, SCALAR overhead, and MTE inefficiency.

## Use When

- The kernel performs a permutation, reorder, transpose, or scatter/gather operation that is **not naturally coalesced** on the NPU memory path.
- The same operation has a well-optimized host-side library primitive (e.g. `torch.scatter_`, `torch.index_select`, `torch.gather`, `torch.permute`).
- Profiling shows the kernel is dominated by non-compute work: **high SCALAR instr%** (`>50%` in `[Pipe Distribution]`) or **high MTE2 cycles%** relative to VECTOR, with **low pipe overlap ratios**.
- The kernel's code complexity (number of input pointers, condition branches, index computations) is inflated by the permutation logic, and removing it would simplify the kernel significantly.
- The host-side operation runs on a tensor that is already resident in device memory (no extra host-device transfer).
- Removing the permutation from the kernel unlocks other structural optimizations (larger BLOCK_SIZE, fewer warps, simpler masks).

## Avoid When

- The host-side operation would require a device-to-host copy that dominates end-to-end time.
- The permutation is a core part of a larger fused kernel, and splitting it would create an intermediate tensor that exceeds memory budget.
- The permutation is already efficiently handled within the kernel via `block-pointer-dimensionality` or `layout-materialization-elision`.
- The host-side operation has no optimized library path on the target platform.
- The kernel's non-contiguous work is small enough that UB staging (`extract_slice`/`insert_slice` via `slice_coalesce`) is more efficient than offloading.

## Signals

### Code

Look for these patterns in the Triton kernel and its host wrapper:

- The kernel signature includes index/permutation tensors (`index_ptr`, `routing_map_ptr`, `shuffle_indices_ptr`) whose **only purpose** is to compute load or store addresses — they are not consumed by any arithmetic.
- The kernel body contains `tl.store(out_ptr + row_base + index_tensor, values)` or `tl.load(inp_ptr + row_base + index_tensor)` — memory access addresses depend on dynamically loaded indices.
- The `tl.store` target offset involves more computation than a simple sequential `program_id * stride + arange`.
- Removing the permutation would eliminate one or more input pointers, corresponding `tl.load` calls, and complex address expressions.
- The host wrapper function could conceptually be refactored to split the computation: one kernel for the contiguous work, one host call for the rearrangement.

### Profile (from `report.txt`)

| Section | Signal | Threshold | Meaning |
|---------|--------|-----------|---------|
| `[Pipe Distribution]` | SCALAR instr% | > 50% | Kernel spends most instructions on address/control, not compute |
| `[Pipe Distribution]` | VECTOR instr% | < 15% while SCALAR > 50% | Compute is a minority of kernel work |
| `[Pipe Overlap Ratio]` | `%(SCALAR&MTE2/SCALAR)` | < 5% | Address generation and memory transfer are serialized |
| `[Pipe Overlap Ratio]` | `%(VECTOR&MTE2/MTE2)` | < 10% | Compute and memory write are serialized |
| `[Pipe Overlap Ratio]` | `%(SCALAR&VECTOR/SCALAR)` | < 5% | Address generation blocks compute |
| `[SCALAR Instr Types]` | ADD+MOV+CMP+JUMPCMP | dominates | Index arithmetic and conditional branching dominate |
| `[TRACE Events]` | arithmetic / total_events | < 20% | Little useful arithmetic per kernel launch |

These signals together indicate that **the kernel is spending most of its time on address computation and memory rearrangement rather than useful computation** — the profile of a kernel that would benefit from offloading non-contiguous work.

## Related Patterns

- `scatter-store-to-contiguous-store` — a specific instance of this pattern for scattered stores; if the detected problem is a scattered store, read that pattern first.
- `layout-materialization-elision` — removes unnecessary layout transforms entirely; this pattern moves them to the host when they cannot be eliminated.
- `discrete_memory_access` — handles the case where scattered loads stay in the kernel via UB staging; consider when offloading is not viable.
- `slice_coalesce` — handles scatter/gather within the kernel using slice operations; consider when UB-resident scatter is preferable to host offload.
- `algebraic-optimization` — reduces the number of kernel passes; offloading can similarly reduce per-kernel work.
- `program-multiple-rows` — increases per-program work density; offloading reduces per-program overhead from a different angle.

## What To Verify After Applying

- Verify the host-side operation time is measured and confirmed to be smaller than the kernel time saved.
- Verify the intermediate tensor (if any) does not cause OOM at the largest benchmark shape.
- Verify semantic equivalence: the host-side library call must produce identical results for all edge cases (masked elements, boundary conditions, sentinel values like `-inf`).
- Verify `report.txt` shows reduced SCALAR instr%, reduced MTE2 cycles%, and improved pipe overlap ratios.
- Verify kernel code complexity is measurably reduced: fewer input pointers, fewer condition branches, simpler store logic.

---

## Detail

### Key Principle

Triton kernels on Ascend NPU have a limited instruction window and benefit from a tight, vector-friendly hot path. When the kernel is burdened with complex index arithmetic, scattered memory access, or permutation logic, the SCALAR pipe inflates, MTE efficiency drops, and pipe overlap degrades.

The principle is: **keep the kernel simple and contiguous; let the host handle rearrangement**.

Ascend NPU kernels excel at:
- Contiguous elementwise operations (sequential offsets, no index indirection)
- Regular tiled reductions with fixed-stride access
- Structured `tl.dot` matmul loops with predictable memory patterns

Host-side library calls excel at:
- `scatter_` / `index_copy_` — optimized scatter-gather with dedicated DMA or parallel CPU threads
- `gather` / `index_select` — optimized gather from contiguous buffers
- `permute` / `transpose` — potentially zero-copy view or efficient copy kernel

### Recognition Flow

When reviewing a kernel, walk through this decision tree to decide whether to offload work:

```
Step 1: Identify non-contiguous operations
  Does the kernel do any of the following?
  ├── Store through an index tensor → scatter-store-to-contiguous-store specialization
  ├── Load through an index tensor → consider discrete_memory_access or offload
  ├── Permute/transpose data before store → consider layout-materialization-elision or offload
  ├── Compute complex index arithmetic per lane → consider offload
  └── None of the above → offload not applicable

Step 2: Can the operation be eliminated entirely?
  ├── Layout-only transform? → layout-materialization-elision (fold into consumer)
  └── Not eliminable → continue to step 3

Step 3: Does the operation stay in kernel or move to host?
  ├── Small working set, UB staging viable? → discrete_memory_access / slice_coalesce (stay in kernel)
  ├── Large working set, host library available? → THIS PATTERN (offload to host)
  └── Neither viable? → apply compile_hint, autotune for best-effort kernel optimization

Step 4: Verify offload is beneficial
  ├── Host operation time < kernel time saved? → apply
  ├── Intermediate buffer fits memory? → apply
  └── Either condition fails → keep in kernel, optimize differently
```

### General Template

```python
# === BEFORE: Permutation inside kernel ===

@triton.jit
def kernel_before(
    input_ptr,
    index_ptr,    # permutation index — only used for addressing
    output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols

    # Load permutation index (extra MTE2 transaction)
    idx = tl.load(index_ptr + pid * n_cols + col, mask=mask)

    # Load source data
    src = tl.load(input_ptr + pid * n_cols + col, mask=mask)

    # Compute
    result = compute(src)

    # Scattered store — address depends on idx
    tl.store(output_ptr + pid * n_cols + idx, result, mask=mask)


# === AFTER: Kernel does contiguous work, host does permutation ===

@triton.jit
def kernel_after(
    input_ptr,
    output_intermediate_ptr,  # no index_ptr — simplified signature
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    col = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    offset = pid * n_cols + col

    # Contiguous load
    src = tl.load(input_ptr + offset, mask=mask)

    # Same compute
    result = compute(src)

    # Contiguous store — coalesced
    offset = tl.max_contiguous(tl.multiple_of(offset, BLOCK_SIZE), BLOCK_SIZE)
    tl.store(output_intermediate_ptr + offset, result, mask=mask)


def host_wrapper(input_tensor, index_tensor):
    n_rows, n_cols = input_tensor.shape
    intermediate = torch.empty_like(input_tensor)
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))
    kernel_after[grid](input_tensor, intermediate, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    output = intermediate.new_full(input_tensor.shape, fill_value)
    output.scatter_(dim, index_tensor, intermediate)
    return output
```

### Code Transformation Checklist

When applying this pattern, follow these steps in order:

1. **Identify the permutation operation** to offload — is it a store, load, or both?
2. **Remove permutation-related kernel arguments** (index tensors, stride maps).
3. **Remove `tl.load` calls** that only load permutation indices.
4. **Simplify memory access offsets** from `base + index_tensor` to `base + sequential_offset`.
5. **Add compiler alignment hints** (`tl.max_contiguous`, `tl.multiple_of`) on the simplified offsets.
6. **Rename pointer arguments** to reflect they are now intermediate buffers.
7. **Add the host-side rearrangement** after the kernel launch.
8. **Re-tune launch parameters** — simplified kernels often benefit from larger BLOCK_SIZE and fewer warps.

### Common Offload Candidates

| Kernel Operation | Host Replacement | When To Offload |
|-----------------|------------------|-----------------|
| `tl.store(out + base + indices, val)` | `out.scatter_(dim, indices, val)` | Indices are random, row/col size is large |
| `tl.store(out + base + indices * stride, val)` | `out.index_copy_(dim, indices, val)` | Indices are 1D per dimension |
| Permute/transpose as final operation before store | Fold into consumer or `intermediate.permute(...)` | Layout transform is separable from compute |
| `tl.load(inp + base + indices)` for large source | Host-side `torch.gather(inp, dim, indices)` or UB staging | Source too large for UB staging |
| Complex multi-condition boolean mask assembly | Simplify to single `tl.where`; move mask logic to host if separable | Branch conditions inflate SCALAR pipe |
| Scatter-add / atomic scatter | Host-side `index_add_` or `scatter_add_` | Atomic contention on NPU is expensive |

### When Offloading Both Load and Store

Some kernels have permutation on both the input and output side. In such cases:

```python
# Pattern: Both ends are permuted — offload the worse one, stage the other
# Prefer: keep contiguous reads, offload scatter writes (writes are harder to coalesce)
# Alternative: contiguous read + contiguous write, host handles both permutations

@triton.jit
def kernel_straight_read_write(input_ptr, output_intermediate_ptr, ...):
    # Contiguous read
    src = tl.load(input_ptr + sequential_offset, mask=mask)
    result = compute(src)
    # Contiguous write
    tl.store(output_intermediate_ptr + sequential_offset, result, mask=mask)

# Host handles input gather and output scatter
def host_wrapper(input_tensor, gather_idx, scatter_idx):
    contiguous_input = torch.gather(input_tensor, dim, gather_idx)  # host gather
    intermediate = torch.empty_like(contiguous_input)
    kernel_straight_read_write[grid](contiguous_input, intermediate, ...)
    output.scatter_(dim, scatter_idx, intermediate)  # host scatter
```

### Interaction With Other Optimizations

After offloading the permutation to the host, the simplified kernel becomes a better target for:

1. **Larger BLOCK_SIZE** (`tiling` + `autotune`): Contiguous access patterns benefit more from wider tiles than scattered patterns.
2. **Compiler hints** (`compile_hint`): `tl.max_contiguous` and `tl.multiple_of` are effective on sequential offsets, but have limited effect on scattered ones.
3. **Fewer warps** (`autotune`): With less index computation and memory transactions per program, fewer warps may suffice.
4. **Grid flattening** (`grid-flatten-and-ub-buffering`): Simplified contiguous kernels with fewer input pointers are easier to batch across rows.
5. **Exact-tile fast path** (`exact-tile-no-boundary-fast-path`): Contiguous kernels with aligned dimensions can eliminate boundary masks entirely.

### Detection Checklist (from `report.txt`)

Use this ordered checklist to confirm the pattern applies:

| # | Check | Where in `report.txt` | Decision |
|---|-------|----------------------|----------|
| 1 | SCALAR instr% > 50%? | `[Pipe Distribution]` | If YES, kernel is address/control-heavy |
| 2 | VECTOR instr% < SCALAR instr%? | `[Pipe Distribution]` | If YES, compute is not the bottleneck |
| 3 | SCALAR&MTE2/SCALAR < 5%? | `[Pipe Overlap Ratio]` | If YES, address gen and memory are serialized |
| 4 | VECTOR&MTE2/MTE2 < 10%? | `[Pipe Overlap Ratio]` | If YES, compute and memory write are serialized |
| 5 | SCALAR Instr Types: ADD/MOV/CMP dominating? | `[SCALAR Instr Types]` | If YES, index arithmetic is the hot path |
| 6 | Does kernel code have index-based addressing? | Kernel source | If YES and 1-5 all YES, offload is likely beneficial |

If all checks pass, proceed with the code transformation checklist above.
