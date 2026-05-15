# Discrete Memory Access Staging Pattern

## Summary

When loading through discrete indices, do not use `tl.load` to read the target values directly from global memory. Instead, use `tl.load` to stage a contiguous range into the Unified Buffer first, then use the fast on-chip indexing path (`tl.gather` or equivalent) to select the target values.

For fixed-channel AoS layouts such as `[N, 3]` coordinates, apply the same principle by passing a channel-first SoA buffer (`[3, N]`) into the kernel so each channel is loaded with contiguous `atom_idx` offsets instead of repeated stride-3 global loads.

## Use When

- The central bottleneck is discrete memory access that semantically looks like `out = x[idx]`.
- Index-driven global loads dominate the hot path, and contiguous staging plus local selection is more plausible than direct scattered reads.
- The gather source array is small or medium enough that contiguous staging in shared memory is plausible.
- The hot loop repeatedly reads fixed fields from AoS records with stride-C offsets, such as `[N, 3]` coordinates loaded as `atom_idx * 3 + channel`, and the input is reused enough to amortize wrapper-side SoA materialization.

## Avoid When

- The source range is too large to stage or transpose profitably for the active shape.
- Memory access patterns are already sequential and contiguous.
- The fixed field dimension is consumed as a whole and splitting it would require vector extraction.
- The rewrite would introduce unsupported Ascend tensor indexing such as `vec[0]` on a loaded vector/tile.

## Signals

### Code

- Code uses index arrays to access non-contiguous memory locations on the hot path.
- Channel-wise loads use stride-2/3/4 addressing in the hot vector path.
- Direct global-memory gather reads dominate more than the surrounding arithmetic.
- Attempts to coalesce fixed fields would require extracting scalar components from a vector.
- A small fixed set of indexed setup values is used only for scalar frame/basis initialization.

### Profile

- Direct discrete global-memory reads appear as the dominant cost in the hot path.
- MTE bandwidth utilization is low relative to the amount of data consumed, indicating poor coalescing.

## Related Patterns

- `constexpr-tile-discrete-access`
- `slice_coalesce`
- `tiling`

## What To Verify After Applying

- Verify the source array size is still reasonable for shared memory after the rewrite.
- Verify the kernel stages the source array contiguously before the indexed selection step.
- Verify boundary masking and semantic equivalence with the original gather behavior.

---

## Detail

This pattern implements the Triton-style behavior `out = x[idx]`.

Inputs:

| Input | Shape |
|-------|-------|
| x     | (M,)  |
| idx   | (N,)  |

Output:

| Input | Shape |
|-------|-------|
| out   | (N,)  |

### Key Principle

- GPU-style code reads discrete values directly from global memory.
- NPU-style code first stages data from global memory into shared memory, then selects the target values from the staged buffer.

### Code Transformation

```diff
@triton.jit
def pick_kernel(
        x_ptr,
        idx_ptr,
        y_ptr,
        stride_x,
        stride_idx,
        stride_y,
        M: tl.constexpr,
        N: tl.constexpr
):
    pid = tl.program_id(0)
+   rm = tl.arange(0, M)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

-   # GPU path
-   val = tl.load(x_ptr + idx * stride_x, mask=mask)  # Direct discrete global-memory access
+   # NPU path
+   x_shared = tl.load(x_ptr + rm * stride_x)  # [M] Stage the full range into shared memory
+   val = tl.gather(x_shared, idx, 0)  # Select target values from the shared-memory buffer

    tl.store(y_ptr + rn * stride_y, val, mask=mask)

```

### Detection Pattern (Before)

```python
# Problematic: Direct discrete global memory access on NPU
idx = tl.load(idx_ptr + rn * stride_idx)
val = tl.load(x_ptr + idx * stride_x)  # Discrete access pattern

# Problematic: Index-based memory access
indices = compute_indices()
data = tl.load(base_ptr + indices * stride)  # Scattered loading
```

## AoS Fixed-Channel Variant

Before:

```python
atom_idx = base + tl.arange(0, BLOCK_SIZE)
x0 = tl.load(coordinate_ptr + atom_idx * 3 + 0, mask=mask)
x1 = tl.load(coordinate_ptr + atom_idx * 3 + 1, mask=mask)
x2 = tl.load(coordinate_ptr + atom_idx * 3 + 2, mask=mask)
```

Wrapper-side layout staging:

```python
coordinate = coordinate.contiguous()
coordinate_soa = coordinate.t().contiguous()  # [3, N]
```

Kernel-side contiguous channel loads:

```python
coord0_ptr = coordinate_soa_ptr
coord1_ptr = coordinate_soa_ptr + N
coord2_ptr = coordinate_soa_ptr + N * 2

x0 = tl.load(coord0_ptr + atom_idx, mask=mask)
x1 = tl.load(coord1_ptr + atom_idx, mask=mask)
x2 = tl.load(coord2_ptr + atom_idx, mask=mask)
```

Keep tiny indexed setup loads scalar:

```python
idx0 = tl.load(frame_idx_ptr + frame_id * 3 + 0).to(tl.int32)
a0 = tl.load(coord0_ptr + idx0)
a1 = tl.load(coord1_ptr + idx0)
a2 = tl.load(coord2_ptr + idx0)
```

## Ascend Vector-Extract Guardrail

Ascend Triton lowering does not reliably support vector extraction patterns such as `vec[0]` or
tensor indexing to split a loaded vector/tile. Do not convert three scalar values into a vector only
to extract components. Prefer separate scalar loads for tiny fixed setup values, and separate
contiguous channel pointers for hot vector paths.
