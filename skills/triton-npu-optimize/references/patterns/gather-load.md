# NPU Gather Operation Optimization Pattern

## Summary

Stage gather-like input through contiguous loads before selecting indexed values so the kernel reduces expensive discrete global-memory reads on Ascend NPU.

## Problem Description

On Huawei Ascend NPU devices, direct discrete memory access patterns (gather operations) suffer from poor performance when accessing global memory. The NPU architecture favors contiguous memory access and has significantly higher bandwidth in shared memory compared to global memory for discrete access patterns.

## Optimization Strategy

Convert direct discrete global memory access into a two-phase approach:
1. **Contiguous data loading**: Load the entire source array from global memory to shared memory using contiguous access patterns
2. **Discrete selection**: Perform gather operations on the fast shared memory instead of slow global memory

### Key Principles

1. **Identify discrete access patterns**: Look for code that uses index arrays to access non-contiguous memory locations
2. **Leverage shared memory**: Utilize NPU's high-bandwidth shared memory for discrete operations
3. **Maintain semantic equivalence**: Ensure the logical result remains identical to the original implementation
4. **Consider memory footprint**: Only apply when the source array fits reasonably in shared memory

## Detection Pattern

Look for code patterns like:

```python
# Problematic: Direct discrete global memory access on NPU
idx = tl.load(idx_ptr + rn * stride_idx)
val = tl.load(x_ptr + idx * stride_x)  # Discrete access pattern

# Problematic: Index-based memory access
indices = compute_indices()
data = tl.load(base_ptr + indices * stride)  # Scattered loading
```

## Optimization Example

### Before Optimization (GPU-style)

```python
@triton.jit
def pick_kernel(
    x_ptr, idx_ptr, y_ptr,
    stride_x, stride_idx, stride_y,
    M: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, N)

    # Load indices
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Problem: Direct discrete global memory access (slow on NPU)
    val = tl.load(x_ptr + idx * stride_x, mask=mask)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

### After Optimization (NPU-optimized)

```python
@triton.jit
def pick_kernel(
    x_ptr, idx_ptr, y_ptr,
    stride_x, stride_idx, stride_y,
    M: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    rm = tl.arange(0, M)  # Full range for source array
    rn = tl.arange(0, N)  # Range for indices

    # Load indices
    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Optimization: Two-phase approach for NPU
    # 1. Contiguous load of entire source array to shared memory
    x_shared = tl.load(x_ptr + rm * stride_x)

    # 2. Discrete access on fast shared memory using tl.gather
    val = tl.gather(x_shared, idx, 0)

    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

## Use When

1. **Discrete access patterns**: When using index arrays to access non-contiguous memory
2. **Small to medium source arrays**: When the source array can fit in shared memory
3. **Performance-critical sections**: Where gather operations are bottleneck

## Avoid When

1. **Large source arrays**: When M is too large for shared memory capacity
2. **Already contiguous access**: When memory access patterns are already sequential
3. **GPU targets**: This optimization is NPU-specific and may not benefit GPU architectures
4. **Single-element access**: When only accessing a few discrete elements

## Implementation Checklist

- [ ] Identify discrete memory access patterns using index arrays
- [ ] Ensure source array size M is reasonable for shared memory
- [ ] Create full range for source array using `tl.arange(0, M)`
- [ ] Load entire source array contiguously to shared memory
- [ ] Use `tl.gather` on shared memory instead of direct memory access
- [ ] Maintain proper masking for boundary conditions
- [ ] Verify semantic equivalence with original implementation
