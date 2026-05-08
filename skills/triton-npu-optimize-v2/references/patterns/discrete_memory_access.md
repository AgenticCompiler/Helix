# Discrete Memory Access Staging Pattern

## Summary

When loading discrete indices, rather than using `tl.load` to load the
discrete set directly, use `tl.load` to load a continuous range first, then use `tl.gather` to select
the target values.

## Use When

- The central bottleneck is discrete memory access that semantically looks like `out = x[idx]`.
- Index-driven global loads dominate the hot path, and contiguous staging plus local selection is more plausible than direct scattered reads.

## Detail

This example shows how to load data efficiently for discrete-memory-access workloads.

### Operation

Implement the following Triton-style behavior:

```python
out = x[idx]
```

Inputs:

| Input | Shape |
|-------|-------|
| x     | (M,)  |
| idx   | (N,)  |

Output:

| Input | Shape |
|-------|-------|
| out   | (N,)  |

### Key Difference Summary

- GPU-style code reads discrete values directly from global memory.
- NPU-style code first stages data from global memory into shared memory, then selects the target values from the staged buffer.

### Detailed Difference

Code diff of NPU and CUDA

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
