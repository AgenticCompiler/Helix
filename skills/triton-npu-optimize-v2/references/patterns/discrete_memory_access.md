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

## NPUKernelBench field inventory

**Scan date:** 2026-05-08. **Tree:** `workspace/NPUKernelBench_level_1_2_triton`.

This inventory lists operator workspaces whose `opt-round-*/attempts.md` files linked this card under pattern triage supporting evidence. Citation means the round considered the pattern, not that every hypothesis succeeded. For outcomes, read each operator `opt-note.md` and the linked `summary.md` / `attempts.md` for the cited rounds.

**Operator workspaces (deduped):**

- `18_Index`

## NPUKernelBench round narratives (pilot: `18_Index`, 2026-05-08, log-backed)

*Operator: **`18_Index`**. Tree: `workspace/NPUKernelBench_level_1_2_triton/`. Five-field template per `skills/triton-npu-kernel-bench-logs/SKILL.md`.*

### `18_Index`

**`opt-round-1` (parent `baseline`)** — `18_Index/opt-round-1/attempts.md`

- **Kernel / round / parent:** `18_Index` / `opt-round-1` / baseline.
- **Pre-change scenario:** Baseline `index_select` hot path loaded one index per output element and decoded up to 4D coordinates (`//`, `%`) per lane, while representative workloads are contiguous along `inner_size`.
- **Change:** Reframed the kernel as contiguous row-copy work from flattened `[outer, axis, inner]` views; initial `(selected_row, inner_block)` launch exceeded Ascend `coreDim`, then repaired to one program per selected row with an inner block loop.
- **Evidence:** Correctness passed after repair; `compare-perf` vs baseline reported **Avg +74.8%**, **Geomean 11.03x**, **Total 41.17x** in `attempts.md`; promoted as best branch.
- **Interpretation:** This card's staging pattern applies directly: replace per-lane discrete address reconstruction with contiguous spans plus local selection/looping under launch-cap constraints.

