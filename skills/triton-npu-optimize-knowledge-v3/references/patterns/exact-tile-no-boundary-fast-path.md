# Exact-Tile No-Boundary Fast Path Pattern

## Summary

Split exact-tile hot paths from generic masked kernels when dispatch-time shape guards can prove there are no tail tiles. On Ascend NPU, this can remove conservative boundary checks, padding values, element masks, and control branches that lowering keeps even when a benchmark case is fully aligned.

This is a late-stage control-overhead cleanup. It should keep the original boundary-safe kernel as fallback.

## Problem Description

Many Triton kernels are written with masks or block-pointer `boundary_check` so they work for arbitrary shapes. For dominant exact-tile cases, those guards can be redundant, but the compiler may still lower them into extra compare/select/branch work. A shape-specialized aligned kernel can make the no-tail fact explicit.

## Use When

- The hot case has exact tile coverage, for example `M % BLOCK_M == 0` and `N % BLOCK_N == 0`.
- Dispatch code can guard the aligned path before launch.
- MLIR/LLVM still shows boundary checks, padding, masks, or control branches on the hot path.
- The old masked kernel can remain as fallback for tail or irregular shapes.
- The kernel is close enough that removing control overhead can matter.

## Avoid When

- The mask is semantic data logic, not only a boundary/tail guard.
- Exact-divisibility cannot be proven at dispatch.
- Tail-heavy shapes dominate the benchmark mix.
- The bottleneck is overwhelmingly random global memory, atomics, or large compute where boundary control is negligible.
- The new fast path would duplicate too much complex logic and become hard to audit.

## Signals

### Code

- `tl.load(..., mask=tail_mask, other=...)` where `tail_mask` only guards block edges.
- `tl.store(..., mask=tail_mask)` for full-tile hot cases.
- `tl.make_block_ptr` loads/stores with `boundary_check` on exact shapes.
- Fast path can be expressed by removing masks, padding, and boundary checks without changing address math.

### Profile

- The parent kernel is already structurally reasonable.
- The hot exact-tile case is close but still control-heavy in IR or profiler traces.
- Expected gain is bounded, often around small single-digit percent to low double-digit percent.

## Optimization Strategy

1. Identify the dominant exact-tile shape and the tile divisibility conditions.
2. Add a dispatch guard for those conditions.
3. Clone only the minimal kernel body needed for the aligned path.
4. Remove boundary-only masks, padding, and `boundary_check` from the aligned kernel.
5. Keep the original generic kernel for all non-exact cases.
6. Verify correctness against the fallback and compare parent-vs-child performance.

## Optimization Example

### Dispatch Split

```python
if M % BLOCK_M == 0 and N % BLOCK_N == 0:
    _kernel_aligned_no_boundary[grid](...)
else:
    _kernel_masked_fallback[grid](...)
```

### Boundary-Safe Generic Body

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
value = tl.load(src + offs_m[:, None] * stride_m + offs_n[None, :], mask=mask)
tl.store(dst + offs_m[:, None] * out_m + offs_n[None, :], value, mask=mask)
```

### Exact-Tile Fast Body

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
value = tl.load(src + offs_m[:, None] * stride_m + offs_n[None, :])
tl.store(dst + offs_m[:, None] * out_m + offs_n[None, :], value)
```

## Known Evidence

NPUKernelBench `20_Gather` used this for a rank-2 `dim=0` gather fast path. For `bf16 x=(5120,27648), dim=0, index=(2560,27648)`, the aligned/no-boundary split reduced about `4239us -> 3850us` (**~1.10x**). The remaining bottleneck was still random global-memory gather, so the win should be treated as control-overhead cleanup rather than a memory-pattern fix.

## Failure Modes And Anti-signals

- Removing masks that encode algorithm semantics instead of tails.
- Applying the aligned kernel to tail shapes because guards are incomplete.
- Large code duplication that makes fallback and fast path drift.
- Claiming a broad pattern win from one exact benchmark without checking shape coverage.
- Spending effort here before fixing bigger layout, tiling, or access-pattern issues.

## Risks

- Out-of-bounds reads/writes if guards do not exactly match tile assumptions.
- Narrow wins may not survive different shape mixes.
- Duplicated kernels can increase maintenance and autotune surface.

## What To Verify After Applying

- Fast path and fallback produce identical values on representative exact-tile shapes.
- Fallback still covers non-divisible shapes.
- The aligned kernel IR no longer contains the target boundary checks or masks.
- Parent-vs-child performance improves on the targeted hot case without broad regressions.

## Related Patterns

- `compile_hint`
- `gather-load`
- `layout-store-and-block-pointers`
- `padded_row_col_copy`
- `scalar-latency-traps`
