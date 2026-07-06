# Prefer Single-Kernel Fusion Over Multi-Kernel Dispatch on Ascend NPU

## Summary

On Ascend NPU, multiple `@triton.jit` functions in the same Python module cause cross-kernel JIT interference: compiling one kernel degrades the generated code of another, even when the kernels are independent and never invoked together in a single call. Prefer fusing algorithmic variants into a single kernel using `tl.constexpr` gates over creating separate `@triton.jit` functions with host-side dispatch.

## Use When

- You are considering adding a second `@triton.jit` function for shape-specific or dtype-specific dispatch.
- Two algorithmic variants can be expressed as compile-time branches within a single kernel using `tl.constexpr` boolean gates.
- Profiling shows unexpected regressions on shapes that should use one kernel variant, after a second kernel was added to the module—even though the first kernel's source did not change.
- Re-running the same benchmark with the same multi-kernel code produces inconsistent geomean results, suggesting non-deterministic compiler behavior.

## Signals

### Code

- The module has one `@triton.jit` kernel and you plan to add a second for a different shape or dtype regime.
- Host-side dispatch selects between kernels based on runtime shape or dtype checks.

### Profile

- Adding a second kernel causes regressions on shapes that use only the first kernel, even though the first kernel's source and launch parameters are unchanged.
- Two successive benchmark runs of identical multi-kernel code give different geomean speedups.

### Benchmark

- A single-kernel approach using a suboptimal algorithm for some shapes still produces a net geomean win over a multi-kernel approach where each shape gets its optimal algorithm. The cross-kernel JIT penalty from the second kernel outweighs the per-shape benefit.

## Core Rewrite

Instead of two separate `@triton.jit` kernels with host dispatch:

```python
@triton.jit
def kernel_variant_a(..., BLOCK_SIZE: tl.constexpr):
    # algorithm A
    ...

@triton.jit
def kernel_variant_b(..., BLOCK_SIZE: tl.constexpr):
    # algorithm B
    ...

def wrapper(x, ...):
    if condition:
        kernel_variant_a[grid](x, ..., BLOCK_SIZE=bs_a)
    else:
        kernel_variant_b[grid](x, ..., BLOCK_SIZE=bs_b)
```

Use a single `@triton.jit` kernel with a `tl.constexpr` gate:

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, USE_VARIANT_A: tl.constexpr):
    if USE_VARIANT_A:
        # algorithm A
        ...
    else:
        # algorithm B
        ...

def wrapper(x, ...):
    if condition:
        kernel[grid](x, ..., BLOCK_SIZE=bs_a, USE_VARIANT_A=True)
    else:
        kernel[grid](x, ..., BLOCK_SIZE=bs_b, USE_VARIANT_A=False)
```

The `tl.constexpr` gate lets the compiler specialize each path, eliminating the unused branch. This preserves per-shape specialization while keeping a single `@triton.jit` function—avoiding cross-kernel JIT interference.

If the two variants are too different to coexist in one kernel (incompatible argument lists, fundamentally different loop structures, or different BLOCK_SIZE requirements that cannot share a `tl.constexpr` parameter), prefer a single best-effort kernel that handles all shapes over two specialized kernels. The JIT penalty from a second kernel often outweighs the per-shape specialization benefit.

## Avoid When

- The target is not Ascend NPU and cross-kernel JIT interference has not been observed on the platform.
- A single kernel already handles all shapes well—no dispatch is needed.
- The two variants have fundamentally incompatible requirements that cannot be unified with constexpr gating, AND the per-shape benefit of the second kernel is large enough (>3x on key cases) to justify the JIT interference cost. Verify with explicit A/B benchmarks comparing single-kernel against dual-kernel.

## What To Verify After Applying

- The single kernel produces correct results for all shapes and dtypes previously handled by separate kernels.
- Geomean speedup improves compared to the multi-kernel approach.
- Re-running the benchmark gives consistent geomean results—the non-deterministic JIT behavior is eliminated.
- No individual shape case regresses beyond what the single algorithm can achieve. Some per-shape specialization loss is expected and acceptable if net geomean improves.

## Related Patterns

- `size-gated-kernel-algorithm-dispatch`: The host-side shape-gating that selects between kernel variants. This pattern ensures those variants live in a single `@triton.jit` function to avoid JIT interference.
- `autotune`: After consolidating variants into a single kernel, autotune the BLOCK_SIZE and other parameters for the unified kernel.
- `eliminate-intermediate-precision-cast`: The bounding-step elimination that may precede multi-kernel dispatch. Fusing variants can avoid the need for separate precision-specific kernels.
- `broadcast-tensor-materialization-elision`: Uses the same `tl.constexpr` gate pattern to keep the non-broadcast path paying zero overhead.
