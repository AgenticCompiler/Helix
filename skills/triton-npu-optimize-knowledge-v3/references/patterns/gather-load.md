# NPU Gather Load Optimization Pattern

## Summary

Optimize gather-like kernels by reshaping index-heavy scattered reads into load patterns closer to contiguous copy work. On Ascend NPU, gather performance often improves when hot paths reduce per-element index decoding and minimize high-width index traffic.

This card focuses on gather-specific load shaping (index dtype, axis specialization, row/span mapping), not generic tiling.

## Problem Description

Direct gather-style global-memory access can underperform on Ascend NPU because discrete index-driven reads are expensive compared with contiguous movement patterns.

## Use When

- The operation is semantically gather/index-select and gather loads dominate latency.
- Dominant cases have contiguous structure on at least one axis.
- Kernel time is inflated by index decode and address reconstruction.

## Avoid When

- Access is already contiguous and gather logic is not the bottleneck.
- Data movement is tiny and launch/setup overhead dominates.
- Core issue is dot/reduction structure or broad launch geometry.

## Signals

### Code

- Direct global loads driven by index vectors in the hot path.
- Repeated coordinate decode for rank/axis handling.
- `int64` index tensors where safe `int32` fast paths exist.

### Profile

- Gather kernel dominates one representative case.
- Scalar ratio remains high after basic cleanup.

## Optimization Strategy

1. Add `int32` index fast paths when axis bounds allow.
2. Specialize by dominant rank/axis cases.
3. Map work to contiguous spans where semantics permit.
4. Keep robust fallback paths for noncontiguous regimes.
5. Validate parent-vs-parent after each specialization.

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

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Direct discrete global-memory gather
    val = tl.load(x_ptr + idx * stride_x, mask=mask)
    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

### After Optimization (NPU-oriented)

```python
@triton.jit
def pick_kernel(
    x_ptr, idx_ptr, y_ptr,
    stride_x, stride_idx, stride_y,
    M: tl.constexpr, N: tl.constexpr
):
    pid = tl.program_id(0)
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    # Stage contiguous source span, then gather locally.
    x_shared = tl.load(x_ptr + rm * stride_x)
    val = tl.gather(x_shared, idx, 0)
    tl.store(y_ptr + rn * stride_y, val, mask=mask)
```

## Common Repairs

### Index dtype narrowing

Use `int32` indices on fast paths when bounds guarantee safety.

### Rank/axis specialization

Split dominant gather regimes into dedicated kernels instead of forcing one generic path.

### Row/span remap

When output selection aligns with contiguous inner spans, prefer contiguous movement plus local indexing.

### Launch-shape repair

After remapping to contiguous spans, adjust grid/program mapping to stay balanced and valid.

## Failure Modes And Anti-signals

- Narrowing indices without guardrails breaks large-axis correctness.
- Generic-only kernels persist even after strong case-specific evidence.
- One fast-path specialization regresses broader shape mix without fallback dispatch.
- Main gains actually come from store/layout cleanup rather than gather-load shape.

## Risks

- Extra dispatch complexity.
- Specialized paths drift if fallback parity tests are weak.
- Over-broad index conversion introduces helper-op overhead.

## What To Verify After Applying

- Correctness across all indexed shapes and boundary axis sizes.
- Parent-vs-child performance on identical benchmark mix.
- Dominant gather case improves without unacceptable regressions elsewhere.
- Dispatch guards for dtype/shape paths are explicit and correct.

## Related Patterns

- `discrete_memory_access`
- `layout-store-and-block-pointers`
- `scalar-latency-traps`
- `program-multiple-rows`
