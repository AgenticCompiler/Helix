# Comparison Vectorization Pattern

## Summary

Rewrite hot-path integer compare-heavy logic into vector-friendly form so mask/selection code does not degrade into scalar bottlenecks on Ascend NPU.

This pattern primarily targets explicit compare-and-mask logic that feeds `tl.where` or similar control paths.
When range-safe, casted compare paths can lower to vector-friendly compare/cast instruction families (for example `vec_cmp`/`vec_cast`) rather than scalarized integer-compare chains. On Ascend, explicit integer compares frequently fall onto scalar-heavy control paths and can become disproportionate hotspots.

## Use When

- Explicit `i64`/`i32` comparisons dominate hot-path masking logic.
- Compare-heavy control flow appears outside the compiler's normal load/store mask fast path.
- Profile evidence shows scalar/control pressure around compare sections.

## Avoid When

- Comparisons are already in efficient inlined `tl.load`/`tl.store` masks and perform well.
- Operand ranges cannot safely support the planned cast strategy.
- Compare path is cold or non-critical.
- Cast overhead or semantics risk outweighs expected gain.

```python
# Often already good enough if this inlined form is lowering well:
x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
```

## Signals

### Code

- Integer compare masks are built explicitly and reused in `tl.where`/conditional assignments.
- Compare scaffolding repeats inside inner loops.
- Index-width choices (`idx32` vs `idx64`) materially change compare behavior and cost.

### Profile

- Scalar pressure remains high in otherwise vector-friendly kernels.
- Tiling/overlap tuning gives weak gains until compare path is simplified.

### Detection snippet

```python
valid = x_ids > y_ids  # hot i32/i64 compare
output = tl.where(valid, data, 0.0)
```

## Optimization Strategy

1. Identify explicit hot-path integer comparisons.
2. Convert operands to a vector-friendly compare dtype where semantically safe.
3. Keep mask computation localized and reusable.
4. Preserve or reuse compiler-fast load/store mask paths when already optimal.

## Reference Example

### Before

```python
@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    valid_indices = offsets < n_elements
    output = tl.where(valid_indices, x * 2, 0.0)
    tl.store(output_ptr + offsets, output)
```

### Mask hoisted before load (often still worth rewriting)

```python
# Before: comparison is outside tl.load and may still become scalar-heavy.
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)

# After: compare path is explicitly vectorized.
offsets_fp32 = tl.cast(offsets, tl.float32)
n_elements_fp32 = tl.cast(n_elements, tl.float32)
mask_fp32 = offsets_fp32 < n_elements_fp32
x = tl.load(x_ptr + offsets, mask=mask_fp32)
```

### After

```python
@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    offsets_fp32 = tl.cast(offsets, tl.float32)
    n_elements_fp32 = tl.cast(n_elements, tl.float32)
    valid_indices = offsets_fp32 < n_elements_fp32
    output = tl.where(valid_indices, x * 2, 0.0)
    tl.store(output_ptr + offsets, output)
```

## Practical Notes

- Some compare families are fundamentally better than others on Ascend (for example manual tie-resolution can outperform mask-argmax style forms in certain reductions).
- Index width is part of compare strategy: moving to wider index types without a real compare-path win can regress or flatten gains.
- Keep NaN behavior explicit when compare helpers (`tl.maximum`, `tl.minimum`) are involved; `propagate_nan` changes semantics on all platforms and also acts as a compiler instruction-selection hint on Ascend NPU.
- Do not blindly rewrite inlined `tl.load(..., mask=offsets < n)` / `tl.store(..., mask=...)` when current lowering is already optimal; focus first on explicit hot compare paths feeding `tl.where` or branch logic.
- **For compare helpers on hot paths, always benchmark both with and without `propagate_nan`.** On Ascend NPU the flag may significantly improve performance even when NaN-propagation semantics are irrelevant. Do not dismiss it as "only a correctness flag" without measurement.

## What To Verify After Applying

- Boolean semantics are preserved on representative and boundary cases.
- Cast range assumptions are valid.
- Hot-path scalar pressure decreases in profile.
- End-to-end benchmark gain remains after cast overhead.
- Any NaN propagation policy change is intentional and documented.

## Related Patterns

- `scalar-latency-traps`
- `loop-invariant-hoisting`
- `tiling`
