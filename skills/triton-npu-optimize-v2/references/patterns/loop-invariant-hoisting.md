# Loop-Invariant Hoisting Pattern

## Summary

Apply loop-invariant code motion (LICM) in Triton kernels: move work that does not depend on the loop induction variable out of the hot loop so each iteration executes only truly varying computation.

On Ascend NPU this most often reduces scalar/control pressure (address generation, bounds bookkeeping, repeated setup), and can improve end-to-end throughput when loop bodies are control-heavy.

## Use When

- A hot inner loop repeatedly rebuilds pointer bases, masks, or scalar setup terms.
- Profiling suggests scalar/control overhead is disproportionate to useful math.
- The kernel structure is already mostly correct, but loop body bookkeeping remains heavy.

## Avoid When

- Main bottleneck is still layout/store shape, launch geometry, or algorithm choice.
- Candidate expressions actually vary with loop index and cannot be safely hoisted.
- A rewrite would blur correctness-sensitive numeric paths without clear guardrails.

## Signals

### Code

- Repeated expressions of the form `base(pid, offs) + delta(k)` inside the loop.
- Mask parts that are invariant across iterations are rebuilt every iteration.
- Repeated per-iteration scalar setup for parameters that are launch-invariant.

### Profile / IR

- Scalar instruction mix dominates loop runtime.
- Loop body shows repeated arithmetic/index-cast chains with little true variation.

## Optimization Strategy

1. **Factor loop expressions** into invariant base and loop-varying delta.
2. **Hoist invariant base/mask fragments** outside the loop.
3. **Keep only delta updates inside** each iteration.
4. **Revalidate broadcasting orientation and masks** after hoisting.
5. **Compare against immediate parent**; keep only parent-positive changes.

## Common Repairs

### Pointer-base hoisting

Precompute pointer bases that depend on program offsets but not on loop index; only add loop-index delta per iteration.

### Partial mask hoisting

Precompute invariant mask terms once, then combine with loop-varying mask pieces inside the loop.

### Host-side invariant precompute

For launch-invariant scalar terms (for example optimizer coefficients), precompute on host and pass simplified constants to the kernel.

### Fusion-adjacent hoisting

When one phase computes values consumed immediately by the next phase, remove redundant invariant recomputation around that boundary.

### Simplified code sketch

```python
# Before: recomputes base and invariant mask each K iteration.
for k in tl.range(0, K, BLOCK_K):
    ptrs = base_ptr + (pid_m * stride_m + offs_m) + (k + offs_k) * stride_k
    mask = (offs_m < M) & ((k + offs_k) < K)
    x = tl.load(ptrs, mask=mask, other=0.0)

# After: hoist invariant pieces once.
row_base = base_ptr + (pid_m * stride_m + offs_m)
row_mask = offs_m < M
for k in tl.range(0, K, BLOCK_K):
    kk = k + offs_k
    x = tl.load(row_base + kk * stride_k, mask=row_mask & (kk < K), other=0.0)
```

## Failure Modes And Anti-signals

- **Over-hoisting**: moving expressions that actually depend on loop index introduces correctness bugs.
- **Orientation mistakes**: broadcast axes (`[:, None]` vs `[None, :]`) drift after refactor.
- **Flat or regressive impact**: LICM micro-passes after major structural wins can be near-noise or negative.
- **Wrong lever timing**: trimming invariants alone may not help until later specialization/dispatch passes.

## Risks

- Subtle semantic drift in masks or pointer arithmetic.
- Extra refactor complexity in already fragile kernels.
- Misattributing wins that actually come from adjacent structural changes.

## What To Verify After Applying

- Correctness on boundary/tail shapes and mask-sensitive cases.
- Parent-vs-child benchmark comparison on the same harness.
- Reduced scalar/control pressure in profiler or cleaner loop-body IR shape.
- No regressions in numerically sensitive or fallback paths.

## Related Patterns

- `compile_hint`: apply after LICM when alignment/contiguity facts remain implicit.
- `software-pipeline`: combine after loop body is simplified.
- `layout-store-and-block-pointers`: use when layout/transfer shape is still the dominant issue.
- `program-multiple-rows`: combine when launch granularity, not loop bookkeeping, is the bigger lever.
