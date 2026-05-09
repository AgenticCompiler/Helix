# Compiler Hint Pattern

## Summary

Use compile hints to communicate layout facts the compiler cannot safely infer from pointer math alone:

- `tl.compile_hint(x, "dot_pad_only_k")` for dot inputs when `M` and `N` are already aligned and only `K` needs padding behavior.
- `tl.multiple_of(ptr_or_idx, N)` to assert alignment.
- `tl.max_contiguous(ptr_or_idx, N)` to assert contiguous access width.

This pattern is a **late-stage refinement**. It works best after the main structure (layout, launch geometry, and kernel decomposition) is already stable.

## Use When

- The hot kernel is already structurally good, but profiling still shows conservative lowering or extra movement/scalar overhead.
- You can prove stronger alignment/contiguity facts than the code currently expresses.
- Dot-style kernels are stable and only need targeted lowering guidance.
- Parent comparisons show the kernel is close to the frontier and small IR/lowering shifts can matter.

## Avoid When

- The dominant issue is still structural (wrong tiling, launch geometry, fusion split, or scalarized algorithm shape).
- Alignment/contiguity assumptions are shape-conditional and not yet guarded by dispatch.
- Hints are being used as a substitute for fixing invalid pointer/index math.

## Signals

### Code

- Repeated contiguous slice loads/stores with masks that are mostly full tiles.
- Dot inputs where one axis (`K`) is the only true padding edge.
- Pointer/index expressions whose alignment is known from host-side contracts.

### Profile / outcome pattern

- Baseline is already strong, and hint-only rounds produce moderate but non-universal gains.
- Some hint rounds regress despite beating historical baselines, indicating parent-vs-parent sensitivity.

## Optimization Strategy

1. **Stabilize structure first** (layout/store shape, launch policy, and major fusion decisions).
2. **Add the smallest hint set** on the verified hot path only.
3. **Guard shape-dependent assumptions** with dispatch predicates.
4. **Compare against immediate parent** on the same harness.
5. **Keep hints that win on parent metrics; revert or narrow those that are flat/regressive.**

## Common Repairs

### `dot_pad_only_k` on true dot inputs

Use when `M` and `N` alignment is already satisfied by construction and only `K` tail behavior needs special handling.

### Alignment/contiguity hints on proven slices

Apply `multiple_of` and `max_contiguous` where host/kernel contracts guarantee those facts for the active branch.

### Hint scope narrowing

If a hint helps only one branch (dtype/shape/path), scope it to that branch instead of forcing it globally.

### Hint rollback after parent regression

When a hint round regresses the current best parent, remove or narrow the hint even if baseline-relative numbers still look positive.

### Simplified code sketch

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)

# Only K needs padding behavior for this dispatched branch.
tl.compile_hint(a, "dot_pad_only_k")
tl.compile_hint(b, "dot_pad_only_k")

# Pointer/index facts are guaranteed by dispatch contract.
tl.multiple_of(offs_k, 16)
tl.max_contiguous(offs_k, 16)
acc = tl.dot(a, b, acc)
```

## Failure Modes And Anti-signals

- **Invalid `multiple_of` assertion**: declared alignment is not always true (common in index-driven or masked small-shape paths).
- **Hint stacking on fragile fast paths**: additional hints can regress occupancy or scheduling even with unchanged arithmetic.
- **Baseline-only optimism**: a hint round can beat baseline yet still lose to the immediate parent.
- **Wrong sequencing**: applying hints before fixing structure yields noisy outcomes and weak transferability.

## Risks

- Overstated assumptions can cause subtle correctness or boundary behavior issues.
- Hint-heavy code becomes harder to audit and maintain.
- Benefits can be regime-specific and unstable across shape mixes.

## What To Verify After Applying

- All asserted alignment/contiguity facts are true for every dispatched regime.
- Correctness on boundary/tail shapes (especially where masks flip behavior).
- Parent-vs-child performance on the same benchmark mix.
- Profile/IR evidence supports the intended lowering improvement.
- No new regressions in small-shape or index-heavy fallback paths.

## Related Patterns

- `layout-store-and-block-pointers`: establish correct address/layout shape before hint tuning.
- `tiling`: fix tile geometry and footprint before hint refinement.
- `program-multiple-rows`: resolve launch granularity first, then hint mature paths.
- `scalar-latency-traps`: address scalarized control/address issues that hints alone cannot fix.
