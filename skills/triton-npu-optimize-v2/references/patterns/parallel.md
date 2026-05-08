# Dual Vector-Core Parallel Pattern

## Summary

Use `tl.parallel` to run independent vector-side work concurrently across the two vector cores in one AICore. This pattern helps when compute-side branches are independent and substantial enough to amortize parallel-control overhead.

The key constraint is independence: parallel branches must not require cross-branch ordering inside the loop body.

## Use When

- Two or more compute-side substeps are independent and currently executed sequentially.
- Candidate work is vector compute (type conversion, scaling, elementwise transforms), not shared-bandwidth loading.
- Branch work is large enough that `tl.parallel` overhead is small relative to useful work.

## Avoid When

- Branches share true data dependencies.
- Dominant bottleneck is memory movement or load bandwidth.
- Candidate kernels are too small/fine-grained to benefit from branch parallelization.

## Signals

### Code

- Sequential, independent compute phases on the same iteration.
- Natural split of work into separate tensor operands or independent transforms.

### Performance

- Vector-side compute takes meaningful time after major layout/tiling fixes.
- Parallel structure is expected to reduce serial vector phases, not memory stalls.

## Optimization Strategy

1. **Identify independent branches** inside the hot loop.
2. **Parallelize only compute-side sections** with `tl.parallel`.
3. **Keep branch boundaries clean** (no shared mutable intermediates without explicit ordering).
4. **Measure end-to-end effect**; keep only changes that improve parent metrics.

## Common Repairs

### Parallelize independent operand transforms

When two operand-side transforms are independent, split them across parallel branches before the dependent consumer (for example dot).

### Keep loads out of `tl.parallel`

Let memory-loading strategy be handled by layout/tiling patterns; reserve `tl.parallel` for compute work where vector-core concurrency is real.

### Increase branch work granularity

If parallel overhead dominates, move branch boundary outward so each branch does more useful computation.

### Simplified code sketch

```python
# Independent transforms before a shared consumer.
with tl.parallel():
    x_t = transform_x(x_tile)  # branch A
    y_t = transform_y(y_tile)  # branch B

# Join point: dependent operation remains sequential.
acc = tl.dot(x_t, y_t, acc)
```

## Failure Modes And Anti-signals

- **Parallelizing loads**: shared bandwidth limits gains and can add overhead.
- **Implicit dependencies** between branches cause correctness or ordering bugs.
- **Too-fine branching**: overhead outweighs useful work.
- **No parent improvement**: structural parallelism looks better but benchmark remains flat/regressive.

## Risks

- Harder reasoning about branch-local temporaries and ordering.
- More complex code paths can hinder maintainability.
- Benefit can be highly shape- and kernel-specific.

## What To Verify After Applying

- Correctness equivalence with sequential version.
- Branch independence assumptions hold across all dispatched regimes.
- End-to-end benchmark improves vs immediate parent.
- Profile indicates reduced serialized vector compute rather than unchanged memory stalls.

## Related Patterns

- `program-multiple-rows`: widen work per program before or after compute parallelism.
- `tiling`: establish efficient memory/compute granularity first.
- `software-pipeline`: overlap load/compute stages; combine only after branch independence is clear.
- `cache_use`: if bandwidth remains dominant, prefer reducing movement phases over adding compute parallel branches.
