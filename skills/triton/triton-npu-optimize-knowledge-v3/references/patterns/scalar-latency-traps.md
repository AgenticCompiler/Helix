# Scalar Latency Trap Removal Pattern

## Summary

Remove scalarizing control and index constructs that force vector-friendly kernels into avoidable scalar work and long dependency chains.

Use this pattern early as trap elimination: simplify scalar/control structure first, then tune tiling, launch geometry, and pipeline depth on top of a cleaner kernel.

## Use When

- Runtime shape constants are passed as normal arguments instead of `tl.constexpr`.
- Hot loops rely on loop-carried pointer `+=` recurrences.
- `%` is used for tail handling where masks would preserve semantics.
- `tl.where` handles effectively uniform predicates or single-lane exceptions.
- Hot control/index math stays in `int64` despite proven `int32`-safe range.
- Long one-dimensional prefix flows (for example `tl.cumsum`) show scalar degradation.

## Avoid When

- Bottleneck is memory layout, store shape, or launch geometry rather than scalar control.
- Wraparound `%` semantics are mathematically required.
- `int32` safety cannot be proven.
- Candidate rewrite changes numerical behavior without explicit correctness budget.

## Signals

### Code

- Repeated coordinate decode (`//`, `%`, wide-index arithmetic) inside inner loops.
- Invariant setup rebuilt every iteration.
- Degenerate lane predicates expressed as full-vector conditionals.

### Profile / IR

- Scalar/control pipelines dominate.
- Long address-generation dependence chains in loop body.
- Limited gain from tile-only tuning before scalar cleanup.

## Repair Catalogue

### Promote true constants to `tl.constexpr`

```python
@triton.jit
def kernel(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
```

### Replace loop-carried pointer recurrences

```python
for i in tl.range(0, K, BLOCK_K):
    ptrs = base + (i + offs_k) * stride_k + offs_n
    vals = tl.load(ptrs, mask=(i + offs_k) < K)
```

### Remove modulo tails when masks are enough

```python
offs = block_start + tl.arange(0, BLOCK)
mask = offs < N
vals = tl.load(x + offs, mask=mask, other=0.0)
```

### Eliminate degenerate vector `tl.where`

If only one lane differs or predicate is tile-uniform, prefer split dispatch or targeted handling instead of whole-vector conditional work.

### Keep hot index/control math in `int32` when safe

Cast once near construction/load, run hot math in `int32`, and widen only where required by interface/semantics.

### Split long prefix flows

For long 1D prefix operations, consider staged decomposition (for example 2D tiling + block-prefix composition) while preserving exact prefix order.

## Practical guidance from bench syntheses

- Many attempts labeled "scalar cleanup" become noise when structural bottlenecks are elsewhere; stop when parent-vs-child evidence is flat.
- Scalar fixes pair best with explicit shape-specialized fast paths and stable launch geometry.
- Literal-heavy or "bit-trick" rewrites can regress lowering quality; treat them as hypotheses, not guaranteed wins.
- Preserve negative anchors: failed scalar tweaks are useful evidence that another pattern is primary.

## Risks

- `tl.constexpr` increases specialization cardinality and may affect compile/cache behavior.
- Incorrect modulo removal or mask assumptions can break boundary semantics.
- Unsafe narrowing to `int32` can overflow silently.
- Prefix decomposition must preserve exact ordering and numerics.

## What To Verify After Applying

- Record the exact trap removed and code location in `attempts.md`.
- Run correctness before trusting performance.
- Confirm benchmark gains with standard compare-perf parent-vs-child discipline.
- Re-check launch contracts and call sites when specialization/signatures change.

## Related Patterns

- `loop-invariant-hoisting`
- `tiling`
- `program-multiple-rows`
- `algebraic-optimization`
