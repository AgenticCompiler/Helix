# Scalar Latency Trap Removal Pattern

## Summary

Use this pattern to remove scalarized control and index work from otherwise vector-friendly Ascend Triton kernels.

Apply it when the hot path is dominated by per-lane integer decode, degenerate masking, or loop-carried address dependencies. The goal is to keep vector lanes doing useful data work and move control decisions to compile-time or launch-time dispatch when safe.

## Use When

- Hot loops repeatedly do per-lane `//`, `%`, or wide-index arithmetic for coordinate decode.
- Runtime values are effectively shape constants but are still passed as normal arguments instead of `tl.constexpr`.
- Pointer updates rely on loop-carried `+=` recurrences instead of base-plus-offset addressing.
- `tl.where` is used with effectively uniform predicates (all lanes same decision, or only one exceptional lane).
- `int64` index/control math dominates even though value ranges are provably `int32`-safe.
- Long one-dimensional prefix-style vector flows (for example `tl.cumsum`) show scalar degradation in profile or IR.

## Avoid When

- The dominant bottleneck is memory traffic, layout/store shape, or launch geometry, not scalar control.
- Wraparound via `%` is part of required math semantics, not tail handling.
- Index ranges are not proven safe for `int32`.
- A replacement path changes reduction/precision behavior without an explicit correctness budget.
- A tuned vendor/library path already outperforms the candidate rewrite in representative workloads.

## Signals

### Code

- Repeated scalar-looking coordinate reconstruction around simple load/store or reduction kernels.
- Uniform or near-uniform predicates expressed as vector `tl.where` on every iteration.
- Invariant setup terms rebuilt inside inner loops.
- Frequent scalar guards in cases where exact-tile or no-padding regimes are common.

### Profile

- Scalar/control pipelines dominate while vector work remains underutilized.
- Flat or weak gains from tile-only tuning until control/index simplification is applied.
- Regressions when replacing backend-optimized paths despite seemingly simpler kernel logic.

### IR

- Repeated `index_cast`/arith chains inside loop bodies that do not need per-iteration recomputation.
- Long scalar dependence chains tied to address generation.

## Repairs

### Promote true constants to `tl.constexpr`

Move fixed shape and mode parameters to compile-time specialization keys when they are stable by launch contract.

### Replace loop-carried pointer recurrences

Use base-plus-offset addressing from stable bases each iteration to reduce scalar dependence chains.

### Remove modulo-based tails when masks are sufficient

Use contiguous offsets plus explicit masks for boundary handling when wraparound is not semantic.

### Eliminate degenerate vector conditionals

If a branch condition is tile-uniform or single-position special case, prefer split dispatch or targeted handling over full-lane `tl.where`.

### Narrow hot index/control math to `int32`

Cast once near construction/load and keep hot vector math in `int32`; cast back only at interfaces that require wider types.

### Split long prefix work into shorter vector-friendly stages

For long 1D prefix flows, use axis decomposition or staged prefix composition when it reduces scalarized inner work and preserves exact semantics.

### Simplified code sketch

```python
# Before: scalar-heavy decode and loop-carried pointer updates.
for k in tl.range(0, K, BLOCK_K):
    ptrs = a_ptr + ((pid * BLOCK_M + offs_m) * stride_m + (k + offs_k) % K) * stride_k
    vals = tl.load(ptrs)
    out = tl.where((k + offs_k) == special_k, vals * scale, vals)

# After: base-plus-offset addressing, mask tails, and avoid degenerate where.
base = a_ptr + (pid * BLOCK_M + offs_m) * stride_m
for k in tl.range(0, K, BLOCK_K):
    kk = k + offs_k
    ptrs = base + kk * stride_k
    vals = tl.load(ptrs, mask=kk < K, other=0.0)
    if special_lane_enabled:
        vals = tl.where(non_uniform_mask, vals * scale, vals)
```

## Synthesized Guidance

- Try this pattern early when the code shape clearly exposes scalar decode/control traps.
- Apply structural cleanup before micro-tuning: first remove scalar traps, then tune tiles and launch policy.
- Keep specialized fast paths guarded and preserve safe fallbacks; wins are often regime-specific.
- Treat regressions after simplification as an anti-signal that another pattern (layout, tiling, or launch geometry) is now primary.
- Stop when additional scalar-control cleanup gives flat results and profiles point elsewhere.

## Related Patterns

- `program-multiple-rows`
- `layout-store-and-block-pointers`
- `tiling`
- `loop-invariant-hoisting`

## What To Verify After Applying

- Correctness on boundary/tail cases and mixed-shape regimes.
- Parent-vs-child benchmark comparisons using the standard harness.
- Profile evidence that scalar/control pressure dropped in the intended hot path.
- Launch and specialization behavior (including cache cardinality) remains acceptable.
