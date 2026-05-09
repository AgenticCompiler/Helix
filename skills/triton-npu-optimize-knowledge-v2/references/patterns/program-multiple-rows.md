# Program Multiple Rows Pattern

## Summary

Map multiple logical rows to one Triton program (`BLOCK_M > 1`) to amortize per-program overhead and improve vector utilization in row-wise kernels.

This pattern targets **program granularity**: fewer, heavier programs doing more row work each launch. It is often the first strong lever after basic correctness/layout stabilization.

## Use When

- Kernel is row-structured (row reductions, row-wise fused epilogues, row-major transforms).
- Current launch maps one row per program and profiling shows many thin programs or scalar-heavy overhead.
- Problem size has enough rows to amortize wider per-program row bundles.
- Inner dimension streaming over `N` can stay single-pass while widening row count.

## Avoid When

- Row count is tiny; wider row bundles add overhead without amortization.
- Increasing `BLOCK_M` forces extra full data passes or unstable numeric behavior.
- Gains are dominated by unrelated bottlenecks (layout/store shape, compile hints, or scalar decode elsewhere).

## Signals

### Code

- `pid` maps directly to single-row ownership.
- Per-row pointer/control setup repeated for each program.
- Hot loops already tile inner dimension (`BLOCK_N`) but row axis remains under-batched.

### Profile / outcomes

- Scalar/control pressure or launch overhead remains high with one-row programs.
- Parent rounds show clear gains after moderate row-batching increases.
- Over-widening eventually flattens or regresses performance.

## Optimization Strategy

1. **Introduce `BLOCK_M > 1`** and remap row ownership to row blocks.
2. **Keep streaming semantics stable** (prefer one pass over inner dimension where possible).
3. **Tune `BLOCK_M` progressively** (small → medium → larger) with parent comparisons at each step.
4. **Add size/shape gates** when one global `BLOCK_M` does not fit all regimes.
5. **Compose with adjacent levers** (column tiles, exact-path dispatch, fallback paths) only after each row-batching step is validated.

## Common Repairs

### Row-block launch remap

Use block-row indexing (`rows = pid * BLOCK_M + arange`) and mask tails safely.

### Regime-gated row batching

Apply different `BLOCK_M` for tiny/medium/large regimes (or by dtype/head width/seq length) when one setting regresses part of the suite.

### Multi-path composition

Merge row batching with existing exact/fulltile or specialization paths only after each component is independently validated.

### Inner-dimension co-tuning

If wider rows increase pressure, co-tune inner tile (`BLOCK_N` or equivalent) to preserve single-pass behavior.

## Failure Modes And Anti-signals

- **Monotonicity assumption fails**: larger `BLOCK_M` can regress due to register/L2/occupancy pressure.
- **Ungated widening** hurts tiny-shape or short-sequence regimes.
- **Second full pass introduced** accidentally while widening rows.
- **Forced PMR everywhere**: some operators/regimes need different primary levers; row batching is not universal.

## Risks

- Wider row blocks can increase temporary footprint and scheduling pressure.
- Complex dispatch for gated `BLOCK_M` raises maintenance burden.
- Cross-path composition can hide regressions without strict parent comparisons.

## What To Verify After Applying

- Correctness on boundary rows and mask tails.
- Parent-vs-child performance per representative regime, not just baseline headline.
- Launch/program count reductions are real for target shapes.
- No unintended extra global passes were introduced.
- Gated paths route as intended for tiny vs large regimes.

## Related Patterns

- `grid-flatten-and-ub-buffering`: reshape logical-to-physical mapping when launch topology remains the bottleneck.
- `tiling`: co-tune inner dimension and footprint after row batching.
- `parallel`: orthogonal compute-branch concurrency inside a program.
- `layout-store-and-block-pointers`: fix transfer shape when row batching alone plateaus.
