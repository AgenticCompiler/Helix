# Cache And UB Reuse Pattern

## Summary

Use this pattern when performance is limited by memory hierarchy movement rather than pure compute. The goal is to reduce avoidable global-memory traffic, improve reuse in UB/L1/L2, and remove wrapper-level full-tensor copies that hide kernel wins.

In this card, UB means on-chip unified buffer. Practical sizing guidance remains important: L2 is shared and much larger than on-core buffers, while L1/UB are constrained and must be budgeted explicitly.

## Use When

- Profiling shows transfer-heavy behavior (for example MTE-heavy time, repeated tensor passes, weak locality).
- The algorithm/kernel structure is already stable, but data movement still dominates.
- Read-mostly tables/coefficients are consumed repeatedly and can be staged/reused.
- Host wrappers still do avoidable full-tensor materialization around the hot path.

## Avoid When

- The primary issue is still structure, launch geometry, or scalar control (`tiling`, `program-multiple-rows`, `scalar-latency-traps` first).
- Reuse expansion increases register pressure or reduces occupancy enough to negate movement savings.
- Larger transfer tiles exceed practical issue/latency sweet spots for the workload.

## Signals

### Code

- One-use intermediates are written then immediately reloaded by the next phase.
- Adjacent kernels repeatedly fetch the same coefficient tables.
- Wrapper code uses extra `clone`/copy/expand/cast around already-hot kernels.
- Duplicate probe/count passes are present even when prior metadata proves coverage.

### Profile

- Transfer-side counters dominate while cube/vector utilization is secondary.
- Parent rounds that reduce movement show clear gains even with small arithmetic changes.
- Aggressive cache/repack edits can catastrophically regress if traversal does not match.

## Optimization strategy

1. Remove redundant global intermediates first.
2. Stage and reuse read-mostly data across adjacent operations.
3. Eliminate duplicate global passes using already-available metadata.
4. Tune transfer width/tiles gradually and stop at first stable parent regression.
5. Always validate against the immediate parent, not only baseline.

## Common repairs

### Remove one-use intermediates

Fuse producer-consumer paths when an intermediate has only one immediate consumer.

### Keep read-mostly tables local

Stage reusable tables once per tile/program region and consume before eviction.

### Cut wrapper churn

Drop avoidable host-side tensor transforms that re-materialize data near the hot kernel.

### Reuse probe metadata

Skip second passes when first-pass probe/count already proves complete tile coverage.

### Bound tile widening

Increase transfer block size only while measured throughput improves; treat first parent regression as the practical boundary.

## Failure modes and anti-signals

- Over-aggressive reuse increases live ranges and lowers occupancy.
- Oversized transfer tiles regress despite fewer logical transactions.
- Wrapper copies remain and erase kernel-level gains.
- Repack/prepack without matching kernel traversal causes severe regressions.
- Cache-modifier or contiguous-fast-path edits help baseline yet still lose to the promoted parent.

## Risks

- Hardware hierarchy assumptions may not transfer cleanly across targets.
- Reuse-oriented rewrites can increase maintenance/debug complexity.
- Reduced memory movement can shift pressure into registers and harm concurrency.

## What To Verify After Applying

- Correctness across representative and boundary shapes.
- Parent-vs-child performance on identical harness settings.
- Profile evidence confirms fewer hot-path transfers/full passes.
- Wrapper-side helper churn does not reappear and offset gains.

## Related Patterns

- `layout-store-and-block-pointers`
- `program-multiple-rows`
- `tiling`
- `loop-invariant-hoisting`
