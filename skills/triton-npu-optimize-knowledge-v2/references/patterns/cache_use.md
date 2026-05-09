# Cache And UB Reuse Pattern

## Summary

Use this pattern when the kernel is memory-hierarchy bound: reduce avoidable global-memory movement, keep read-mostly data resident through adjacent phases, and remove wrapper-level full-tensor copies that hide kernel improvements.

**UB** here means the on-chip unified buffer. In practice, this card is about improving locality across **UB/L1/L2 and global memory** rather than changing math.

## Use When

- Profiling shows high transfer pressure (for example MTE-heavy time, many full passes over the same tensor, or obvious reloads between adjacent phases).
- The algorithm already has a stable structure, but hot tensors are still moved through extra intermediate buffers or thin wrapper kernels.
- Read-mostly tables or coefficients are consumed repeatedly (for example broadcast tables, mask tables, rope-style coefficients) and can be staged/reused.
- Wrapper code performs avoidable materialization (`clone`, `copy`, `expand + cast`, or duplicate count/probe launches) around an otherwise fast kernel.

## Avoid When

- The real bottleneck is scalar control, poor launch geometry, or missing specialization; use `scalar-latency-traps`, `program-multiple-rows`, or `tiling` first.
- Reuse expansion significantly increases register live ranges or reduces occupancy.
- A wider transfer tile exceeds the practical memory/issue sweet spot for the workload; bigger is not always faster.

## Signals

- Repeated read/write of intermediates that are consumed once by the next phase.
- Similar tensors crossing host wrappers and device kernels multiple times.
- Table-driven epilogues re-fetching the same coefficients in adjacent passes.
- Probe/count or metadata passes repeated even though a prior pass already proves full coverage.

## Optimization strategy

1. **Collapse redundant phases first**: remove unnecessary global intermediates and duplicate full-tensor wrapper moves.
2. **Stage and reuse hot data**: keep frequently reused tiles/tables resident across neighboring operations when correctness permits.
3. **Eliminate duplicate passes**: if an earlier probe/count is already complete for the current shape, skip the second global sweep.
4. **Tune transfer width carefully**: widen copy/transfer tiles while tracking occupancy and regression boundaries.
5. **Validate against parent**: compare to the immediate previous winner, not only to baseline.

## Common repairs

### Remove one-use intermediates

If an intermediate is produced only for one immediate consumer, compute it in the consumer kernel instead of writing/reading a full tensor between kernels.

### Keep read-mostly tables local

When tables are reused across adjacent sub-steps, stage them once per task tile and consume before eviction.

### Cut wrapper churn

Remove avoidable host-side tensor transforms around the hot kernel path (for example extra clone/copy or expanded mask materialization).

### Reuse probe metadata

If a probe pass already covers all tiles for a shape, skip launching a second counting phase.

### Bound tile widening

Increase copy/transfer block size only until measured throughput improves. Treat the first parent regression as the limit for that regime.

## Failure modes and anti-signals

- **Over-aggressive reuse** increases live ranges and loses occupancy, regressing despite fewer logical loads.
- **Oversized transfer tiles** saturate scheduling or issue resources and underperform a smaller tile.
- **Hidden wrapper copies** remain in the path, masking kernel-side wins.
- **Cross-card confusion**: forcing cache-focused rewrites before fixing launch/scalar/layout issues gives weak or noisy results.

## Risks

- Platform-specific hierarchy limits differ by target; hard-coded assumptions can break portability.
- Reuse-oriented fusion can increase code complexity and make correctness debugging harder.
- Memory-saving rewrites may shift pressure into registers and reduce concurrency.

## What To Verify After Applying

- Correctness across representative shapes and edge regimes.
- Parent-vs-child performance on the same harness.
- Profiling confirms fewer full-tensor passes or lower transfer pressure on the hot path.
- No new wrapper-side helper ops were introduced that offset kernel-level gains.

## Related Patterns

- `layout-store-and-block-pointers`: improve address shape before reuse tuning.
- `program-multiple-rows`: fix launch/program granularity before hierarchy tuning.
- `tiling`: choose transfer/reduction block shapes; then apply cache/UB reuse on top.
- `loop-invariant-hoisting`: remove repeated setup work that competes with memory bandwidth.
