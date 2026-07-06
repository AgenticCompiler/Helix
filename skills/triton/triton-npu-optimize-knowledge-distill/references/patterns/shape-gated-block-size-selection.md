# Shape-Gated Block Size Selection for Column Reductions

## Summary

Choose BLOCK_SIZE for column-wise reduction loops from tiered problem-size thresholds during host dispatch. Larger block sizes cut loop iterations and unlock unmasked steady-state optimization. Overly aggressive sizes regress on mid-sized problems where few steady-state iterations fail to amortize the wider block cost. Use two or three size tiers with V-based thresholds to widen blocks only when enough full iterations exist.

## Use When

- A row-wise kernel loops over a column dimension with fixed BLOCK_SIZE steps.
- Larger BLOCK_SIZE reduces the loop trip count and enables unmasked inner-loop optimization.
- Mid-sized problems show regressions when BLOCK_SIZE is set too aggressively for all shapes.
- Host dispatch has access to the runtime column extent and can select BLOCK_SIZE before launch.
- Autotune is unavailable or unreliable on the target platform.

## Signals

### Code

- `BLOCK_SIZE` is selected from a fixed constant or simple rule (e.g. `4096 if V > 4096 else 2048`) instead of being tuned per-problem.
- The column loop iterates `V / BLOCK_SIZE` times — doubling BLOCK_SIZE halves the trip count.
- The loop processes multiple BLOCK_SIZE-worth of elements per iteration, making trip-count reduction more impactful.

### Profile

- Scalar overhead or mask evaluation time in the inner loop scales with iteration count.
- Increasing BLOCK_SIZE on large-V cases improves throughput but the same increase regresses mid-V cases.
- Mid-V regressions show the wider blocks consume more memory bandwidth with fewer total iterations to amortize the cost.

## Strategy

1. Identify the column extent as the key dispatch parameter.
2. Start with a baseline BLOCK_SIZE that works for all small/medium extents.
3. Add a tier with larger BLOCK_SIZE for large extents where at least 2 steady-state iterations remain.
4. Optionally add an intermediate tier for mid-range extents where a moderately larger block cuts iterations without triggering the too-few-iterations regression.
5. Measure geomean across all cases; reject any tier that introduces regressions.

Each tier should satisfy `extent / (iterations_per_step * BLOCK_SIZE) >= 2` so the steady-state path has at least 2 iterations.

```python
if V >= 65536:
    BLOCK_SIZE = min(V, 16384)
elif V >= 49152:
    BLOCK_SIZE = min(V, 12288)
else:
    BLOCK_SIZE = min(V, 8192)
```

## Avoid When

- The column extent is uniformly small and a single BLOCK_SIZE covers all cases.
- The kernel is not a row-wise reduction over a single column dimension.
- Autotune works reliably on the target platform and the search space is tractable.
- Adding tiers fragments the JIT cache without measurable benefit.
- The chosen BLOCK_SIZE would cause UB overflow — verify the working set fits on-chip memory.

## What To Verify After Applying

- Geomean speedup improves across all shape cases; no individual case regresses.
- Each tier has at least 2 steady-state iterations for the intended extent range.
- The BLOCK_SIZE value fits within UB budget for the kernel's live data.
- The dispatch logic selects the correct BLOCK_SIZE for all boundary extent values.

## Related Patterns

- `adaptive-launch-element-wise` — similar tiered dispatch approach, scoped to 1D element-wise kernels with flat/chunked dispatch.
- `ub-bounded-column-block-size-maximization` — maximizes column block up to UB capacity; shape-gating selects from the resulting candidates by problem size.
- `size-gated-kernel-algorithm-dispatch` — chooses between fundamentally different algorithms; this pattern selects block sizes within one algorithm.
