# Optimization Pattern Index

Use this index for fast matching: scan for the symptom that looks closest to your current kernel, then open the linked pattern card.

## `attention-cv-pipeline`
- Use when attention-like kernels already have a good score (`tl.dot`) stage, but mask/scale/softmax/dropout/post-processing still dominates.
- Strong signal: the simpler branch (for example feature-off path) still pays the same heavy epilogue machinery as the feature-on branch.
- Not a fit for pure elementwise kernels or kernels dominated by launch/transfer overhead outside the attention pipeline.
- Reference: `patterns/attention-cv-pipeline.md`.

## `autotune`
- Use when kernel structure is already stable and you only need to choose among a small set of plausible launch/meta configurations.
- Best for “no single tile/warp/stage wins all shapes” situations where bounded runtime search is cheaper than manual threshold logic.
- Do not start here if algorithm/layout is still wrong; autotune will only optimize a bad structure.
- Reference: `patterns/autotune.md`.

## `cache_use`
- Use when profiling shows transfer pressure (many full passes/reloads), even though compute structure is mostly okay.
- Typical wins come from eliminating redundant wrapper copies and keeping read-mostly data resident across adjacent steps.
- Skip if scalar control or launch geometry is still clearly the main bottleneck.
- Reference: `patterns/cache_use.md`.

## `classic-matmul`
- Use when your hot loop is logically `sum_k A*B` but implemented as manual reduction code with heavy scalar pointer/index work.
- This is the “re-express as tiled `tl.dot` first” card; it fixes structure before micro-tuning.
- Not for gather/scatter kernels or tiny-shape elementwise paths.
- Reference: `patterns/classic-matmul.md`.

## `compile_hint`
- Use late, after structure is already good, when compiler lowering still looks conservative due to missing alignment/contiguity facts.
- Typical hints: `dot_pad_only_k`, `multiple_of`, `max_contiguous` on branches where assumptions are truly guaranteed.
- Avoid if you are still fixing core tiling/layout/launch problems.
- Reference: `patterns/compile_hint.md`.

## `diagonal`
- Use when matrix-style kernels are tiled correctly but lose performance because many cores hit the same cache regions simultaneously.
- Core move: change block traversal order (row-major -> diagonal-like progression) while keeping tile math unchanged.
- Skip for small grids where mapping overhead is larger than locality benefit.
- Reference: `patterns/diagonal.md`.

## `discrete_memory_access`
- Use for index-driven reads (`out = x[idx]`) where per-element scattered global loads and index decode dominate.
- Pattern: stage contiguous spans first, then do local selection/gather from staged data.
- Not ideal when source ranges are too large to stage efficiently or access is already contiguous.
- Reference: `patterns/discrete_memory_access.md`.

## `gather-load`
- Use for gather/index-select kernels where one or two cases dominate and generic gather logic is too scalar-heavy.
- Common levers: int32 index fast paths, axis/rank specialization, and row/span remapping to contiguous inner movement.
- Don’t use as first lever when the real issue is unrelated reduction/matmul structure.
- Reference: `patterns/gather-load.md`.

## `grid-flatten-and-ub-buffering`
- Use when logical task count is far above core count and work is fragmented into too many tiny per-program transfers.
- Combines two moves: flatten task-to-core mapping and batch contiguous per-core movement through small UB slabs.
- Not a fit when continuity assumptions are weak or workload is already near core count.
- Reference: `patterns/grid-flatten-and-ub-buffering.md`.

## `layout-store-and-block-pointers`
- Use when latency is dominated by address/layout expression (flattened decode, transpose-at-store, many tiny stores), not by arithmetic.
- This card focuses on expressing memory movement in contiguous tile form, often with block pointers and merged stores.
- Skip when structural bottlenecks (algorithm, launch, scalar control) are still unresolved.
- Reference: `patterns/layout-store-and-block-pointers.md`.

## `loop-invariant-hoisting`
- Use when inner loops rebuild the same pointer bases/mask fragments every iteration and scalar setup dominates loop cost.
- Goal: hoist invariant pieces once, keep only loop-varying delta in the hot loop.
- Don’t force this if the expressions actually vary with loop index or if the main bottleneck is elsewhere.
- Reference: `patterns/loop-invariant-hoisting.md`.

## `parallel`
- Use when there are independent, compute-heavy vector branches currently executed serially in one loop body.
- Good fit for branchable vector transforms before a shared consumer (for example before dot), not for memory-loading bottlenecks.
- Avoid when branches have real dependencies or are too small to amortize parallel overhead.
- Reference: `patterns/parallel.md`.

## `program-multiple-rows`
- Use when row-wise kernels launch one row per program and spend too much overhead per program.
- Main lever: increase row ownership per program (`BLOCK_M > 1`) while preserving single-pass inner streaming.
- Not a fit for tiny row counts or regimes where larger row blocks force extra passes / unstable numerics.
- Reference: `patterns/program-multiple-rows.md`.

## `remove-implicit-transpose`
- Use in GEMM/Linear-style kernels when one operand is stored as `[N, K]` but consumed as `[K, N]` through stride tricks.
- The fix is to materialize preferred layout explicitly (often host-side transpose+contiguous) to avoid implicit transpose lowering overhead.
- Especially relevant when IR/profile show transpose-related staging and wait-flag symptoms.
- Reference: `patterns/remove-implicit-transpose.md`.

## `reorder-load`
- Use in memory-bound loops where independent loads are issued serially due to incidental ordering, not true data dependency.
- Core question: can a load be moved earlier without changing semantics, so memory-level parallelism increases?
- Avoid where ordering is semantically required or dependency graph is genuinely tight.
- Reference: `patterns/reorder-load.md`.

## `scalar-latency-traps`
- Use when vector-friendly kernels are dragged down by per-lane scalar control/index work (`//`, `%`, loop-carried pointers, degenerate `where`).
- Typical repairs include constexpr promotion, base-plus-offset addressing, mask-based tails, and safe int32 narrowing.
- Not a fit when bottleneck is clearly transfer/layout/launch rather than scalar control.
- Reference: `patterns/scalar-latency-traps.md`.

## `slice_coalesce`
- Use for movement-dominated scatter/gather paths where both sides are random and global transfers are poorly coalesced.
- Strategy: make one side contiguous (read or write), stage in UB chunks, and pay randomness only on the unavoidable side.
- Avoid when access is already mostly contiguous or index locality is too weak for useful staging.
- Reference: `patterns/slice_coalesce.md`.

## `slice_intermediate`
- Use when UB pressure comes from full-shape intermediates (not from the base algorithm itself) and causes capacity cliffs/overflows.
- Split the computation into independent slices, run same math per slice, and reassemble deterministically.
- Not ideal for strongly cross-slice dependent operations or low-pressure kernels where slicing overhead dominates.
- Reference: `patterns/slice_intermediate.md`.

## `software-pipeline`
- Use after tiled structure is already correct, when timeline still shows load-then-compute serialization and wait-heavy gaps.
- Typical recipe: block pointers + initial prefetch + steady-state “compute current while fetching next”.
- Skip if UB headroom is insufficient for multi-stage live tiles or loop trip count is too small.
- Reference: `patterns/software-pipeline.md`.

## `tiling`
- Use when per-program working set is too large for UB stability and widening tiles causes failures/regressions.
- Main move: introduce two-level tiling (`BLOCK` for scheduling, `SUB_BLOCK` for memory-safe inner processing).
- This is a footprint control lever; if footprint is already safe and overlap is the issue, switch to software pipeline.
- Reference: `patterns/tiling.md`.

## `vec-cmp`
- Use when explicit integer compare/mask logic is a hot-path scalar bottleneck outside compiler-fused load/store mask paths.
- Repair by moving compare path to a vector-friendly dtype where range guarantees semantic equivalence.
- Don’t apply blindly when value ranges are unsafe or compare path is not performance-critical.
- Reference: `patterns/vec-cmp.md`.
