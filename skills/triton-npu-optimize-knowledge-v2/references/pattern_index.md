# Optimization Pattern Index

Use this index for fast matching: scan for the symptom closest to your kernel, then open the linked pattern card.

## `algebraic-optimization`
- Use when the kernel repeats full scans or computes equivalent math in multiple passes (for example separate stats passes).
- Best fit for semantics-preserving rewrites that remove redundant work before touching tiles or launch policy.
- Also useful when broadcast-heavy logical ops can be reordered to reduce expanded work.
- Reference: `patterns/algebraic-optimization.md`.

## `attention-cv-pipeline`
- Use when attention-like kernels already have a good score (`tl.dot`) stage, but post-processing (mask/scale/softmax/dropout) dominates.
- Strong signal: simple feature-off branch still pays the same heavy epilogue pipeline as the feature-on branch.
- Fixes are usually staged: branch/dataflow cleanup first, hotspot tuning second.
- Reference: `patterns/attention-cv-pipeline.md`.

## `autotune`
- Use when structure is already stable and you need to choose among a small set of plausible launch/meta configs.
- Good for “no single tile/warp/stage wins all shapes” regimes with bounded search (`@triton.autotune`).
- Do not start here if algorithm/layout is still wrong.
- Reference: `patterns/autotune.md`.

## `cache_use`
- Use when profiling shows transfer pressure (reloads/full passes), even though compute structure is mostly correct.
- Typical wins come from removing wrapper-level copies and reusing read-mostly data across adjacent phases.
- Skip if scalar control or launch mapping is still the primary bottleneck.
- Reference: `patterns/cache_use.md`.

## `classic-matmul`
- Use when the hot loop is logically `sum_k A*B` but written as manual reduction with heavy scalar pointer/index work.
- This is the “rewrite to tiled `tl.dot`” structural card, before micro-optimizations.
- Not for gather/scatter-dominated or tiny elementwise workloads.
- Reference: `patterns/classic-matmul.md`.

## `compile_hint`
- Use late, after structure is good, when lowering still looks conservative due to missing alignment/contiguity facts.
- Typical hints: `dot_pad_only_k`, `multiple_of`, `max_contiguous` on branches where assumptions are provably true.
- Avoid if core tiling/layout/launch problems are still unresolved.
- Reference: `patterns/compile_hint.md`.

## `diagonal`
- Use when matrix-style kernels are tiled correctly but lose performance because many cores hit the same cache regions simultaneously.
- Core move: change block traversal order (row-major -> diagonal-like progression) while keeping tile math unchanged.
- Skip for small grids where remap overhead exceeds locality gains.
- Reference: `patterns/diagonal.md`.

## `discrete_memory_access`
- Use for index-driven reads (`out = x[idx]`) where per-element scattered global loads and decode arithmetic dominate.
- Pattern: stage contiguous spans first, then do local gather/selection from staged data.
- Not ideal when spans are too large to stage or accesses are already contiguous.
- Reference: `patterns/discrete_memory_access.md`.

## `gather-load`
- Use for gather/index-select kernels where generic gather logic is too scalar-heavy and one or two regimes dominate.
- Common levers: int32 index fast paths, axis specialization, contiguous row/span remapping.
- Don’t use as first lever when bottleneck is unrelated reduction/matmul structure.
- Reference: `patterns/gather-load.md`.

## `grid-flatten-and-ub-buffering`
- Use when logical task count is far above core count and work is fragmented into many tiny per-program transfers.
- Combines two moves: flatten task-to-core mapping and batch contiguous per-core transfers through small UB slabs.
- Not a fit when continuity assumptions are weak or workload is already near core count.
- Reference: `patterns/grid-flatten-and-ub-buffering.md`.

## `layout-store-and-block-pointers`
- Use when latency is dominated by address/layout expression (flattened decode, transpose-at-store, many tiny stores).
- Focus is expressing movement in contiguous tile form, often with block pointers and merged stores.
- Skip when structural bottlenecks (algorithm/launch/scalar control) are still primary.
- Reference: `patterns/layout-store-and-block-pointers.md`.

## `loop-invariant-hoisting`
- Use when inner loops rebuild the same pointer bases/masks every iteration and scalar setup dominates loop cost.
- Goal: hoist invariant pieces once and keep only loop-varying delta in the hot body.
- Don’t force this when expressions truly vary with loop index.
- Reference: `patterns/loop-invariant-hoisting.md`.

## `padded_row_col_copy`
- Use when a kernel spends time on row/column copy plus padding boundaries, especially when padded and unpadded regions are mixed.
- This pattern specializes copy paths so dense interior and boundary tails are handled with cheaper, clearer movement.
- Best after confirming the bottleneck is copy/pad structure, not arithmetic.
- Reference: `patterns/padded_row_col_copy.md`.

## `parallel`
- Use when independent compute-heavy vector branches are serialized in one loop body.
- Good for branchable vector transforms before a shared consumer; not for memory-bandwidth bottlenecks.
- Avoid when branches have true dependencies or are too fine-grained.
- Reference: `patterns/parallel.md`.

## `program-multiple-rows`
- Use when row-wise kernels launch one row per program and pay too much per-program overhead.
- Main lever: increase row ownership (`BLOCK_M > 1`) while preserving single-pass inner streaming.
- Not a fit for tiny row counts or regimes where larger row blocks force extra passes.
- Reference: `patterns/program-multiple-rows.md`.

## `remove-implicit-transpose`
- Use in GEMM/Linear-style kernels when one operand is stored as `[N, K]` but consumed as `[K, N]` via stride tricks.
- Fix is explicit preferred layout materialization (often host transpose+contiguous) to avoid implicit transpose lowering overhead.
- Especially relevant when IR/profile show transpose-related staging and wait-flag symptoms.
- Reference: `patterns/remove-implicit-transpose.md`.

## `reorder-load`
- Use in memory-bound loops where independent loads are issued serially due to incidental ordering, not true dependency.
- Core question: can a load move earlier without changing semantics to increase memory-level parallelism?
- Avoid where ordering is semantically required by real dependencies.
- Reference: `patterns/reorder-load.md`.

## `scalar-latency-traps`
- Use when vector-friendly kernels are dragged down by per-lane scalar control/index work (`//`, `%`, loop-carried pointers, degenerate `where`).
- Typical repairs: constexpr promotion, base-plus-offset addressing, mask tails, safe int32 narrowing.
- Not a fit when bottleneck is clearly transfer/layout/launch rather than scalar control.
- Reference: `patterns/scalar-latency-traps.md`.

## `slice_coalesce`
- Use for movement-dominated scatter/gather paths where both sides are random and global transfers are poorly coalesced.
- Strategy: make one side contiguous (read or write), stage in UB chunks, and pay randomness only on the unavoidable side.
- Avoid when access is already mostly contiguous or locality is too weak for useful staging.
- Reference: `patterns/slice_coalesce.md`.

## `slice_intermediate`
- Use when UB pressure comes from full-shape intermediates (not base algorithm itself) and causes capacity cliffs/overflows.
- Split computation into independent slices, run same math per slice, reassemble deterministically.
- Not ideal for strong cross-slice dependency or low-pressure kernels where slicing overhead dominates.
- Reference: `patterns/slice_intermediate.md`.

## `software-pipeline`
- Use after tiled structure is correct, when timeline still shows load-then-compute serialization and wait-heavy gaps.
- Typical recipe: block pointers + prefetch + steady-state “compute current while fetching next”.
- Skip if UB headroom is insufficient for multi-stage live tiles.
- Reference: `patterns/software-pipeline.md`.

## `tiling`
- Use when per-program working set is too large for UB stability and widening tiles causes failures/regressions.
- Main move: two-level tiling (`BLOCK` scheduling, `SUB_BLOCK` memory-safe inner processing).
- If footprint is already safe and overlap is now the issue, switch to software pipeline.
- Reference: `patterns/tiling.md`.

## `vec-cmp`
- Use when explicit integer compare/mask logic is a hot-path scalar bottleneck outside compiler-fused load/store mask paths.
- Repair by moving compare path to a vector-friendly dtype where range guarantees semantic equivalence.
- Don’t apply blindly when value ranges are unsafe or compare path is not performance-critical.
- Reference: `patterns/vec-cmp.md`.
