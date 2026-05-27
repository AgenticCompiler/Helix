# Optimization Pattern Index

Use this index to quickly pick one or two candidate patterns before opening full cards.
Each section is retrieval-focused: when to try, what to look for, and what not to confuse it with.

## algebraic-optimization
- Use when the same tensor is scanned multiple times for related statistics or elementwise logic.
- Strong signal: mathematically equivalent rewrites can remove full passes, broadcasts, or redundant temporaries.
- Avoid when the issue is clearly tile/launch/layout first and no algebraic duplication exists.
- First move: apply one semantics-preserving rewrite, then verify correctness and parent-vs-child perf.

## attention-cv-pipeline
- Use for fused attention-like kernels where post-dot vector work dominates total latency.
- Strong signal: repeated mask/scale/state handling in hot loops with weak Cube-Vector overlap.
- Avoid when memory transfer is primary or architecture-gated options cannot be safely constrained.
- First move: stage by branch split, mask precompute, scale+mask fusion, then bounded hotspot tuning.

## autotune
- Use when kernel structure is reasonable but several plausible tile/launch tuples remain.
- Strong signal: manual config picks are unstable or clearly non-monotonic across shapes.
- Avoid when structural correctness, layout, or UB footprint issues are still unresolved.
- First move: build a bounded config set and key on true shape/runtime regimes.

## cache_use
- Use when locality/reuse is the bottleneck rather than arithmetic throughput.
- Strong signal: repeated reloads and weak cache/UB reuse despite otherwise valid kernel shape.
- Avoid when continuity assumptions are weak or occupancy/launch geometry is clearly dominant.
- First move: align traversal/tile footprint with cache hierarchy and validate with profiler evidence.

## classic-matmul
- Use when manual reduction loops are effectively matmul but not expressed as regular tiled `tl.dot`.
- Strong signal: scalar-heavy K-loop address logic around what should be standard GEMM structure.
- Avoid for pure elementwise, gather/scatter, or tiny shapes where matmul setup will not amortize.
- First move: rewrite into canonical tiled matmul skeleton, then tune tuple and dtype policy.

## compile_hint
- Use when code is structurally good but compiler lacks explicit alignment/contiguity knowledge.
- Strong signal: dot/load paths satisfy stronger assumptions than IR currently infers.
- Avoid as a first lever when structure/layout is still unstable or wrong.
- First move: add narrow hints (`multiple_of`, `max_contiguous`, dot-pad) and compare to parent.

## diagonal
- Use when large tiled matrix workloads still show conflict-heavy locality under row-major traversal.
- Strong signal: changing block order can improve reuse even with identical arithmetic/tiles.
- Avoid for small grids or kernels that still need foundational tiling/layout fixes.
- First move: add threshold-gated diagonal mapping with row-major fallback and benchmark both.

## discrete_memory_access
- Use when hot path is `out = x[idx]`-style discrete reads from global memory.
- Strong signal: direct indexed loads dominate and contiguous staging is feasible.
- Avoid when source spans are too large for useful local staging or access is already contiguous.
- First move: load contiguous span, then select via local gather/indexing.

## gather-load
- Use when gather-like reads are the central bottleneck and indices are explicit in hot code.
- Strong signal: random global loads dominate more than surrounding compute.
- Avoid when operation is not gather-dominated or staging cannot fit practical on-chip footprint.
- First move: convert to two-phase path (contiguous stage then indexed select).

## grid-flatten-and-ub-buffering
- Use when logical task count far exceeds cores and per-program movement is too fragmented.
- Strong signal: uneven per-core work, short bursts, and tiny row-wise transfers after rewrites.
- Avoid when workload already matches core scale or row continuity is too weak for UB batching.
- First move: flatten logical tasks to physical cores, then batch read/write slabs in UB.

## layout-store-and-block-pointers
- Use when transfer shape/layout expression, not math, is limiting latency.
- Strong signal: many tiny stores, transpose-at-store penalties, or flattened pointer decode chains.
- Strong signal (Ascend NPU profiling): when `extracted_bin_data/report.txt` exists under `opt-round-*` or operator workspace root, very low `OverlapRatio(VECTOR/CUBE & MTE2)` and `OverlapRatio(VECTOR/CUBE & MTE3)` near **0%**, with very high `OverlapRatio(MTE2 & MTE3)` near **100%** — compute-DMA serialization plus DMA-engine contention from flattened 1D scalar access.
- Avoid when continuity is unproven or bottleneck is still algorithm/launch/scalar control first.
- First move: merge contiguous stores, raise block-pointer dimensionality, and fix transpose ordering.

## loop-invariant-hoisting
- Use when hot loops repeatedly rebuild pointer bases, masks, or scalar setup terms.
- Strong signal: scalar/IR chains are repeated inside loop bodies with minimal true variation.
- Avoid when candidate expressions are loop-dependent or dominant issue is elsewhere.
- First move: split base+delta, hoist invariant base/mask fragments, validate tails and perf.

## padded_row_col_copy
- Use for constant-pad or regular bounds-copy kernels with heavy last-dim mask/control overhead.
- Strong signal: flat 1D decode with expensive div/mod and multi-phase tail masking.
- Avoid for gather/scatter-like index access or tail-unsafe shortcuts without proof.
- First move: switch to row-by-column tiling, hoist row base, keep strict tail-safe col masks.

## parallel
- Use when two compute-heavy vector branches are independent but currently serialized.
- Strong signal: natural A/B transform split before shared consumer (for example `tl.dot`).
- Avoid when candidate work is shared-bandwidth loads or has branch dependencies.
- First move: parallelize only independent compute branches with `tl.parallel`, then reprofile.

## program-multiple-rows
- Use for row-structured kernels where one-row-per-program causes high launch/control overhead.
- Strong signal: moderate `BLOCK_M` batching helps, but one-row mapping is still too thin.
- Avoid when batching forces second full passes or tiny-row regimes cannot amortize setup.
- First move: map `BLOCK_M > 1` rows per program with single-pass inner streaming over `N`.

## remove-implicit-transpose
- Use when math expects `[K,N]` but storage/access emulates transpose via `[N,K]` stride tricks.
- Strong signal: implicit-transpose lowering, wait-heavy matmul path, transform/reorder overhead.
- Avoid when host materialization cost is not amortized or layout contract cannot be maintained.
- First move: materialize target layout explicitly and index naturally in-kernel.

## reorder-load
- Use when independent loads are unnecessarily serialized behind dependent chains.
- Strong signal: memory-bound loops where load order, not arithmetic, appears to gate progress.
- Avoid if reordering crosses true dependencies or kernel is too small to benefit.
- First move: move independent loads earlier while preserving loop-carried correctness.

## scalar-latency-traps
- Use when vector-friendly kernels are slowed by scalar control/index/address anti-patterns.
- Strong signal: modulo tails, loop-carried pointer recurrences, degenerate `tl.where`, wide int control.
- Avoid when memory layout/launch geometry is clearly the primary bottleneck.
- First move: remove one trap at a time (`constexpr`, mask tails, int32-safe narrowing, etc).

## slice_coalesce
- Use for movement-dominated scatter/gather flows where one side can be made contiguous.
- Strong signal: both sides currently random, causing excessive fragmented global transactions.
- Avoid when access is already coalesced or UB staging would add overhead without locality gain.
- First move: choose one-sided randomness and use UB slice ops to batch transfers.

## slice_intermediate
- Use when intermediate temporaries push an otherwise-valid kernel over UB capacity.
- Strong signal: larger tiles fail from live-footprint cliffs rather than compute complexity.
- Avoid when cross-slice dependencies break semantics or slicing only adds control overhead.
- First move: slice temporary-heavy axis, keep per-slice math identical, reassemble deterministically.

## software-pipeline
- Use when a tiled loop still behaves like synchronous load-then-compute.
- Strong signal: wait-heavy gaps between memory engines and compute despite valid tiling.
- Avoid for tiny trip counts, UB-insufficient multi-stage tiles, or unresolved structural issues.
- First move: block pointers + prefetch + overlapped steady-state loop, then tune stage depth.

## tiling
- Use when per-program footprint is too large (UB pressure/overflow) despite correct high-level structure.
- Strong signal: widening blocks triggers memory faults or strong instability.
- Avoid when footprint is already safe and the next problem is overlap or algorithm shape.
- First move: keep outer scheduling block and introduce UB-safe inner sub-block loop.

## vec-cmp
- Use when explicit integer compare/mask logic becomes a scalar hotspot on Ascend.
- Strong signal: `i32/i64` comparisons feeding `tl.where` dominate control-heavy hot paths.
- Avoid when inlined load/store mask lowering is already efficient and semantics/range risks are high.
- First move: rewrite hot explicit compares to safe vector-friendly casted compare paths.
