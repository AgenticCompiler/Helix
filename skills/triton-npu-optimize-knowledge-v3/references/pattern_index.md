# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

### `autotune`

- Summary: **Autotune** here means Triton’s `@triton.autotune` decorator: runtime benchmarks a **small, bounded** set of launch/meta configurations and caches the fastest by key.
- Source: [autotune.md](patterns/autotune.md)

### `grid-flatten-and-ub-buffering`

- Summary: Use this pattern when latency is dominated by oversized logical grids, uneven per-core work, or tiny per-program transfers after gather/scatter-style rewrites.
- Source: [grid-flatten-and-ub-buffering.md](patterns/grid-flatten-and-ub-buffering.md)

## Generated Pattern Summaries

### `algebraic-optimization`

- Summary: Look for **semantics-preserving** rewrites that reduce **memory passes**, **redundant full scans**, or **dependency depth** before micro-tuning launch geometry.
- Source: [algebraic-optimization.md](patterns/algebraic-optimization.md)
- Use When:
  - The hot path performs **two or more full traversals** of the same data for statistics, normalization, or mergeable subexpressions.
  - Profiler or IR suggests **duplicate transfer-heavy** phases differing only by scalar statistics from the same tensor.
  - A manual transcendental expansion exists where a stable intrinsic can express the same semantics.
  - Elementwise **logical** ops (`logical_or`, `logical_and`, etc.) with broadcasting evaluate truth on expanded numeric tensors.
  - You need structural simplification **before** `tiling`, `software-pipeline`, or `autotune` retuning.
- Avoid When:
  - The bottleneck is clearly launch geometry / occupancy / UB footprint with no redundant algorithmic work (use `tiling` first).
  - A rewrite only changes expression spelling, without reducing pass count, dependency depth, or effective work.
  - Approximate formulas are introduced without strict correctness and parent-vs-child performance evidence.
  - Custom Triton fusion is attempted before proving a simpler reorder or rewrite is sound.
- Signals / Code:
  - Two loops (or kernels) with nearly identical `tl.load(x)` tiling over the same axis.
  - Recomputed closed-form subexpressions in the hot epilogue.
  - Manual transcendental forms (exp/divide compositions) in tight loops.
  - `broadcast_tensors(x, y)` then truth tests on the expanded numeric tensors.
- Signals / Profile:
  - Repeated transfer-dense stages that could be merged if math structure were reorganized.
  - `NotEqual` / `BroadcastTo` work scaling with broadcast-expanded size.
  - Plateau after launch/pipeline tweaks, suggesting structural math/dataflow remains the limiter.
- Signals / IR:
  - Repeated load or reduction structure around the same logical axis where one traversal could feed multiple accumulators.
  - Epilogue dependency chains that can be algebraically shortened.

### `attention-cv-pipeline`

- Summary: Use this pattern for fused attention-style kernels that have two stages:
- Source: [attention-cv-pipeline.md](patterns/attention-cv-pipeline.md)
- Use When:
  - A regular matrix score path already exists, but post-processing dominates total latency.
  - Mask conditions are recomputed repeatedly inside hot loops even though they depend only on host-known metadata (lengths, windows, causal mode).
  - Scale and mask are applied as two separate passes over the same score tile.
  - A simpler branch (optional feature disabled) is forced through the same heavy path as the feature-enabled branch.
  - The code keeps extra state format conversions only to match a particular exponential implementation choice.
  - Profile evidence shows the matrix stage is not the main limit; post-processing instruction count is.
- Avoid When:
  - The kernel is pure elementwise work with no meaningful matrix/post-processing split.
  - The dominant bottleneck is launch or transfer overhead outside this stage.
  - Correctness tolerance is extremely tight and cannot tolerate operation reordering or branch refactoring.
- Signals / Code:
  - Repeated mask index arithmetic appears inside the normalization loop.
  - Separate scale and mask passes both read/write the same score tensor.
  - Simpler semantic branches still route through generic fallback code.
  - Forward state uses one exponential-base convention while backward expects another.
- Signals / Profile:
  - Post-processing instruction count remains high after basic launch cleanup.
  - The same post-processing kernels dominate across many shape regimes.
  - Simpler branches still spend visible time in broadcast/cast preparation outside kernels.

### `autotune`

- Summary: **Autotune** here means Triton’s `@triton.autotune` decorator: runtime benchmarks a **small, bounded** set of launch/meta configurations and caches the fastest by key.
- Source: [autotune.md](patterns/autotune.md)
- Use When:
  - Kernel logic is stable and correct, but several plausible launch/meta tuples remain.
  - Hand-picked parameters are inconsistent across shape regimes.
  - You can define reliable `key=` dimensions so unrelated workloads do not share one cached winner.
  - You can compare against the immediate best parent, not only an old baseline.
- Avoid When:
  - Core algorithm/layout is still changing quickly.
  - Search space is too large relative to compile/first-run budget.
  - Launch contract ownership is unclear (same meta passed in both decorator and launch kwargs).
  - Config trials write to shared accumulating outputs without per-trial reset/isolation.
- Signals / Code:
  - Repeated manual edits of `BLOCK_*`, `num_warps`, `num_stages` with no robust winner.
  - Duplicate launch metadata between decorator configs and call site.
  - Key dimensions omit semantic/shape modes that change valid or optimal tuples.
- Signals / Profile:
  - Multiple nearby tuples perform similarly, and winner shifts by shape/mode bucket.
  - Parent branch still shows underutilization after structural optimizations.
  - Search overhead or compile churn becomes visible due to over-wide config grids.

### `cache_use`

- Summary: Use this pattern when performance is limited by memory hierarchy movement rather than pure compute. The goal is to reduce avoidable global-memory traffic, improve reuse in UB/L1/L2, and remove wrapper-level full-tensor copies that hide kernel wins.
- Source: [cache_use.md](patterns/cache_use.md)
- Use When:
  - Profiling shows transfer-heavy behavior (for example MTE-heavy time, repeated tensor passes, weak locality).
  - The algorithm/kernel structure is already stable, but data movement still dominates.
  - Read-mostly tables/coefficients are consumed repeatedly and can be staged/reused.
  - Host wrappers still do avoidable full-tensor materialization around the hot path.
- Avoid When:
  - The primary issue is still structure, launch geometry, or scalar control (`tiling`, `program-multiple-rows`, `scalar-latency-traps` first).
  - Reuse expansion increases register pressure or reduces occupancy enough to negate movement savings.
  - Larger transfer tiles exceed practical issue/latency sweet spots for the workload.
- Signals / Code:
  - One-use intermediates are written then immediately reloaded by the next phase.
  - Adjacent kernels repeatedly fetch the same coefficient tables.
  - Wrapper code uses extra `clone`/copy/expand/cast around already-hot kernels.
  - Duplicate probe/count passes are present even when prior metadata proves coverage.
- Signals / Profile:
  - Transfer-side counters dominate while cube/vector utilization is secondary.
  - Parent rounds that reduce movement show clear gains even with small arithmetic changes.
  - Aggressive cache/repack edits can catastrophically regress if traversal does not match.

### `classic-matmul`

- Summary: Rewrite a manual matmul or K-reduction hot loop into a regular tiled `tl.dot` matmul so the kernel shape matches what Ascend Triton lowers well.
- Source: [classic-matmul.md](patterns/classic-matmul.md)
- Use When:
  - The kernel computes an `M x N` output tile with regular reduction over `K`.
  - Current code is effectively `sum_k A[..., k] * B[..., k]`.
  - Profile/IR shows heavy scalar address/control overhead in the hot loop.
  - Partial pointer/layout fixes helped but the loop is still not a regular matmul skeleton.
  - Dtype- or shape-specialized dispatch is acceptable if one regime clearly benefits.
- Avoid When:
  - Purely elementwise kernels.
  - Gather/scatter-dominated kernels.
  - Very small shapes where tile setup does not amortize.
  - Workloads where the kernel is already a solid tiled matmul and remaining issues are tile/pipeline/hint details.

### `compile_hint`

- Summary: Use compile hints to communicate layout facts the compiler cannot always infer safely:
- Source: [compile_hint.md](patterns/compile_hint.md)
- Use When:
  - The kernel is structurally sound, but lowering still appears conservative.
  - You can prove stronger alignment/contiguity facts than current code expresses.
  - Dot kernels are stable and only need targeted lowering guidance.
  - Parent comparisons are close enough that IR/lowering improvements can matter.
- Avoid When:
  - Core bottleneck is still structural (wrong tiling, launch shape, decomposition).
  - Alignment/contiguity assumptions are shape-conditional but not dispatch-guarded.
  - Hints are being used to compensate for invalid pointer/index math.
- Signals / Code:
  - Dot inputs where only `K` is a true padding edge.
  - Mostly full-tile contiguous slice accesses with conservative masks.
  - Pointer/index expressions whose alignment is guaranteed by host contracts.
- Signals / Profile:
  - Strong parent kernel with small remaining inefficiencies.
  - Hint-only rounds produce mixed outcomes (some wins, some regressions), indicating sensitivity.

### `diagonal`

- Summary: When tiled matrix-style kernels are already structurally sound but still suffer cache contention, change traversal order rather than math. Diagonal/grouped block traversal reduces simultaneous pressure on the same cache regions and can improve effective L2 reuse.
- Source: [diagonal.md](patterns/diagonal.md)
- Use When:
  - Large matrix block grids still show locality/conflict issues after basic tiling is reasonable.
  - Many concurrent programs touch similar matrix regions under row-major/horizontal ordering.
  - Both `M` and `N` span enough blocks that traversal order materially changes reuse behavior.
- Avoid When:
  - Problem size is too small for traversal order to matter.
  - Kernel still lacks fundamental tile/layout correctness.
  - Mapping overhead outweighs locality benefit.
- Signals / Code:
  - Work assignment is row-major and causes synchronized reuse conflicts.
  - Kernel math is stable; only block-order decisions remain unsettled.
  - One operand footprint is large enough that naive ordering increases eviction/reload traffic.
- Signals / Profile:
  - Throughput varies significantly with block scheduling order despite identical arithmetic.
  - Memory contention or reuse loss persists after tile-size tuning.

### `discrete_memory_access`

- Summary: When the logical operation is index-driven (for example `out = x[idx]`), avoid direct per-element scattered global loads on the hot path. Stage contiguous source spans first, then select locally (for example with `tl.gather` from staged data).
- Source: [discrete_memory_access.md](patterns/discrete_memory_access.md)
- Use When:
  - The central bottleneck is discrete indexed access rather than arithmetic.
  - Index-driven global loads dominate runtime.
  - Contiguous staging plus local selection is feasible for the active shapes.
- Avoid When:
  - Source spans are too large to stage efficiently.
  - Access is already mostly contiguous and indexing is not the bottleneck.
  - The primary issue is launch geometry or decomposition rather than access shape.
- Signals / Code:
  - Hot loops repeatedly execute direct indexed global loads (`x[idx]` style).
  - Per-lane index decode (`//`, `%`, address reconstruction) dominates surrounding math.
  - One program could own contiguous rows/spans but current mapping is fully elementwise.

### `gather-load`

- Summary: Optimize gather-like kernels by reshaping index-heavy scattered reads into load patterns closer to contiguous copy work. On Ascend NPU, gather performance often improves when hot paths reduce per-element index decoding and minimize high-width index traffic.
- Source: [gather-load.md](patterns/gather-load.md)
- Use When:
  - The operation is semantically gather/index-select and gather loads dominate latency.
  - Dominant cases have contiguous structure on at least one axis.
  - Kernel time is inflated by index decode and address reconstruction.
- Avoid When:
  - Access is already contiguous and gather logic is not the bottleneck.
  - Data movement is tiny and launch/setup overhead dominates.
  - Core issue is dot/reduction structure or broad launch geometry.
- Signals / Code:
  - Direct global loads driven by index vectors in the hot path.
  - Repeated coordinate decode for rank/axis handling.
  - `int64` index tensors where safe `int32` fast paths exist.
- Signals / Profile:
  - Gather kernel dominates one representative case.
  - Scalar ratio remains high after basic cleanup.

### `grid-flatten-and-ub-buffering`

- Summary: Use this pattern when latency is dominated by oversized logical grids, uneven per-core work, or tiny per-program transfers after gather/scatter-style rewrites.
- Source: [grid-flatten-and-ub-buffering.md](patterns/grid-flatten-and-ub-buffering.md)
- Use When:
  - Logical task count is far larger than physical core count.
  - Batch/sequence partitioning causes visible load imbalance.
  - Programs still process tiny contiguous rows/chunks one-at-a-time.
  - Grid/index decode overhead is nontrivial.
- Avoid When:
  - Workload is already near physical core count.
  - Continuity is too weak for safe UB slab batching.
  - Main bottleneck is still algorithm structure, scalar traps, or base tiling.
- Signals / Code:
  - `TOTAL_TASKS >> NUM_CORES` style mapping.
  - Many thin launches or highly fragmented per-program work.
  - Flattened pid decode chains that can be replaced by direct multidimensional mapping.
- Signals / Profile:
  - Poor core utilization with short bursts.
  - Launch-side overhead remains high after first structural rewrites.

### `layout-store-and-block-pointers`

- Summary: Use this pattern when latency is limited by memory layout expression and transfer shape, not arithmetic complexity. The goal is to present movement as contiguous vector-friendly tiles instead of flattened scalarized address chains, transpose-at-store paths, or many tiny stores.
- Source: [layout-store-and-block-pointers.md](patterns/layout-store-and-block-pointers.md)
- Use When:
  - Adjacent destinations are written by many narrow `tl.store` operations.
  - Store order is effectively transposed relative to destination contiguity.
  - Contiguous multidimensional tensors are accessed through flattened 1D decode chains.
  - Inner dimensions are looped/pid-decoded even though they can be encoded in tile shape.
  - Dot paths use avoidable transpose/cast ordering.
- Avoid When:
  - Main bottleneck is still launch geometry, scalar control, or algorithm shape.
  - Continuity assumptions are weak or shape-conditional without dispatch guards.
  - Block-pointer metadata cannot be expressed robustly for all active regimes.
- Signals / Code:
  - Repeated scalar index arithmetic around otherwise simple movement.
  - Accumulators or intermediates are transposed only at final store.
  - Many small loads/stores where contiguous vectors are possible.
- Signals / Profile:
  - Transfer overhead remains high after first-order tiling.
  - Gains plateau until layout/store representation changes.

### `loop-invariant-hoisting`

- Summary: Apply loop-invariant code motion (LICM) in Triton kernels: move work that does not depend on the loop variable out of the hot loop so each iteration only executes truly varying computation.
- Source: [loop-invariant-hoisting.md](patterns/loop-invariant-hoisting.md)
- Use When:
  - Hot inner loops repeatedly rebuild pointer bases, masks, or scalar setup.
  - Profiling shows scalar/control overhead disproportionately high vs useful math.
  - Kernel structure is mostly correct, but loop bookkeeping remains heavy.
- Avoid When:
  - Main bottleneck is still layout/store shape, launch geometry, or algorithm selection.
  - Candidate expressions actually vary with loop index.
  - Refactor risks numerically sensitive semantics without clear validation.
- Signals / Code:
  - Repeated `base(pid, offs) + delta(k)` expressions in loop body.
  - Invariant mask fragments rebuilt each iteration.
  - Per-iteration scalar setup for launch-invariant terms.

### `padded_row_col_copy`

- Summary: Optimize constant-pad and similar bounded copy kernels by replacing a flat 1D traversal over `numel(out)` with an `out_rows x out_dim_last` structure:
- Source: [padded_row_col_copy.md](patterns/padded_row_col_copy.md)
- Use When:
  - The operator is constant pad or another regular per-axis bounds copy (not gather/scatter).
  - Baseline kernel heavily uses `//` and `%` per element to recover coordinates.
  - Profiling shows scalar/mask overhead concentrated on last-dim boundary handling.
- Avoid When:
  - The hot path is index-driven gather/scatter (use `gather-load` / `discrete_memory_access`).
  - Dynamic tile-kind branching inside the column loop cannot be validated on backend lowering.
  - Tail stores omit `col_mask` without proof for `out_dim_last % BLOCK_COLS != 0`.
- Signals / Code:
  - Single linear `offsets` path with repeated high-stride coordinate decode.
  - One global validity mask combines all dimensions for each lane.
  - Multi-phase column loops with different store masks for left/interior/right regions.
- Signals / Profile:
  - Scalar/control cost out of proportion to copy bandwidth.
  - Hot path dominated by masked-load/compare chains for pad bounds.

### `parallel`

- Summary: Use `tl.parallel` to run independent vector-side work concurrently across the two vector cores in one AICore. This helps when compute branches are independent and substantial enough to amortize parallel-control overhead.
- Source: [parallel.md](patterns/parallel.md)
- Use When:
  - Two compute-side substeps are independent but currently sequential.
  - Candidate work is vector compute (casts, scales, elementwise transforms), not shared-bandwidth loads.
  - Branch work is large enough that `tl.parallel` overhead is small relative to useful work.
- Avoid When:
  - Branches have real data dependencies.
  - Dominant bottleneck is memory bandwidth.
  - Candidate work is too small/fine-grained.
- Signals / Code:
  - Independent operand transforms exist before a shared consumer (for example `tl.dot`).
  - Natural split across inputs or independent epilogue branches.

### `program-multiple-rows`

- Summary: Map multiple logical rows to one Triton program (`BLOCK_M > 1`) to amortize per-program overhead and improve vector utilization in row-structured kernels.
- Source: [program-multiple-rows.md](patterns/program-multiple-rows.md)
- Use When:
  - Kernel is naturally row-wise (row reductions, row-wise fused epilogues, row-major transforms).
  - Current launch maps one row per program and profiling shows many thin programs or scalar-heavy overhead.
  - Inner-dimension streaming over `N` can remain single-pass while widening row count.
  - Row count is large enough to amortize wider per-program bundles.
- Avoid When:
  - Row count is tiny and wider bundles cannot amortize setup.
  - Increasing `BLOCK_M` introduces second full passes or unstable numeric behavior.
  - Main bottleneck is elsewhere (layout/store shape, algorithm structure, unrelated scalar traps).
  - Ping-pong/multibuffer variants are introduced without clear MTE-vector overlap evidence.
- Signals / Code:
  - `program_id(0)` maps directly to one row.
  - Repeated per-row pointer/control setup dominates loop body.
  - Inner-dimension tiling exists (`BLOCK_N`), but row axis remains under-batched.
- Signals / Profile:
  - Scalar/control pressure stays high with one-row programs.
  - Moderate row batching gives clear gains, but over-widening regresses.
  - Useful cues include `aiv_scalar_ratio`, `aiv_mte2_ratio`, and `op_statistic` Avg/Count deltas; treat `BAR` cycles as diagnostic context, not a success metric by itself.
  - Barrier/wait growth with many short programs is a common indicator that row granularity is too fine.
  - `op_statistic` Avg should be compared on matched shapes/workload; Count changes can otherwise hide regressions.
  - If `aiv_mte2_ratio` dominates while scalar ratio is low, row batching may be secondary to transfer/layout levers.
  - If scalar ratio remains high after moderate `BLOCK_M` increases, combine with scalar-control cleanups rather than widening blindly.

### `remove-implicit-transpose`

- Summary: Eliminate implicit transpose-style operand access by materializing the required physical layout explicitly (often on host) instead of relying on stride tricks in-kernel.
- Source: [remove-implicit-transpose.md](patterns/remove-implicit-transpose.md)
- Use When:
  - Math needs `[K, N]` but operand is stored as `[N, K]`.
  - Kernel indexes operand with transpose-emulating strides.
  - Profile/IR suggests transform-heavy lowering and wait-heavy matmul execution.
- Signals / Code:
  - Operand storage follows framework default (for example `weight: [N, K]`) while kernel consumes `[K, N]`.
  - Addressing uses stride tricks to reinterpret layout rather than explicit transformed storage.

### `reorder-load`

- Summary: Reorder independent loads so false sequencing does not serialize memory traffic and create avoidable wait in memory-bound kernels.
- Source: [reorder-load.md](patterns/reorder-load.md)
- Use When:
  - Hot loops contain multiple loads with no true dependency but issue serially.
  - Loop-carried dependencies force one load to wait, and unrelated loads are placed after it.
  - Profile evidence suggests memory latency dominates more than arithmetic throughput.
- Avoid When:
  - Load reordering would violate true data dependencies or semantics.
  - Kernel is too small for scheduling changes to matter.
  - Root bottleneck is not memory sequencing.
- Signals / Code:
  - Independent `tl.load` operations appear after dependent pointer resolution.
  - Address setup and dependent loads are interleaved so independent loads start late.
- Signals / Profile:
  - Memory-bound behavior persists with low arithmetic sensitivity.
  - Small arithmetic changes do not move runtime, but sequencing changes might.

### `scalar-latency-traps`

- Summary: Remove scalarizing control and index constructs that force vector-friendly kernels into avoidable scalar work and long dependency chains.
- Source: [scalar-latency-traps.md](patterns/scalar-latency-traps.md)
- Use When:
  - Runtime shape constants are passed as normal arguments instead of `tl.constexpr`.
  - Hot loops rely on loop-carried pointer `+=` recurrences.
  - `%` is used for tail handling where masks would preserve semantics.
  - `tl.where` handles effectively uniform predicates or single-lane exceptions.
  - Hot control/index math stays in `int64` despite proven `int32`-safe range.
  - Long one-dimensional prefix flows (for example `tl.cumsum`) show scalar degradation.
- Avoid When:
  - Bottleneck is memory layout, store shape, or launch geometry rather than scalar control.
  - Wraparound `%` semantics are mathematically required.
  - `int32` safety cannot be proven.
  - Candidate rewrite changes numerical behavior without explicit correctness budget.
- Signals / Code:
  - Repeated coordinate decode (`//`, `%`, wide-index arithmetic) inside inner loops.
  - Invariant setup rebuilt every iteration.
  - Degenerate lane predicates expressed as full-vector conditionals.

### `slice_coalesce`

- Summary: Use UB slice operations (`tl.extract_slice`, `tl.insert_slice`) to reshape scatter/gather-like transfers so random global-memory traffic is paid on only one side of the path.
- Source: [slice_coalesce.md](patterns/slice_coalesce.md)
- Use When:
  - Scatter/gather movement dominates runtime and random global accesses are expensive.
  - Work resembles token rearrangement, sparse index remap, or other index-directed copy paths.
  - Access direction implies either reads or writes can be coalesced (even if not both).
  - A stable block/chunk structure exists for UB staging.
- Avoid When:
  - Accesses are already mostly contiguous/coalesced.
  - UB is too constrained for useful staging.
  - Index locality is too weak for chunked staging to help.
  - The primary bottleneck is not transfer shape.
- Signals / Code:
  - Hot loops issue repeated scattered loads and scattered stores together.
  - Per-token movement is handled one-by-one although chunk staging is possible.
  - Index arithmetic is mixed directly into every transfer step.
- Signals / Profile:
  - Transfer-heavy timeline with modest arithmetic pressure.
  - Strong sensitivity to randomness/locality despite similar compute counts.

### `slice_intermediate`

- Summary: Use staged slice processing when intermediate tensors (not algorithm shape itself) push a kernel over UB capacity.
- Source: [slice_intermediate.md](patterns/slice_intermediate.md)
- Use When:
  - Intermediate tensors are the main source of UB pressure.
  - Formula is sound, but full-size temporaries cannot coexist in UB.
  - Elementwise/fused updates create multiple same-shape live tensors.
  - One or more axes can be partitioned into independent chunks.
- Avoid When:
  - UB pressure is low and slicing only adds loop/control cost.
  - Cross-slice dependencies would alter reduction order or semantics.
  - Layout/launch/tiling rewrites can remove pressure more directly.
- Signals / Code:
  - Broadcasted scales/masks/updates remain live with full-size accumulators.
  - Larger tiles trigger UB overflow or instability.
  - Arithmetic is simple, but live-footprint count is too high.
- Signals / Profile:
  - Performance cliffs appear at larger tiles due to memory pressure.
  - Throughput does not scale with larger blocks as expected.

### `software-pipeline`

- Summary: Use software pipelining to overlap memory transfer and compute in an already well-tiled hot loop.
- Source: [software-pipeline.md](patterns/software-pipeline.md)
- Use When:
  - Loop structure is already tiled and semantically stable.
  - Execution still looks like synchronous load-then-compute.
  - Profiles show wait-heavy gaps while memory engines feed compute units.
  - UB headroom can hold the required live tile sets.
- Avoid When:
  - Inner-loop trip count is tiny (pipeline setup dominates).
  - UB cannot hold multi-stage live tiles safely.
  - Iteration dependencies prevent overlap.
  - Kernel still needs first-order structural rewrite (for example convert manual reduction to regular tiled `tl.dot` first).
- Signals / Code:
  - `tl.load` and compute are serialized every iteration.
  - Manual pointer arithmetic dominates loop body.
  - `tl.make_block_ptr` / `tl.advance` are absent despite regular tiled access.
- Signals / Profile:
  - Cube/Vector idle gaps while MTE transfers run.
  - Tiling-only changes plateau before overlap is addressed.

### `tiling`

- Summary: Use hierarchical tiling to reduce per-program working-set size so tiles, intermediates, and multi-tensor live state fit Unified Buffer (UB) safely.
- Source: [tiling.md](patterns/tiling.md)
- Use When:
  - Large block sizes or live intermediates risk UB overflow.
  - Kernel structure is mostly correct, but memory footprint per program is too large.
  - Runtime failures or instability appear when widening tiles.
  - You need `BLOCK_SIZE` for scheduling/core-dim behavior and `BLOCK_SIZE_SUB` for memory safety.
- Avoid When:
  - UB pressure is not the bottleneck.
  - Kernel still needs first-order structural rewrite (for example manual reduction should first become regular tiled `tl.dot`).
  - Footprint is already safe and the next issue is overlap (prefer `software-pipeline`).
- Signals / Code:
  - Multiple tensors and temporaries are simultaneously live in one tile iteration.
  - Large `BLOCK_SIZE` values trigger overflow or access violations.
  - Performance degrades sharply when increasing tile width.

### `vec-cmp`

- Summary: Rewrite hot-path integer compare-heavy logic into vector-friendly form so mask/selection code does not degrade into scalar bottlenecks on Ascend NPU.
- Source: [vec-cmp.md](patterns/vec-cmp.md)
- Use When:
  - Explicit `i64`/`i32` comparisons dominate hot-path masking logic.
  - Compare-heavy control flow appears outside the compiler's normal load/store mask fast path.
  - Profile evidence shows scalar/control pressure around compare sections.
- Avoid When:
  - Comparisons are already in efficient inlined `tl.load`/`tl.store` masks and perform well.
  - Operand ranges cannot safely support the planned cast strategy.
  - Compare path is cold or non-critical.
  - Cast overhead or semantics risk outweighs expected gain.
- Signals / Code:
  - Integer compare masks are built explicitly and reused in `tl.where`/conditional assignments.
  - Compare scaffolding repeats inside inner loops.
  - Index-width choices (`idx32` vs `idx64`) materially change compare behavior and cost.
- Signals / Profile:
  - Scalar pressure remains high in otherwise vector-friendly kernels.
  - Tiling/overlap tuning gives weak gains until compare path is simplified.
