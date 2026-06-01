# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

### `autotune`

- Summary: **Autotune** here means Triton’s `@triton.autotune` decorator: the runtime tries a **small, bounded** list of launch configurations (tile sizes, warp counts, pipeline stages, and other meta-parameters) and picks one that performs best on measured micro-benchmarks of the kernel.
- Source: [autotune.md](patterns/autotune.md)

### `grid-flatten-and-ub-buffering`

- Summary: Use this pattern when performance is limited by too many logical tasks, uneven per-core work, or tiny per-program transfers after a gather/scatter-style rewrite.
- Source: [grid-flatten-and-ub-buffering.md](patterns/grid-flatten-and-ub-buffering.md)

## Generated Pattern Summaries

### `algebraic-optimization`

- Summary: Look for **semantics-preserving** rewrites that reduce **memory passes**, **redundant full scans**, or **live ranges** before micro-tuning loads. The scope includes **floating-point identities** (for example single-pass mean/variance) and **operator-defined** equivalences (for example PyTorch **logical** ops with dtype-specific truthiness and broadcasting). Always validate against the reference; forms that are equivalent on paper can still **regress** after lowering to Ascend Triton (dependency chains, UB pressure, launch overhead).
- Source: [algebraic-optimization.md](patterns/algebraic-optimization.md)
- Use When:
  - The hot path performs **two or more full traversals** of the same data for statistics, normalization, or mergeable closed-form subexpressions.
  - Profiler or IR suggests **duplicate MTE-heavy** phases that differ only by a scalar statistic of the same tensor.
  - Elementwise **logical** ops (`logical_or`, `logical_and`, …) use **broadcasting**, and truth tests (`ne`, `!= 0`) run on **fully expanded** numeric tensors.
  - You want fewer global passes or cheaper elementwise work **before** changing tile sizes, pipelines, or autotune grids.
- Avoid When:
  - The bottleneck is clearly **only** bad tile size or UB overflow with **no** redundant algorithmic passes (prefer `tiling` or footprint patterns first).
  - Custom Triton fusion is attempted before a simpler **host/graph reorder** is proven correct and cheaper end-to-end.
- Signals / Code:
  - Two loops or kernels with nearly identical `tl.load(x)` tiling along the same axis.
  - `broadcast_tensors(x, y)` followed by elementwise truth tests on **wide dtypes** over the full broadcast shape.
- Signals / Profile:
  - `NotEqual` / `BroadcastTo` (or equivalent ops) scale with **broadcast-expanded** `numel`, not with `numel(x) + numel(y)`.
  - Repeated transfer-dense stages that could be merged if math structure were reorganized.
- Signals / IR:
  - Repeated load or reduction structure around the same logical axis where a single pass could feed multiple accumulators (case-dependent).

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

- Summary: **Autotune** here means Triton’s `@triton.autotune` decorator: the runtime tries a **small, bounded** list of launch configurations (tile sizes, warp counts, pipeline stages, and other meta-parameters) and picks one that performs best on measured micro-benchmarks of the kernel.
- Source: [autotune.md](patterns/autotune.md)
- Use When:
  - The kernel body is **stable** (correctness and rough structure are settled).
  - There are **several plausible** `(tile M, tile N, …)` or `(warp count, pipeline stages)` combinations, and no single choice wins on all benchmark shapes.
  - You can keep the **total number of combinations small** (a practical upper bound is on the order of **20**; exceeding that often explodes compile time or search noise).
  - You can define **`key=`** fields so unrelated shapes do not share a cached wrong winner.
  - You can run a **parent comparison**: autotune must beat the **previous best hand-tuned** version on the same harness, not only beat an old baseline.
- Avoid When:
  - The bottleneck is still **wrong algorithm or layout** (autotune will only reshuffle a bad approach).
  - Compile or search time dominates (large search spaces or very heavy kernels).
  - Correctness depends on **accumulation order** or **shared output buffers** unless you add explicit reset or isolation for each config trial.
  - Launch metadata is **duplicated** between the decorator `Config` and the launch site (this causes hard failures or silent wrong configs).

### `cache_use`

- Summary: Use this pattern when the kernel is memory-hierarchy bound: reduce avoidable global-memory movement, keep read-mostly data resident through adjacent phases, and remove wrapper-level full-tensor copies that hide kernel improvements.
- Source: [cache_use.md](patterns/cache_use.md)
- Use When:
  - Profiling shows high transfer pressure (for example MTE-heavy time, many full passes over the same tensor, or obvious reloads between adjacent phases).
  - The algorithm already has a stable structure, but hot tensors are still moved through extra intermediate buffers or thin wrapper kernels.
  - Read-mostly tables or coefficients are consumed repeatedly (for example broadcast tables, mask tables, rope-style coefficients) and can be staged/reused.
  - Wrapper code performs avoidable materialization (`clone`, `copy`, `expand + cast`, or duplicate count/probe launches) around an otherwise fast kernel.
- Avoid When:
  - The real bottleneck is scalar control, poor launch geometry, or missing specialization; use `scalar-latency-traps`, `program-multiple-rows`, or `tiling` first.
  - Reuse expansion significantly increases register live ranges or reduces occupancy.
  - A wider transfer tile exceeds the practical memory/issue sweet spot for the workload; bigger is not always faster.

### `classic-matmul`

- Summary: Rewrite a manual matmul or K-reduction hot loop into a regular tiled `tl.dot`-based matmul so the kernel structure matches what Ascend Triton lowers well.
- Source: [classic-matmul.md](patterns/classic-matmul.md)
- Use When:
  - the kernel computes an `M x N` output tile with a regular reduction over `K`
  - the current implementation is effectively `sum_k A[..., k] * B[..., k]`
  - profiling or IR suggests the hot loop is spending too much effort on scalar address generation or repeated reduction structure
  - a block-pointer rewrite reduced one scalar chain but the full loop is still not a regular matmul
  - dtype-specialized or shape-specialized paths are acceptable when one tiled regime is clearly better but a unified rewrite would change numerics too much
- Avoid When:
  - purely elementwise kernels
  - gather/scatter dominated kernels
  - tiny shapes where tile setup cost is unlikely to amortize
  - should this manual reduction loop become a regular tiled matmul at all

### `compile_hint`

- Summary: Use compile hints to communicate layout facts the compiler cannot safely infer from pointer math alone:
- Source: [compile_hint.md](patterns/compile_hint.md)
- Use When:
  - The hot kernel is already structurally good, but profiling still shows conservative lowering or extra movement/scalar overhead.
  - You can prove stronger alignment/contiguity facts than the code currently expresses.
  - Dot-style kernels are stable and only need targeted lowering guidance.
  - Parent comparisons show the kernel is close to the frontier and small IR/lowering shifts can matter.
- Avoid When:
  - The dominant issue is still structural (wrong tiling, launch geometry, fusion split, or scalarized algorithm shape).
  - Alignment/contiguity assumptions are shape-conditional and not yet guarded by dispatch.
  - Hints are being used as a substitute for fixing invalid pointer/index math.
- Signals / Code:
  - Repeated contiguous slice loads/stores with masks that are mostly full tiles.
  - Dot inputs where one axis (`K`) is the only true padding edge.
  - Pointer/index expressions whose alignment is known from host-side contracts.

### `diagonal`

- Summary: Use this pattern when tiled matrix-style kernels suffer from cache contention caused by traversal order, not by missing tiling.
- Source: [diagonal.md](patterns/diagonal.md)
- Use When:
  - Large matrix-style workloads already use sensible tile shapes but still show locality/conflict issues.
  - Many programs touch similar source regions concurrently under row-major or simple swizzle traversal.
  - Both primary axes span enough blocks that traversal order materially affects reuse behavior.
  - The bottleneck looks like cache/traffic scheduling rather than arithmetic throughput.
- Avoid When:
  - Problem size is small enough that traversal order has little impact.
  - Kernel is still missing first-order tiling/layout fixes.
  - Overhead of complex traversal mapping outweighs expected locality gains.
  - The dominant bottleneck is scalar control, UB capacity, or launch geometry unrelated to cache-region contention.
- Signals / Code:
  - Work assignment is row-major/horizontal and causes repeated synchronized access to the same matrix regions.
  - Tile math is already stable, but performance remains sensitive to block launch ordering.
  - One operand has large footprint where eviction/reload behavior is likely under naive traversal.
- Signals / Profile:
  - Throughput varies with scheduling order despite similar arithmetic work.
  - Signs of memory-system contention or reuse loss persist after tile-size tuning.
  - Performance degrades as matrix block grid grows, even when per-block kernel math is unchanged.

### `discrete_memory_access`

- Summary: When the logical operation is index-driven (`out = x[idx]`-style), avoid per-element scattered global loads on the hot path. Stage contiguous source spans first, then select locally (for example with gather/select from staged data).
- Source: [discrete_memory_access.md](patterns/discrete_memory_access.md)
- Use When:
  - The kernel is dominated by index-driven reads from global memory.
  - Workloads have meaningful contiguous structure (for example large `inner_size` spans) even if the API is gather-like.
  - Profiling shows scalar-heavy index decode (`//`, `%`, pointer reconstruction) around the data movement loop.
- Avoid When:
  - Source ranges are too large to stage efficiently for the active branch.
  - Accesses are already naturally contiguous and direct loads are not the bottleneck.
  - The main issue is launch geometry or kernel decomposition rather than index access shape.

### `gather-load`

- Summary: Optimize gather-like kernels by transforming index-heavy scattered reads into load shapes that are closer to contiguous copy work. On Ascend NPU, gather performance usually improves when the hot path reduces per-element index decoding and minimizes high-width index traffic.
- Source: [gather-load.md](patterns/gather-load.md)
- Use When:
  - The operation is semantically gather/index-select, and profiling shows gather loads dominate latency.
  - The dominant cases have contiguous structure on at least one axis, even if API semantics are indexed.
  - The kernel is scalar-heavy from index decode and address reconstruction.
- Avoid When:
  - Access is already contiguous and gather logic is not the bottleneck.
  - Source/value movement is tiny and launch/setup overhead dominates.
  - The main issue is dot/reduction structure (use `classic-matmul`) or broad tiling/launch geometry first.
- Signals / Code:
  - Direct global loads using index vectors on the hot path.
  - Repeated per-lane coordinate decode for rank handling.
  - High-width index tensors (for example `int64`) used where narrower indices are valid.
- Signals / Profile:
  - Gather kernel consumes most time on one representative case.
  - Scalar ratio remains high after simple address cleanup.

### `grid-flatten-and-ub-buffering`

- Summary: Use this pattern when performance is limited by too many logical tasks, uneven per-core work, or tiny per-program transfers after a gather/scatter-style rewrite.
- Source: [grid-flatten-and-ub-buffering.md](patterns/grid-flatten-and-ub-buffering.md)
- Use When:
  - Logical task count is far larger than physical core count.
  - Work partitioning by batch/sequence causes visible imbalance.
  - After a first rewrite, programs still move tiny contiguous chunks one row at a time.
  - Grid/index decode overhead (`div/mod` recovery from flattened IDs) is nontrivial.
- Avoid When:
  - Workload is already near core count and flattening adds loop overhead.
  - Destination/source continuity is weak enough that UB slab batching is not valid.
  - Main bottleneck is still algorithm shape, scalar traps, or tiling fundamentals.
- Signals / Code:
  - `TOTAL_TASKS >> NUM_CORES` style mapping.
  - Program-level loops that process many tiny slices.
  - Flattened pid decode chains that can be replaced by direct grid mapping.
- Signals / Profile:
  - Launch fragmentation, short bursts, and poor core utilization.
  - Large Block Dim / too many thin programs despite contiguous transfer opportunities.

### `layout-store-and-block-pointers`

- Summary: Use this pattern when latency is limited by **memory layout expression** and **store/load shape**, not by arithmetic complexity. The goal is to present memory movement to the NPU as contiguous, vector-friendly tiles instead of flattened scalarized address chains, transposed store paths, or many tiny stores.
- Source: [layout-store-and-block-pointers.md](patterns/layout-store-and-block-pointers.md)
- Use When:
  - Stores target adjacent addresses but are emitted as multiple small `tl.store` ops.
  - Store order is effectively transposed relative to destination contiguity.
  - A contiguous multidimensional tensor is accessed through flattened 1D offsets with heavy decode overhead.
  - Inner dimensions are looped or pid-decoded even though they can be represented in tile/block shape.
  - Dot paths use avoidable transpose/cast ordering that hurts load/store shape.
- Avoid When:
  - Main bottleneck is still launch geometry, scalar traps, or algorithm structure.
  - Destination/source continuity assumptions are weak or shape-dependent without dispatch guards.
  - Block-pointer metadata (`shape/strides/offsets/order`) cannot be made correct and stable.
- Signals / Code:
  - Repeated scalar address arithmetic (`div/mod`, manual offset chains) around otherwise simple data movement.
  - Transpose-shaped accumulators only to transpose again at store.
  - Repeated narrow loads/stores where contiguous vectors are possible.
- Signals / Profile:
  - High transfer overhead despite moderate arithmetic.
  - Improvements from row/tiling passes plateau until layout/store expression changes.

### `loop-invariant-hoisting`

- Summary: Apply loop-invariant code motion (LICM) in Triton kernels: move work that does not depend on the loop induction variable out of the hot loop so each iteration executes only truly varying computation.
- Source: [loop-invariant-hoisting.md](patterns/loop-invariant-hoisting.md)
- Use When:
  - A hot inner loop repeatedly rebuilds pointer bases, masks, or scalar setup terms.
  - Profiling suggests scalar/control overhead is disproportionate to useful math.
  - The kernel structure is already mostly correct, but loop body bookkeeping remains heavy.
- Avoid When:
  - Main bottleneck is still layout/store shape, launch geometry, or algorithm choice.
  - Candidate expressions actually vary with loop index and cannot be safely hoisted.
  - A rewrite would blur correctness-sensitive numeric paths without clear guardrails.
- Signals / Code:
  - Repeated expressions of the form `base(pid, offs) + delta(k)` inside the loop.
  - Mask parts that are invariant across iterations are rebuilt every iteration.
  - Repeated per-iteration scalar setup for parameters that are launch-invariant.

### `padded_row_col_copy`

- Summary: Optimize **constant pad** and similar **regular bounded copies** by rewriting a **flat 1D** kernel over `numel(out)` into **`out_rows` × `out_dim_last`**: grid over leading logical rows, **column blocks** on the last axis, and a **row-invariant input base** hoisted out of the column loop. Combine **`BLOCK_ROWS > 1`** when the last dim is small, **`NO_COL_PAD`** `constexpr` when the last axis has no pad, **host-side `BLOCK_COLS` refinement**, and optional **`NATIVE_MASKED_LOAD`** split by shape regime.
- Source: [padded_row_col_copy.md](patterns/padded_row_col_copy.md)
- Use When:
  - The operator is **constant pad**, **slice + pad**, or another **per-axis bounds** elementwise map (not gather).
  - The baseline uses **`pid * BLOCK + arange`** over **`numel(out)`** with **heavy div/mod** for **all** coordinates each iteration.
  - Profiling shows **high scalar** or **`tl.load` / mask** cost on **last-dim** pad boundaries.
- Avoid When:
  - The hot path is **gather/scatter** or **index-driven** discrete access (prefer `gather-load` / `discrete_memory_access`).
  - **Dynamic `if`/`elif` on tile kind** inside the column loop is required without proof the backend lowers it safely—prefer a **uniform** column loop on Ascend unless validated.
  - **Interior-only** fast paths that **omit `col_mask`** on `tl.store` without proof for **tail** blocks (`out_dim_last % BLOCK_COLS != 0`).
- Signals / Code:
  - Single linear `offsets` and repeated `//` / `%` on **large strides** to recover the **last** coordinate.
  - One **global `valid`** merging every dimension on each lane of a large flat block.
  - **Multi-phase** column loops (left / interior / right) with **different** `tl.store` masks.
- Signals / Profile:
  - Scalar or control overhead out of proportion to copy bandwidth.
  - Hot path dominated by **masked load** or **compare** chains for pad bounds.

### `parallel`

- Summary: Use `tl.parallel` to run independent vector-side work concurrently across the two vector cores in one AICore. This pattern helps when compute-side branches are independent and substantial enough to amortize parallel-control overhead.
- Source: [parallel.md](patterns/parallel.md)
- Use When:
  - Two or more compute-side substeps are independent and currently executed sequentially.
  - Candidate work is vector compute (type conversion, scaling, elementwise transforms), not shared-bandwidth loading.
  - Branch work is large enough that `tl.parallel` overhead is small relative to useful work.
- Avoid When:
  - Branches share true data dependencies.
  - Dominant bottleneck is memory movement or load bandwidth.
  - Candidate kernels are too small/fine-grained to benefit from branch parallelization.
- Signals / Code:
  - Sequential, independent compute phases on the same iteration.
  - Natural split of work into separate tensor operands or independent transforms.

### `program-multiple-rows`

- Summary: Map multiple logical rows to one Triton program (`BLOCK_M > 1`) to amortize per-program overhead and improve vector utilization in row-wise kernels.
- Source: [program-multiple-rows.md](patterns/program-multiple-rows.md)
- Use When:
  - Kernel is row-structured (row reductions, row-wise fused epilogues, row-major transforms).
  - Current launch maps one row per program and profiling shows many thin programs or scalar-heavy overhead.
  - Problem size has enough rows to amortize wider per-program row bundles.
  - Inner dimension streaming over `N` can stay single-pass while widening row count.
- Avoid When:
  - Row count is tiny; wider row bundles add overhead without amortization.
  - Increasing `BLOCK_M` forces extra full data passes or unstable numeric behavior.
  - Gains are dominated by unrelated bottlenecks (layout/store shape, compile hints, or scalar decode elsewhere).
- Signals / Code:
  - `pid` maps directly to single-row ownership.
  - Per-row pointer/control setup repeated for each program.
  - Hot loops already tile inner dimension (`BLOCK_N`) but row axis remains under-batched.

### `remove-implicit-transpose`

- Summary: Eliminate **implicit transpose-style access** on Ascend NPU by **materializing the transposed operand on the host** (or by storing it in the preferred physical layout), instead of relying on stride tricks inside the kernel.
- Source: [remove-implicit-transpose.md](patterns/remove-implicit-transpose.md)
- Use When:
  - You implement GEMM / Linear-like kernels where one operand is stored as `[N, K]` but the math needs `[K, N]` (e.g. `y = x @ w.T`).
  - Kernel code accesses the operand with **transpose-like strides** (treats `[N, K]` as `[K, N]`).
  - Profiling shows high **scalar/control** and/or large **WAIT_FLAG** time around the matmul path.
- Signals / Code:
  - Weight is stored as `weight: [N, K]` (PyTorch `nn.Linear` default).
  - Kernel computes `b_ptrs` like `b_ptr + k * stride_bk + n * stride_bn` and relies on strides to emulate `[K, N]`.
- Signals / Profile:
  - `WAIT_FLAG_DEVI` dominates the CUBE timeline around matmul.
  - `MOV_OUT_TO_L1_MULTI_ND2NZ` / `nd2nz` and related fixpipe steps appear frequently.
  - AIV shows large scalar `LD_XD_XN_IMM` / `ST_XD_XN_IMM` overhead tied to staging/reorder.
- Signals / IR:
  - `annotation.mark {MayImplicitTransposeWithLastAxis}`
  - `memref.reinterpret_cast ... sizes: [*, *], strides: [1, ?]` on the B tile (common transpose-style view)

### `reorder-load`

- Summary: Reorder independent loads so false sequencing does not block memory-level parallelism or create avoidable wait time in a memory-bound kernel.
- Source: [reorder-load.md](patterns/reorder-load.md)
- Use When:
  - **Loop-carried dependencies**: When current iteration depends on previous iteration's store
  - **Multiple independent loads**: When several load operations have no data dependencies
  - **Memory-bound kernels**: Where memory latency is the performance bottleneck
  - **NPU targets**: Particularly beneficial for NPU's memory execution model
- Avoid When:
  - **Actual data dependencies**: When the load order affects semantic correctness
  - **Very small kernels**: Where optimization overhead outweighs benefits
  - **CPU targets**: CPUs typically have out-of-order execution and hardware scheduling
  - **Complex dependency graphs**: Where reordering might create subtle race conditions
- Signals / Code:
  - Independent load operations are delayed behind unrelated computations or loop-carried dependencies.
  - The hot loop contains several loads with no true dependency between them, but they are still issued serially.
- Signals / Profile:
  - The kernel behaves memory-bound, and reducing avoidable wait between independent loads looks more promising than changing arithmetic.

### `scalar-latency-traps`

- Summary: Use this pattern to remove scalarized control and index work from otherwise vector-friendly Ascend Triton kernels.
- Source: [scalar-latency-traps.md](patterns/scalar-latency-traps.md)
- Use When:
  - Hot loops repeatedly do per-lane `//`, `%`, or wide-index arithmetic for coordinate decode.
  - Runtime values are effectively shape constants but are still passed as normal arguments instead of `tl.constexpr`.
  - Pointer updates rely on loop-carried `+=` recurrences instead of base-plus-offset addressing.
  - `tl.where` is used with effectively uniform predicates (all lanes same decision, or only one exceptional lane).
  - `int64` index/control math dominates even though value ranges are provably `int32`-safe.
  - Long one-dimensional prefix-style vector flows (for example `tl.cumsum`) show scalar degradation in profile or IR.
- Avoid When:
  - The dominant bottleneck is memory traffic, layout/store shape, or launch geometry, not scalar control.
  - Wraparound via `%` is part of required math semantics, not tail handling.
  - Index ranges are not proven safe for `int32`.
  - A replacement path changes reduction/precision behavior without an explicit correctness budget.
  - A tuned vendor/library path already outperforms the candidate rewrite in representative workloads.
- Signals / Code:
  - Repeated scalar-looking coordinate reconstruction around simple load/store or reduction kernels.
  - Uniform or near-uniform predicates expressed as vector `tl.where` on every iteration.
  - Invariant setup terms rebuilt inside inner loops.
  - Frequent scalar guards in cases where exact-tile or no-padding regimes are common.
- Signals / Profile:
  - Scalar/control pipelines dominate while vector work remains underutilized.
  - Flat or weak gains from tile-only tuning until control/index simplification is applied.
  - Regressions when replacing backend-optimized paths despite seemingly simpler kernel logic.
- Signals / IR:
  - Repeated `index_cast`/arith chains inside loop bodies that do not need per-iteration recomputation.
  - Long scalar dependence chains tied to address generation.

### `slice_coalesce`

- Summary: Use this pattern when scatter/gather-like kernels are dominated by random global-memory access and poor transfer coalescing.
- Source: [slice_coalesce.md](patterns/slice_coalesce.md)
- Use When:
  - Scatter or gather style data movement dominates, and batching work in UB could replace many random global accesses with fewer contiguous transfers.
  - The kernel resembles token rearrangement, sparse reordering, or other index-based movement where access direction determines whether reads or writes should be coalesced.
  - The operation has a stable block structure where contiguous chunk staging can be repeated predictably.
  - Profiling suggests data movement shape (not arithmetic complexity) is the primary bottleneck.
- Avoid When:
  - Accesses are already mostly contiguous and coalesced; extra slicing would add overhead without reducing random traffic.
  - UB pressure is already near limit and additional staging would force overly small tiles or expensive synchronization.
  - Index paths are highly irregular with weak locality, so coalescing opportunities are minimal.
  - The dominant issue is scalar control, launch geometry, or reduction structure rather than transfer shape.
- Signals / Code:
  - Tight loops perform repeated elementwise scattered loads and scattered stores in the same path.
  - The kernel can naturally batch contiguous rows/chunks but currently handles tokens one by one.
  - UB-local assembly/disassembly could shift randomness to only one side of the transfer.
- Signals / Profile:
  - Transfer-heavy hotspots dominate while compute utilization remains modest.
  - Performance scales poorly as index randomness increases, even when arithmetic work stays similar.
  - Improvements appear when contiguous chunk size increases, indicating coalescing sensitivity.

### `slice_intermediate`

- Summary: Use this pattern when intermediate tensors, not core algorithm shape, are the main reason a kernel exceeds Unified Buffer (UB) capacity.
- Source: [slice_intermediate.md](patterns/slice_intermediate.md)
- Use When:
  - Intermediate tensors, rather than just inputs or outputs, are the main source of UB pressure.
  - The overall algorithm is still reasonable, but staged slice processing is needed to keep temporary values within on-chip memory limits.
  - The kernel repeatedly performs elementwise or fused updates where temporaries have the same shape as the main accumulator.
  - You can partition one or more axes into independent chunks with predictable boundaries.
- Avoid When:
  - UB pressure is low and slicing would only add loop/control overhead.
  - The operation has strong cross-slice dependencies that would require complex synchronization or change reduction order.
  - The main bottleneck is transfer layout, launch geometry, or scalar index control rather than temporary footprint.
  - A simpler structural rewrite (for example better tiling or fusion split) removes UB pressure more directly.
- Signals / Code:
  - Full-tensor temporaries (broadcasted scales, masks, updates) stay live alongside inputs/outputs in the hot path.
  - UB-related failures or near-limit configurations appear when block size grows.
  - Arithmetic itself is straightforward, but live tensor count per program is too high.
- Signals / Profile:
  - Performance is unstable across tile sizes due to memory-pressure cliffs.
  - Candidate kernels regress sharply when enabling larger blocks that should otherwise help arithmetic efficiency.

### `software-pipeline`

- Summary: Use this pattern to increase overlap between memory movement and compute in an already tiled hot loop.
- Source: [software-pipeline.md](patterns/software-pipeline.md)
- Use When:
  - The kernel already has a stable tiled loop, but execution still looks like synchronous load-then-compute.
  - Profiling shows wait-heavy behavior or visible compute gaps while memory engines fetch next tiles.
  - Block-pointer structure can replace repeated manual pointer arithmetic on the hot path.
  - UB can hold the active tile set needed for prefetch/pipeline depth.
- Avoid When:
  - Inner loop trip count is tiny and pipeline setup overhead dominates.
  - UB headroom is insufficient for multiple live tile sets.
  - Iteration `i+1` depends on compute results from iteration `i` in a way that prevents overlap.
  - The kernel still needs first-order structural rewriting (for example manual reduction should become regular tiled `tl.dot` first).
- Signals / Code:
  - Tiled loops issue `tl.load` then immediately compute, repeatedly, with little decoupling.
  - Pointer arithmetic and offset rebuilds dominate loop body setup.
  - `tl.make_block_ptr` / `tl.advance` are absent despite regular tiled access.
- Signals / Profile:
  - Timeline shows Cube/Vector idle gaps while waiting for memory transfers.
  - Improvements from pure tiling changes plateau before overlap is improved.
  - Wait-dominant behavior persists even after basic launch geometry cleanup.

### `tiling`

- Summary: Use this pattern to reduce per-program working-set size so tiles, intermediates, and multi-tensor live state fit Unified Buffer (UB) safely.
- Source: [tiling.md](patterns/tiling.md)
- Use When:
  - Block sizes, live intermediates, or multi-tensor loads risk UB overflow.
  - Kernel structure is already mostly correct, but tile footprint is too large for stable execution.
  - Performance cliffs appear when increasing tile widths that should otherwise help throughput.
  - A two-level scheme (`BLOCK` for scheduling, `SUB_BLOCK` for memory safety) can be applied without changing semantics.
- Avoid When:
  - UB pressure is not the bottleneck and sub-block loops would only add control overhead.
  - The kernel still needs foundational structural rewriting (for example manual reduction should first become regular tiled `tl.dot`).
  - Main issue is memory/compute overlap after footprint is already safe (use `software-pipeline` next).
  - Access/layout shape is the primary problem rather than working-set size.
- Signals / Code:
  - Large block sizes keep too many tensors live in one iteration.
  - Temporary tensors have near-output shape and overlap in lifetime.
  - Runtime failures or unstable behavior occur when tile size is widened.
- Signals / Profile:
  - Strong regressions or failures at larger tile configurations due to memory pressure.
  - Throughput does not scale with bigger tiles because memory footprint dominates.

### `vec-cmp`

- Summary: Use this pattern when explicit integer comparison logic becomes a scalar bottleneck on Ascend NPU.
- Source: [vec-cmp.md](patterns/vec-cmp.md)
- Use When:
  - Explicit `i64`/`i32` comparisons drive hot-path masks for `tl.where` or similar conditional logic.
  - Comparison-heavy logic appears outside compiler-optimized load/store mask fast paths.
  - Profiling indicates scalar/control pressure tied to compare-and-mask sections.
  - Semantics allow safe conversion to vector-friendly compare dtypes.
- Avoid When:
  - Comparisons are already in compiler-fused `tl.load`/`tl.store` mask expressions and perform well.
  - Values are outside safe representable range for the chosen comparison cast strategy.
  - Comparison path is cold and not performance relevant.
  - Rewriting comparisons would add extra casts/conversions that outweigh benefits.
  - Compare helper rewrites (`tl.maximum`/`tl.minimum`) would change intended NaN propagation semantics.
- Signals / Code:
  - Integer compare masks are constructed explicitly and reused in hot control flow.
  - Repeated compare/cast/mask scaffolding appears in inner loops.
  - Integer compare results gate large vector operations through `tl.where`.
- Signals / Profile:
  - Scalar instruction share remains high in otherwise vector-friendly kernels.
  - Throughput improves little from tiling or overlap tuning until compare path is cleaned up.
