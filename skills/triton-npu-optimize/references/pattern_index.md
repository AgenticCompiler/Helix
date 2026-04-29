# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

## Generated Pattern Summaries

### `attention-cv-pipeline`

- Summary: Reduce latency in Cube+Vector fused attention-like kernels by cutting vector-side instruction pressure, making mask/scale work cheaper, and using architecture-gated compile options only when the target device supports them.
- Source: [attention-cv-pipeline.md](patterns/attention-cv-pipeline.md)
- Use When:
  - A `tl.dot` loop is followed by substantial vector epilogue work such as scale, mask, softmax, dropout, or bias.
  - Profiling suggests Cube and Vector work are close enough that vector-side overhead limits overlap.
  - A loop repeatedly recomputes the same mask tensor from sequence lengths or causal indices.
  - Scale and mask are separate operations before softmax.
  - The code stores log-sum-exp state in a base-2 representation solely because the forward path uses `exp2`.
  - The target is known to be an A5 device such as `ascend950PR` or `ascend950DT`.
- Avoid When:
  - The kernel is pure Vector work rather than Cube-plus-Vector fused work.
  - Profiling shows memory transfer, not vector epilogue work, is the dominant bottleneck.
  - Architecture-specific compile settings cannot be gated on verified target information.
- Signals / Code:
  - A `tl.dot` loop is followed by repeated mask, scale, softmax, dropout, or bias work on the vector side.
  - The same mask tensor is recomputed inside a hot loop even though it depends only on host-known metadata.
  - The forward path stores log-sum-exp state in base-2 form solely because it uses `exp2`.
- Signals / Profile:
  - Profiling suggests Cube and Vector work are close enough that vector-side instruction pressure is limiting overlap.
  - The kernel is structurally sound, but the post-dot vector path still appears to dominate latency.

### `autotune`

- Summary: Make use of autotune in Triton to optimize parameters automatically. Some analysis is still needed to set the possible values of parameters to try (limit the number of combinations to try to at most 20).
- Source: [autotune.md](patterns/autotune.md)
- Use When:
  - The kernel already has several plausible tile or launch parameter choices, and the main structure looks reasonable.
  - Manual parameter picking is likely leaving performance on the table, but the search space can still be kept small and bounded.

### `cache_use`

- Summary: Analyze memory access patterns, try to make use of cache and UB as much as possible. Make note of L2 cache (96MB, shared by all cores) and size of L1 and UB (512KB, 256KB, respectively).
- Source: [cache_use.md](patterns/cache_use.md)
- Use When:
  - The bottleneck looks memory-hierarchy bound rather than purely compute bound.
  - Repeated reloads, weak reuse, or poor locality suggest that L2, L1, or UB usage can be improved through better data placement or tile sizing.

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

- Summary: Try the following compile hints:
- Source: [compile_hint.md](patterns/compile_hint.md)
- Use When:
  - The kernel structure already looks close to good, but the compiler still lacks explicit alignment or contiguity information.
  - `tl.dot` tiles, slices, or pointer math are known to satisfy stronger layout assumptions than the code currently expresses.
- Signals / Code:
  - `tl.dot` inputs are already aligned in `M` and `N`, so only the `K` direction still needs padding hints.
  - Pointer slices are known contiguous or aligned, but the code does not yet communicate that with `tl.max_contiguous` or `tl.multiple_of`.

### `diagonal`

- Summary: While it is good to access data from L2 cache as much as possible, having multiple kernels accessing the *same* data from the L2 cache may cause bank conflicts that slow down operations. One can use the diagonal access pattern to replace the usual swizzle pattern to alleviate this problem. The example applies this technique to matrix multiplication, but it may be applicable in other contexts.
- Source: [diagonal.md](patterns/diagonal.md)
- Use When:
  - Large tiled matrix-style work shows poor locality or bank-conflict-like behavior even though the basic tiling is already reasonable.
  - Many programs touch the same cache regions at the same time, so changing block traversal order may improve effective L2 use.
- Signals / Code:
  - Traditional row-major or horizontal block assignment makes many cores touch the same left-matrix cache region at once.
  - The matrix already spans many blocks along both `M` and `N`, so traversal order is a plausible performance lever rather than a cosmetic rewrite.
  - The right-hand matrix is large enough that ordinary block traversal can churn L2 and lower reuse.

### `discrete_memory_access`

- Summary: When loading discrete indices, rather than using `tl.load` to load the discrete set directly, use `tl.load` to load a continuous range first, then use `tl.gather` to select the target values.
- Source: [discrete_memory_access.md](patterns/discrete_memory_access.md)
- Use When:
  - The central bottleneck is discrete memory access that semantically looks like `out = x[idx]`.
  - Index-driven global loads dominate the hot path, and contiguous staging plus local selection is more plausible than direct scattered reads.

### `gather-load`

- Summary: Stage gather-like input through contiguous loads before selecting indexed values so the kernel reduces expensive discrete global-memory reads on Ascend NPU.
- Source: [gather-load.md](patterns/gather-load.md)
- Use When:
  - **Discrete access patterns**: When using index arrays to access non-contiguous memory
  - **Small to medium source arrays**: When the source array can fit in shared memory
  - **Performance-critical sections**: Where gather operations are bottleneck
- Avoid When:
  - **Large source arrays**: When M is too large for shared memory capacity
  - **Already contiguous access**: When memory access patterns are already sequential
  - **GPU targets**: This optimization is NPU-specific and may not benefit GPU architectures
  - **Single-element access**: When only accessing a few discrete elements
- Signals / Code:
  - Code uses index arrays to access non-contiguous memory locations on the hot path.
  - The gather source array is small or medium enough that contiguous staging in shared memory is plausible.
  - Direct global-memory gather reads dominate more than the surrounding arithmetic.

### `grid-flatten-and-ub-buffering`

- Summary: Change work distribution and UB staging when latency is dominated by too many logical tasks, uneven per-core work, physical-core load balance problems, or tiny row-wise memory transfers after a gather/scatter style rewrite.
- Source: [grid-flatten-and-ub-buffering.md](patterns/grid-flatten-and-ub-buffering.md)
- Use When:
  - The logical grid is much larger than the physical AICore or VectorCore count.
  - Work is partitioned by batch or sequence buckets with visible load imbalance.
  - Each program processes many tiny rows after grid-to-physical-core mapping.
  - Gather-like code has continuous destination rows but still stores one row at a time.
  - Scatter-weight-gradient-like code has repeated row loads that can be batched from continuous source rows.
- Signals / Code:
  - The logical grid is much larger than the physical AICore or VectorCore count.
  - Work is partitioned by batch or sequence buckets that create visible load imbalance.
  - Each physical program still processes many tiny rows or row-at-a-time transfers after grid mapping.
- Signals / Profile:
  - Latency is dominated by too many logical tasks, uneven per-core work, or tiny row-wise memory transfers after a gather or scatter style rewrite.

### `layout-store-and-block-pointers`

- Summary: Improve latency by reshaping memory layout, block-pointer dimensionality, and store granularity so the NPU sees continuous vector-friendly transfers instead of scalarized transpose or many tiny operations.
- Source: [layout-store-and-block-pointers.md](patterns/layout-store-and-block-pointers.md)
- Use When:
  - Multiple stores target adjacent addresses but are emitted as separate small `tl.store` operations.
  - `tl.store` writes a transposed logical tensor and appears to degrade into scalar element stores.
  - A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
  - An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.
  - A `tl.dot` operand uses `tl.trans(x).to(dtype)` before entering Cube work.
  - A matmul epilogue adds bias after `tl.dot` in a way that creates unnecessary broadcast or load ordering overhead.
- Signals / Code:
  - Multiple stores target adjacent addresses but are emitted as separate small `tl.store` operations.
  - A store writes a transposed logical tensor and appears to degrade into scalar element stores.
  - A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
  - An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.

### `loop-invariant-hoisting`

- Summary: Apply **Loop-Invariant Code Motion (LICM)** to Triton kernels: move computations that do **not** depend on the loop induction variable out of the loop, so each iteration performs only the minimal work that truly varies.
- Source: [loop-invariant-hoisting.md](patterns/loop-invariant-hoisting.md)
- Use When:
  - The kernel has a hot inner loop (often a K loop in GEMM-like kernels).
  - Each loop iteration repeats substantial pointer math, mask construction, type casts, or shape bookkeeping.
  - Profiling shows scalar/control work is disproportionately high relative to useful compute.
- Signals / Code:
  - Inner loop recomputes expressions of the form:
  - `base(pid, offs) + delta(loop_var)`
  - e.g. `a_ptr + offs_m*stride_am + k*stride_ak`
  - Masks are rebuilt each iteration even when parts are invariant:
  - e.g. `a_mask_m = offs_m < M` is invariant, but recomputed into `a_mask` each iter.
- Signals / Profile:
  - AIV scalar dominated by `LD_XD_XN_IMM`, `ST_XD_XN_IMM`, `ADD(_IMM)`, `CMP_IMM`.
  - Timeline shows CUBE waiting on flags around the loop, while AIV performs control-heavy work.
- Signals / IR:
  - Repeated arithmetic chains (`muli/addi/index_cast`) inside `scf.while` / `scf.for` bodies.
  - Loop bodies contain repeated `subi/minsi/maxsi` patterns for bounds handling.

### `parallel`

- Summary: Use `tl.parallel` to run tasks in the two vector cores of an aicore at the same time.
- Source: [parallel.md](patterns/parallel.md)
- Use When:
  - Two independent vector-side computations happen in sequence and can be split across vector cores.
  - The bottleneck is not primarily memory movement, so exposing more vector-core concurrency is more promising than reworking loads.
- Avoid When:
  - The candidate work items still share a real data dependency.
  - The operation is mostly memory loading, where shared bandwidth is already the limiting factor.
  - The operation is so small that `tl.parallel` overhead is likely larger than the gain.
- Signals / Code:
  - Independent type conversions, element-wise operations, or scaling steps already exist on the hot path.
  - The candidate work split is compute-side and independent, rather than shared-bandwidth memory loading.

### `program-multiple-rows`

- Summary: Amortize per-program fixed costs and improve vector-friendly batching for **row-reduction or row-wise fused kernels** by mapping **multiple rows** to one Triton `program_id` via `BLOCK_M > 1`, instead of one row per program.
- Source: [program-multiple-rows.md](patterns/program-multiple-rows.md)
- Use When:
  - The kernel is **naturally row-wise**: each output row depends mainly on one row of input (e.g. row-wise LogSumExp, row norms, row softmax statistics).
  - Profiling or timeline views suggest **high scalar/control overhead**, **under-filled vector work per program**, or **many tiny programs** relative to problem size `B` (batch / number of rows).
  - The row-wise math already uses **tile loops along `N`** (`BLOCK_N`); increasing **`BLOCK_M`** does not force an extra full pass over global memory if you keep a **single streaming pass** over `N` per program.
- Avoid When:
  - **Second full pass** over `x` for the same row (e.g. two-pass LSE) usually **increases global reads**; msprof often shows **more MTE / wait** unless the algorithm truly requires it. Prefer **single-pass streaming LSE** when numerically stable.
  - **Ping-pong / multibuffer** without evidence of **MTE–vector overlap** can add **sync and UB** cost; treat as a **separate hypothesis** to validate.
  - Do not conclude from **one** metric (e.g. `BAR` cycles) without **end-to-end** timing and comparable workload.
- Signals / Code:
  - `program_id(0)` indexes **rows 1:1** (`pid_m` is the row index), and the inner loop only tiles **`N`**.
  - Scalar helpers (`program_id`, pointer arithmetic per row) run once **per row**; vector units see **narrow** tensors (e.g. `(1, BLOCK_N)` loads).
- Signals / Profile:
  - **`aiv_scalar_ratio`** or scalar-related time is **disproportionately high** compared to useful vector math, for workloads where `B` is large enough that vector throughput should dominate.
  - **`op_statistic`** (per-kernel): **Avg** latency improves when the same logical work uses **fewer launches** (compare with care: **Count** and input shapes must be comparable across runs).
  - If **`aiv_mte2_ratio`** is **not** the sole dominant bucket, pure “double-buffer the loads” may be the wrong first lever; **program batching** can still help by making each program’s inner loop **wider** along rows.
  - Frequent **barrier / wait** patterns tied to **many short programs** or **thin** vector blocks.
  - **Note:** High **`BAR`** cycle counts alone are **not** a success metric; correlate with **wall time**, **op_statistic Avg**, and correctness.

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

- Summary: Remove scalarizing constructs that make an otherwise vector-friendly Ascend Triton kernel spend time on avoidable scalar control, address arithmetic, or long dependency chains.
- Source: [scalar-latency-traps.md](patterns/scalar-latency-traps.md)
- Use When:
  - Runtime values that are shape constants are passed as normal arguments instead of `tl.constexpr`.
  - Pointer variables are updated with `+=` inside a loop, creating loop-carried address dependencies.
  - Address expressions use modulo addressing (`%`) to wrap tail tiles or index boundaries.
  - `tl.where` masks all lanes except a single special position, or has exactly one false lane in a vector.
  - Integer elementwise arithmetic is done as scalar-looking `int64` work even though the value range is safely `int32`.
  - `tl.cumsum` runs on a long one-dimensional vector and profiling or IR suggests scalar degradation.
- Signals / Code:
  - Runtime values that are shape constants are passed as normal arguments instead of `tl.constexpr`.
  - Pointer variables are updated with `+=` inside a loop, creating loop-carried address dependencies.
  - Address expressions use modulo addressing (`%`) to wrap tail tiles or index boundaries.
  - `tl.where` masks all lanes except a single special position, or has exactly one false lane in a vector.
  - Integer elementwise arithmetic is done as scalar-looking `int64` work even though the value range is safely `int32`.
  - `tl.cumsum` runs on a long one-dimensional vector and profiling or IR suggests scalar degradation.

### `slice_coalesce`

- Summary: When the kernel performs scatter/gather operations with non-contiguous memory access patterns, such as token rearrangement in MOE layers, sparse data processing, or any operation involving index-based data movement, use `extract_slice` or `insert_slice` to data reuse while minimizing expensive global memory transactions.
- Source: [slice_coalesce.md](patterns/slice_coalesce.md)
- Use When:
  - Scatter or gather style data movement dominates, and batching work in UB could replace many random global accesses with fewer contiguous transfers.
  - The kernel resembles token rearrangement, sparse reordering, or other index-based movement where access direction determines whether reads or writes should be coalesced.

### `slice_intermediate`

- Summary: When the kernel computation creates intermediate tensors that, combined with inputs and outputs, would exceed the Unified Buffer (UB) capacity (in attention mechanisms, batch normalization, etc), divide computation into several steps, and use `extract_slice` and `insert_slice` to read/write into UB.
- Source: [slice_intermediate.md](patterns/slice_intermediate.md)
- Use When:
  - Intermediate tensors, rather than just inputs or outputs, are the main source of UB pressure.
  - The overall algorithm is still reasonable, but staged slice processing is needed to keep temporary values within on-chip memory limits.

### `software-pipeline`

- Summary: Improve overlap between memory movement and compute in a hot loop that is already structurally tiled, typically by combining block pointers, prefetching, and pipelined loop structure.
- Source: [software-pipeline.md](patterns/software-pipeline.md)
- Use When:
  - The hot loop already has a real tiled structure, but loads and computation still happen too serially.
  - Profiling suggests wait-heavy or overlap-poor behavior, and the next question is pipeline quality rather than basic kernel structure.
- Avoid When:
  - **Tiny Inner Loops**: If the loop only runs 1 or 2 times, the pre-fetch overhead might exceed the savings.
  - **Extreme Memory Pressure**: If the tile size is so large that the Unified Buffer cannot hold two sets of tiles (current and next).
  - **Dependency Chains**: If Tile `i+1` depends on the result of the computation of Tile `i`.
  - **Pre-tiling rewrite still needed**: If the loop should first be rewritten into a regular tiled matmul or another clearer tile-based structure.
- Signals / Code:
  - The loop is already tiled, but each iteration still follows a mostly synchronous load-then-compute rhythm.
  - Manual pointer arithmetic dominates the tiled loop, and block-pointer plus prefetch structure is still missing.
- Signals / Profile:
  - `msprof` timelines show Cube or Vector gaps while the MTE engines fetch the next tile.
  - Wait-heavy behavior suggests insufficient memory/compute overlap rather than a missing tiled-kernel rewrite.

### `tiling`

- Summary: Reduce per-program working-set size through hierarchical or sub-block tiling so large tiles, intermediates, or multi-tensor loads fit UB safely without collapsing overall task structure.
- Source: [tiling.md](patterns/tiling.md)
- Use When:
  - Block sizes, live intermediates, or multi-tensor loads risk UB overflow or poor locality.
  - The main problem is working-set size and memory footprint, not the need for a completely different kernel structure.
- Avoid When:
  - **Small BLOCK_SIZE** No significant memory pressure
  - **Simple operations** with single tensor - UB usage is minimal
  - **Already optimized** with sub-blocking present
  - **Structure is the real problem** - if the current kernel is really a manual matmul or reduction that should first become a regular tiled `tl.dot` loop
- Signals / Code:
  - Large `BLOCK_SIZE` values, multiple tensor loads, or heavy intermediates keep too much data live per program.
  - The kernel already has a reasonable overall structure, but it still needs smaller sub-blocks to control UB usage.
  - Runtime failures or memory access violations appear when block sizes increase on NPU.

### `vec-cmp`

- Summary: Rewrite explicit integer compare-heavy logic into a form that is more vector-friendly on Ascend NPU, especially when scalarized compares are blocking fast masking or selection.
- Source: [vec-cmp.md](patterns/vec-cmp.md)
- Use When:
  - Explicit `i64` or `i32` comparisons appear on the hot path outside the compiler's normal fast load/store mask cases.
  - Comparison-heavy control flow or masking looks like a real vectorization blocker rather than just minor boundary handling.
- Avoid When:
  - **Comparisons in `tl.load`/`tl.store` masks** - already auto-optimized:
  - **Already using fp32 comparisons** - no optimization needed:
  - **Non-performance-critical code** - optimization overhead may not be justified
- Signals / Code:
  - Integer comparisons produce explicit boolean masks used in `tl.where`, conditional assignments, or similar hot-path logic.
  - The comparison is written outside the compiler's normal `tl.load` or `tl.store` mask fast path.
  - The code still compares integer operands directly even though vector-friendly `fp32` comparison would preserve semantics.
