# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

### `a5-force-simt-only-discrete-access`

- Summary: Launch discrete-memory-access Triton kernels on A5 with `force_simt_only=True`, then retune `num_warps` and grid decomposition. This profile-gated launch-mode experiment targets kernels whose hot path is primarily scalar/index-driven memory access.
- Source: [a5-force-simt-only-discrete-access.md](patterns/a5-force-simt-only-discrete-access.md)

### `autotune`

- Summary: Use Triton-Ascend autotune as the default way to search split sizes, tile sizes, and selected compile options when the kernel structure is already reasonable and the main open question is parameter choice.
- Source: [autotune.md](patterns/autotune.md)

### `software-pipeline-dependency-profiling`

- Summary: Use this pattern as the highest-priority candidate when `extracted_bin_data/report.txt` or equivalent profiling evidence suggests transfer, compute, and store are weakly overlapped, the kernel contains `tl.load`, and the kernel structure can be reorganized to improve overlap. Constructing a steady-state loop around `tl.load`, enabling compiler prefetch, and then tuning `num_stages` can improve performance when the hot path is regular enough to support these changes.
- Source: [software-pipeline-dependency-profiling.md](patterns/software-pipeline-dependency-profiling.md)

## Generated Pattern Summaries

### `a5-force-simt-only-discrete-access`

- Summary: Launch discrete-memory-access Triton kernels on A5 with `force_simt_only=True`, then retune `num_warps` and grid decomposition. This profile-gated launch-mode experiment targets kernels whose hot path is primarily scalar/index-driven memory access.
- Source: [a5-force-simt-only-discrete-access.md](patterns/a5-force-simt-only-discrete-access.md)
- Use When:
  - Target hardware is confirmed as A5 by user statement, profile metadata, runtime/compile logs, runtime device query, or environment/CANN target settings.
  - `msprof` profiling has an `op_summary_*.csv` row whose `opName` matches the Triton kernel name.
  - That row shows `aiv_scalar_ratio` clearly higher than `aiv_vec_ratio` and `cube_utilization`.
  - The kernel body is primarily discrete/index-driven memory access, gather/scatter-like movement, or scalar-heavy pointer/index computation.
  - Correctness validation and representative benchmark reruns are available after changing launch parameters.

### `accumulator-layout-alignment`

- Summary: Align accumulator shapes with output memory layout to avoid store-time transposes that degrade into scalar element writes on Ascend NPU.
- Source: [accumulator-layout-alignment.md](patterns/accumulator-layout-alignment.md)
- Use When:
  - `tl.store` writes a transposed logical tensor and profiling or code inspection suggests the write degraded into scalar element stores.
  - The accumulator shape differs from the output memory layout, forcing an implicit store-time transpose.
  - The kernel performs a reduction that naturally produces the "wrong" shape order, and a simple axis swap in the reduction logic would avoid the store-side transpose entirely.

### `algebraic-optimization`

- Summary: Apply mathematical identities and semantics-preserving rewrites to reduce redundant memory passes, full data scans, or live ranges before micro-tuning loads. Covers floating-point identities (e.g., single-pass mean/variance), operator-defined equivalences (e.g., PyTorch logical ops under dtype-specific truthiness and broadcasting), and algebraic merge rules.
- Source: [algebraic-optimization.md](patterns/algebraic-optimization.md)
- Use When:
  - The hot path performs **two or more full traversals** of the same data for statistics, normalization, or mergeable closed-form subexpressions.
  - Profiler or IR suggests **duplicate MTE-heavy** phases that differ only by a scalar statistic of the same tensor.
  - Elementwise **logical** ops (`logical_or`, `logical_and`, …) use **broadcasting**, and truth tests (`ne`, `!= 0`) run on **fully expanded** numeric tensors.
  - You want fewer global passes or cheaper elementwise work **before** changing tile sizes, pipelines, or autotune grids.

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

### `autotune`

- Summary: Use Triton-Ascend autotune as the default way to search split sizes, tile sizes, and selected compile options when the kernel structure is already reasonable and the main open question is parameter choice.
- Source: [autotune.md](patterns/autotune.md)
- Use When:
  - The kernel structure already looks semantically correct, and the likely headroom is in `BLOCK_*` selection, split shape, or Ascend-specific compile options such as `multibuffer`.
  - The current optimization loop is drifting toward repeated manual tiling edits without strong evidence that a structural rewrite is needed first.
  - The hot path exposes one or more free `tl.constexpr` parameters that are not hard-coded at launch time.
  - Bounds masks or loop structure still map cleanly back to runtime shape arguments, so a shape-keyed autotune cache is plausible.
  - The operator is vector-like rather than a Cube-only kernel path that needs a different optimization route.
  - You are not already in a launch-mode experiment that explicitly changes execution style; if you are applying the A5 SIMT-only discrete-access pattern, `num_warps` and grid decomposition are rechecked there after `force_simt_only=True`.

### `block-pointer-dimensionality`

- Summary: Use `tl.make_block_ptr` to model multidimensional contiguous tensor dimensions directly, enabling wider DMA transfers and reducing scalar address-generation overhead compared to flattened 1D offsets.
- Source: [block-pointer-dimensionality.md](patterns/block-pointer-dimensionality.md)
- Use When:
  - A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
  - An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.
  - Profiling or IR suggests the 1D pointer path produces strided or non-coalesced loads across a dimension that is actually contiguous in memory.

### `classic-matmul`

- Summary: Rewrite a manual matmul or K-reduction hot loop into a regular tiled `tl.dot`-based matmul so the kernel structure matches what Ascend Triton lowers well.
- Source: [classic-matmul.md](patterns/classic-matmul.md)
- Use When:
  - the kernel computes an `M x N` output tile with a regular reduction over `K`
  - the current implementation is effectively `sum_k A[..., k] * B[..., k]`
  - profiling or IR suggests the hot loop is spending too much effort on scalar address generation or repeated reduction structure
  - a block-pointer rewrite reduced one scalar chain but the full loop is still not a regular matmul
  - dtype-specialized or shape-specialized paths are acceptable when one tiled regime is clearly better but a unified rewrite would change numerics too much

### `compile_hint`

- Summary: Use compiler hints to communicate layout facts the compiler cannot safely infer from pointer math alone.
- Source: [compile_hint.md](patterns/compile_hint.md)
- Use When:
  - The hot kernel is already structurally good, but lowering still appears conservative.
  - You can prove stronger alignment or contiguity facts than the current code expresses.
  - `tl.dot` inputs are stable and only need targeted padding guidance on the active path.
  - Parent comparisons are already close enough that small lowering changes can still matter.

### `diagonal`

- Summary: While it is good to access data from L2 cache as much as possible, having multiple kernels accessing the *same* data from the L2 cache may cause bank conflicts that slow down operations. One can use the diagonal access pattern to replace the usual swizzle pattern to alleviate this problem. The example applies this technique to matrix multiplication, but it may be applicable in other contexts.
- Source: [diagonal.md](patterns/diagonal.md)
- Use When:
  - Large tiled matrix-style work shows poor locality or bank-conflict-like behavior even though the basic tiling is already reasonable.
  - Many programs touch the same cache regions at the same time, so changing block traversal order may improve effective L2 use.

### `discrete_memory_access`

- Summary: Stage a contiguous range into the Unified Buffer first, then use on-chip indexing (`tl.gather` or equivalent) to select target values, rather than loading directly from global memory through discrete indices. For fixed-channel AoS layouts, apply the same principle with channel-first SoA buffers to enable contiguous loads.
- Source: [discrete_memory_access.md](patterns/discrete_memory_access.md)
- Use When:
  - The central bottleneck is discrete memory access that semantically looks like `out = x[idx]`.
  - Index-driven global loads dominate the hot path, and contiguous staging plus local selection is more plausible than direct scattered reads.
  - The gather source array is small or medium enough that contiguous staging in shared memory is plausible.
  - The hot loop repeatedly reads fixed fields from AoS records with stride-C offsets, such as `[N, 3]` coordinates loaded as `atom_idx * 3 + channel`, and the input is reused enough to amortize wrapper-side SoA materialization.

### `effective-extent-tiling`

- Summary: Choose tile widths from the live logical extent on each axis instead of a legacy maximum or blanket power-of-two rule, so masked lanes do not dominate loop trip counts, transfer work, or vector-path work.
- Source: [effective-extent-tiling.md](patterns/effective-extent-tiling.md)
- Use When:
  - A **`BLOCK_*`** tile is much larger than the **valid extent** protected by a mask, so the kernel does visibly more padded lane work than useful work.
  - The hot path is either **indexed / masked access** or a **copy-like contiguous axis** whose width does not participate in **`tl.dot`**, cube alignment, or reduction-tree structure.
  - Profiling or IR suggests execution cost scales with the **tile width** more than with the **live element count**.
  - The host already has shape information that could choose a smaller tile or a different launch branch.

### `exact-tile-no-boundary-fast-path`

- Summary: Split exact-tile hot paths from generic masked kernels when dispatch-time shape guards can prove there are no tail tiles, so Ascend lowering can avoid boundary-only masks, padding values, block-pointer `boundary_check`, and related control branches.
- Source: [exact-tile-no-boundary-fast-path.md](patterns/exact-tile-no-boundary-fast-path.md)
- Use When:
  - A dominant benchmark shape is exactly tile-divisible, such as `M % BLOCK_M == 0` and `N % BLOCK_N == 0`.
  - Python dispatch can guard the aligned branch before launch and keep the original masked kernel as fallback.
  - MLIR, LLVM, or profiler traces still show boundary checks, masks, padding, or branch/control overhead on the exact-tile hot path.
  - The kernel is already structurally reasonable, so a bounded control-overhead cleanup can matter.

### `grid-flatten-and-ub-buffering`

- Summary: Flatten logical work items onto physical cores and batch small row-wise memory transfers into wider UB stores to reduce launch overhead and improve per-core work density.
- Source: [grid-flatten-and-ub-buffering.md](patterns/grid-flatten-and-ub-buffering.md)
- Use When:
  - The logical grid is much larger than the physical AICore or VectorCore count.
  - Work is partitioned by batch or sequence buckets with visible load imbalance.
  - Each program processes many tiny rows after grid-to-physical-core mapping.
  - Gather-like code has continuous destination rows but still stores one row at a time.
  - Scatter-weight-gradient-like code has repeated row loads that can be batched from continuous source rows.

### `layout-materialization-elision`

- Summary: Avoid materializing tensors whose only purpose is to change logical layout, such as `permute`, `transpose`, `movedim`, `reshape`, `squeeze`, or `unsqueeze`, when the next step immediately copies, stores, reduces, gathers, or otherwise consumes the data. Instead, express the desired logical layout in the consuming kernel's pointer math or block-pointer metadata and write directly to the final destination layout.
- Source: [layout-materialization-elision.md](patterns/layout-materialization-elision.md)
- Use When:
  - The current implementation creates an intermediate tensor with `permute(...).contiguous()`, `transpose(...).contiguous()`, `movedim(...).contiguous()`, `clone()`, `copy_()`, or a Triton helper that exists only to produce a different physical layout.
  - A later step immediately copies that intermediate into the final output, consumes it in a reduction, feeds it to a simple elementwise/gather/scatter kernel, or stores it in another layout.
  - The layout transform is semantically just axis reordering, singleton-axis insertion/removal, reshape/view-compatible reindexing, or another affine mapping.
  - The source and destination access pattern can be represented with explicit strides, 2D/3D tile offsets, or `tl.make_block_ptr`.
  - Profiling shows `Transpose`, `Contiguous`, `DataCopy`, `Memcpy`, `copy_`, or a separate layout-conversion Triton kernel taking meaningful time.
  - The output destination is known at dispatch time, so the optimized kernel can write the final layout directly.

### `loop-invariant-hoisting`

- Summary: Apply **Loop-Invariant Code Motion (LICM)** to Triton kernels: move computations that do **not** depend on the loop induction variable out of the loop, so each iteration performs only the minimal work that truly varies.
- Source: [loop-invariant-hoisting.md](patterns/loop-invariant-hoisting.md)
- Use When:
  - The kernel has a hot inner loop (often a K loop in GEMM-like kernels).
  - Each loop iteration repeats substantial pointer math, mask construction, type casts, or shape bookkeeping.
  - Profiling shows scalar/control work is disproportionately high relative to useful compute.

### `merge-adjacent-stores`

- Summary: Combine separate small `tl.store` operations into one wider store when the destination addresses form a continuous interval, so the NPU emits a single vector-friendly DMA write instead of multiple tiny transactions.
- Source: [merge-adjacent-stores.md](patterns/merge-adjacent-stores.md)
- Use When:
  - Multiple stores target adjacent addresses but are emitted as separate small `tl.store` operations.
  - The destination addresses are provably continuous and the per-element masks are compatible.
  - Profiling or code inspection shows store granularity, not load or compute, is limiting throughput.

### `padded_row_col_copy`

- Summary: Rewrite flat 1D copy kernels over `numel(out)` into row–column tiled form to reduce scalar overhead from per-element coordinate reconstruction on the last dimension.
- Source: [padded_row_col_copy.md](patterns/padded_row_col_copy.md)
- Use When:
  - The operator is **constant pad**, **slice + pad**, or another **per-axis bounds** elementwise map (not gather).
  - The baseline uses **`pid * BLOCK + arange`** over **`numel(out)`** with **heavy div/mod** for **all** coordinates each iteration.
  - Profiling shows **high scalar** or **`tl.load` / mask** cost on **last-dim** pad boundaries.

### `parallel`

- Summary: Use `tl.parallel` to run tasks in the two vector cores of an aicore at the same time.
- Source: [parallel.md](patterns/parallel.md)
- Use When:
  - Two independent vector-side computations happen in sequence and can be split across vector cores.
  - The bottleneck is not primarily memory movement, so exposing more vector-core concurrency is more promising than reworking loads.

### `pooling-inner-w-slab-gather`

- Summary: For **sliding-window spatial pooling** in **NCHW-style** layouts where **W** is the **innermost contiguous** dimension, load one **W slab** of length **`W_SLAB_LEN = STRIDE_W * (BLOCK_OW - 1) + KERNEL_W`** at **`w_abs_min = ow_pid * BLOCK_OW * STRIDE_W - PAD_W`**, then **`tl.gather(slab, STRIDE_W * arange(BLOCK_OW) + kw)`** per **`kw`** instead of many **`start_w + kw`** masked loads. **2D/3D+** differ only in outer **`kh`/`kd`** loops. **Adoption gate:** not applied unless the hot path shows **`W_SLAB_LEN` load + gather**; **interior/boundary splits** may **combine** but **do not replace** gather. Operator-agnostic—validate on your harness.
- Source: [pooling-inner-w-slab-gather.md](patterns/pooling-inner-w-slab-gather.md)
- Use When:
  - The kernel is **AvgPool / MaxPool** (**values only**), or any **fixed `KERNEL_W`** reduction along **W** on a **contiguous** NCHW (or 5D) tensor, with **`kw`** in a **`tl.constexpr`** loop and **`BLOCK_OW`** output columns per program.
  - Profiling or IR shows **repeated narrow or predicate-heavy global loads** on **W** inside **`kw`**, while **`stride_w`** maps output columns to **regularly strided** input columns.
  - **`out_w`** is large enough that **vectorizing along `ow`** matters, and **`triton.cmotion.cdiv(out_w, BLOCK_OW)`** (launch count along W) stays below a measured knee on the target NPU.
  - You can prove **semantic equivalence** for the branches you enable (**padding**, **ceil**, **divisor** / **count_include_pad** for average, **`-inf` / dtype** rules for max).

### `program-multiple-rows`

- Summary: Amortize per-program fixed costs and improve vector-friendly batching for **row-reduction or row-wise fused kernels** by mapping **multiple rows** to one Triton `program_id` via `BLOCK_M > 1`, instead of one row per program.
- Source: [program-multiple-rows.md](patterns/program-multiple-rows.md)
- Use When:
  - The kernel is **naturally row-wise**: each output row depends mainly on one row of input (e.g. row-wise LogSumExp, row norms, row softmax statistics).
  - Profiling or timeline views suggest **high scalar/control overhead**, **under-filled vector work per program**, or **many tiny programs** relative to problem size `B` (batch / number of rows).
  - The row-wise math already uses **tile loops along `N`** (`BLOCK_N`); increasing **`BLOCK_M`** does not force an extra full pass over global memory if you keep a **single streaming pass** over `N` per program.

### `reduce-avoid-transpose-copy`

- Summary: Avoid implementing a non-last-dimension single-axis reduction by first doing `movedim(...).contiguous()` or an equivalent layout materialization. For contiguous row-major input, compute `[outer, reduce, inner]` from the original shape and reduce directly from the original layout with a strided/tiled kernel.
- Source: [reduce-avoid-transpose-copy.md](patterns/reduce-avoid-transpose-copy.md)
- Use When:
  - The operator reduces exactly one logical axis.
  - The reduce dimension is not the last dimension.
  - The input tensor is contiguous in its original row-major layout.
  - The current implementation uses `movedim(...).contiguous()`, `transpose(...).contiguous()`, `permute(...).contiguous()`, or another full layout materialization before reduction.
  - Profiling shows `Transpose`, `Memcpy`, `DataCopy`, `Contiguous`, or similar layout-conversion work before the reduction kernel.
  - The copy time is comparable to or larger than the reduction-kernel time.
  - The suffix dimension after the reduced axis is large enough to provide reasonably coalesced loads along `inner`.

### `remove-implicit-transpose`

- Summary: Eliminate implicit transpose-style access on Ascend NPU by ensuring operands are in the physical layout the kernel needs, avoiding compiler-injected layout transforms.
- Source: [remove-implicit-transpose.md](patterns/remove-implicit-transpose.md)
- Use When:
  - You implement GEMM / Linear-like kernels where one operand is stored as `[N, K]` but the math needs `[K, N]` (e.g. `y = x @ w.T`).
  - Kernel code accesses the operand with **transpose-like strides** (treats `[N, K]` as `[K, N]`).
  - A `tl.dot` operand uses `tl.trans(x).to(dtype)` where the transpose is applied before the dtype conversion, and the result feeds directly into `tl.dot`.
  - Profiling shows high **scalar/control** and/or large **WAIT_FLAG** time around the matmul path.

### `reorder-load`

- Summary: Reorder independent loads so false sequencing does not block memory-level parallelism or create avoidable wait time in a memory-bound kernel.
- Source: [reorder-load.md](patterns/reorder-load.md)
- Use When:
  - **Loop-carried dependencies**: When current iteration depends on previous iteration's store
  - **Multiple independent loads**: When several load operations have no data dependencies
  - **Memory-bound kernels**: Where memory latency is the performance bottleneck
  - **NPU targets**: Particularly beneficial for NPU's memory execution model

### `scalar-latency-traps`

- Summary: Remove scalarizing constructs that block vector hardware utilization on Ascend NPU, including unnecessary scalar control flow, loop-carried pointer recurrences, modulo addressing, narrow masks, and int64 arithmetic on vector paths.
- Source: [scalar-latency-traps.md](patterns/scalar-latency-traps.md)
- Use When:
  - Runtime values that are shape constants are passed as normal arguments instead of `tl.constexpr`.
  - Pointer variables are updated with `+=` inside a loop, creating loop-carried address dependencies.
  - Address expressions use modulo addressing (`%`) to wrap tail tiles or index boundaries.
  - `tl.where` masks all lanes except a single special position, or has exactly one false lane in a vector.
  - Integer elementwise arithmetic is done as scalar-looking `int64` work even though the value range is safely `int32`.
  - `tl.cumsum` or `tl.associative_scan` runs on the last axis of a tensor and profiling or IR suggests scalar fallback instead of vector lowering.
  - `tl.cumsum` runs on a long one-dimensional vector and profiling or IR suggests scalar degradation.

### `scalar-vector-simulation-signal`

- Summary: Match observed signals in overall content of `report.txt` against the categories below to identify the simulation bottleneck type, then follow the mapped pattern to the optimization.
- Source: [scalar-vector-simulation-signal.md](patterns/scalar-vector-simulation-signal.md)
- Use When:
  - You have a `report.txt` output from `extracted_bin_data` (or you have already extracted simulation data and are about to analyze it). Focus on its overall content section.
  - Simulation data shows **abnormal Pipe distribution** in `report.txt` overall `[Pipe Distribution]` section (e.g., SCALAR instructions > 75%, VECTOR instructions < 15%, MTE2 cycles disproportionately high or zero).
  - `report.txt` overall `[Pipeline Flows]` section shows **no MTE2ToVECTOR flows** despite the kernel loading from global memory (Signal Category 5), or SCALARToVECTOR **avg > 50ns** (Signal Category 2).
  - `report.txt` overall `[TRACE Events]` section has **> 10,000 events** dominated by SIGNEXT/ADD/MUL/DIV/SUB/MADD (Signal Category 1).
  - `report.txt` overall `[VECTOR Unit]` UB Read or Write Conflict exceeds 100 (Signal Category 3).
  - `report.txt` overall `[VECTOR Unit]` Utilization avg < 30% (Signal Category 4).
  - `report.txt` overall `[Pipe Distribution]` has **no CUBE row** (or CUBE instr = 0) AND kernel source contains a K-dimension multiply+accumulate loop using `tl.sum(a*b)` instead of `tl.dot` (Signal Category 6).
  - You need to map an observed simulation signal to a related pattern and a concrete optimization direction.
  - You see `tl.load` with `mask`/`other` and want to determine whether the load is taking the slow SCALAR→VECTOR→MTE2 path (Path A) or the fast SCALAR→MTE2→VECTOR path (Path B).
  - The optimization target is **pure tiling parameter tuning** (BLOCK_M/BLOCK_N/BLOCK_K, num_warps, grid config) — these are invisible in single-program simulation and must use hardware profiling.
  - The optimization target is **multi-program atomic contention** (e.g., `tl.atomic_add` under concurrent access) — simulation cannot reproduce this and signals will be misleading.
  - Simulation data shows **no signal hitting any threshold**.
  - The kernel is **inherently lightweight** (e.g., a trivial load+store touch kernel, a copy kernel, or a kernel with no compute loop) — an abnormal Pipe distribution is the natural characteristic of such operations, not an optimization target.
  - CUBE activity is low but non-zero — this indicates `tl.dot` exists but tile sizes or compiler lowering are suboptimal; route to `tiling` or `software-pipeline` instead of Signal Category 6.

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

### `software-pipeline-dependency-profiling`

- Summary: Use this pattern as the highest-priority candidate when `extracted_bin_data/report.txt` or equivalent profiling evidence suggests transfer, compute, and store are weakly overlapped, the kernel contains `tl.load`, and the kernel structure can be reorganized to improve overlap. Constructing a steady-state loop around `tl.load`, enabling compiler prefetch, and then tuning `num_stages` can improve performance when the hot path is regular enough to support these changes.
- Source: [software-pipeline-dependency-profiling.md](patterns/software-pipeline-dependency-profiling.md)
- Use When:
  - `extracted_bin_data/report.txt` is available, or equivalent profiling evidence reports transfer/compute overlap.
  - Most active `core*.veccore*` blocks satisfy the highest-priority profile gate: very low `OverlapRatio(VECTOR/CUBE & MTE2)`, very low `OverlapRatio(VECTOR/CUBE & MTE3)`, very low `OverlapRatio(MTE2 & MTE3)`, and low or moderate `Ratio(VECTOR/CUBE)`.
  - The kernel contains `tl.load`.
  - `tl.load` is inside a loop, or `tl.load` is outside a loop but the kernel has a regular single-tile program shape; try constructing a steady-state loop to make compiler prefetch possible.

### `software-pipeline`

- Summary: Improve overlap between memory movement and compute in a hot loop that is already structurally tiled, typically by combining block pointers, prefetching, and pipelined loop structure.
- Source: [software-pipeline.md](patterns/software-pipeline.md)
- Use When:
  - The hot loop already has a real tiled structure, but loads and computation still happen too serially.
  - Profiling suggests wait-heavy or overlap-poor behavior, and the next question is pipeline quality rather than basic kernel structure.

### `stencil-resize-gm-to-ub-staging`

- Summary: For 2D sampling kernels (resize, gather-stencil, pooling-like windows) whose hot path reads **multiple overlapping input samples per output point**, the core memory strategy is: **stage one contiguous input slab from global memory (GM) into on-chip buffer (UB) per program**, flatten it, then serve stencil reads via **UB-resident gather** and vector math. Secondary wins — often larger than tuning gather itself — are **eliminating UB internal densify copies**, **aligning slab row stride with gather indexing at allocation time**, **tightening slab bounds to the true input window**, and **enlarging the output tile to cut program count**. Prefer **2D output tiling** `(BLOCK_H, BLOCK_W)` over flattening the output plane onto a single 1D program axis.
- Source: [stencil-resize-gm-to-ub-staging.md](patterns/stencil-resize-gm-to-ub-staging.md)
- Use When:
  - Kernel is memory-bound; input read count scales with `(output pixels × stencil footprint)`.
  - IR shows `{DiscreteMemAccess}` / `{ExtractedLoadOrStore}` on per-lane GM loads in the stencil hot path.
  - IR shows **`hivm.hir.copy` UB strided → UB dense** between slab load and gather.
  - Source maps a **2D output tile** through a **1D linear program layout**, or uses **dynamic-index global loads** per stencil sample instead of a staged slab.
  - msprof shows high **ST/LD** or inner **call_count** while **VGATHER share stays ~1%** — layout/launch issue, not gather-compute bound.

### `tiling`

- Summary: Reduce per-program working-set size through hierarchical or sub-block tiling, keeping live data within UB capacity.
- Source: [tiling.md](patterns/tiling.md)
- Use When:
  - Block sizes, live intermediates, or multi-tensor loads risk UB overflow or poor locality.
  - The main problem is working-set size and memory footprint, not the need for a completely different kernel structure.

### `vec-cmp`

- Summary: Rewrite explicit integer compare-heavy logic into a form that is more vector-friendly on Ascend NPU, especially when scalarized compares are blocking fast masking or selection.
- Source: [vec-cmp.md](patterns/vec-cmp.md)
- Use When:
  - Explicit `i64` or `i32` comparisons appear on the hot path outside the compiler's normal fast load/store mask cases.
  - Comparison-heavy control flow or masking looks like a real vectorization blocker rather than just minor boundary handling.
