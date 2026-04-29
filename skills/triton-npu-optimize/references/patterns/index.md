# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

## How To Use This Index

1. Identify the dominant symptom from code inspection, benchmark evidence, profiling evidence, or IR evidence.
2. Pick the most relevant pattern or the smallest useful set of patterns.
3. Read only those detailed pattern files.
4. Avoid bulk-loading all pattern references unless the operator genuinely shows multiple independent bottlenecks.
5. Do not choose a pattern only because it is easy to try; record the evidence that makes the pattern plausible for this round.

## Pattern Selection Table

### `tiling`

- Use when:
  - block sizes are large
  - Unified Buffer pressure is high
  - the kernel risks UB overflow or poor locality
- Signals:
  - large `BLOCK_SIZE`
  - multiple tensor loads per tile
  - boundary-heavy memory handling
  - benchmark or IR evidence suggests UB pressure or poor locality instead of a pure launch-parameter problem
- Expected benefit:
  - lower on-chip memory pressure
  - better locality
- Main risk:
  - too much loop overhead or reduced effective parallelism
- Read next:
  - [tiling.md](tiling.md)

### `classic-matmul`

- Use when:
  - the kernel is logically an `M x N` output with a regular reduction over `K`
  - the current implementation is row-wise multiply-plus-sum code or another manual reduction that should become `tl.dot`
  - the hot loop is matmul-like even if it is not written as an obvious GEMM today
- Signals:
  - repeated `tl.load` plus elementwise multiply plus `tl.sum`
  - scalar-heavy pointer math or reduction structure around the K loop
  - a fused epilogue already exists and would amortize better over a real output tile
- Expected benefit:
  - more regular matmul lowering
  - less scalar-heavy reduction structure
  - better epilogue amortization on larger shapes
- Common follow-up:
  - if one unified rewrite is not acceptable for every regime, split into dtype-specialized or shape-specialized paths
- Main risk:
  - tile setup overhead can hurt small shapes
  - forced mixed precision may change float32 behavior
- Read next:
  - [classic-matmul.md](classic-matmul.md)
  - then `software-pipeline.md` only if the tiled loop still shows load/compute gaps

### `reorder-load`

- Use when:
  - the kernel appears memory-bound
  - independent loads are serialized by surrounding code order
  - loop-carried dependencies block otherwise parallel loads
- Signals:
  - dependent loads appear before independent ones
  - profiling suggests memory wait dominates
  - the code has obvious false sequencing
- Expected benefit:
  - better instruction-level parallelism
  - improved overlap of memory activity
- Main risk:
  - accidentally violating real dependencies
- Read next:
  - [reorder-load.md](reorder-load.md)

### `software-pipeline`

- Use when:
  - tiled loops repeatedly load then compute in strict sequence
  - profiling shows clear compute stalls waiting for memory
  - block pointer and prefetch style optimization looks applicable
- Signals:
  - repeated inner loops over K-style dimensions
  - sequential `load -> compute -> load -> compute`
  - visible MTE/compute gaps in `msprof`
- Expected benefit:
  - improved overlap between memory transfer and compute
  - lower scalar overhead through block pointers
- Main risk:
  - extra UB pressure from keeping multiple tiles live
- Read next:
  - [software-pipeline.md](software-pipeline.md)
  - prefer `classic-matmul.md` first if the loop is still manual reduction code

### `autotune`

- Use when:
  - the kernel already has several plausible launch or tile parameter choices
  - manual parameter picking is likely leaving performance on the table
- Signals:
  - block sizes, warps, or stages are hand-picked with no evidence they are near-optimal
  - there are a few bounded parameter dimensions worth exploring
  - benchmark, profiling, or prior round evidence suggests the kernel is otherwise structurally healthy
- Expected benefit:
  - stronger parameter selection without rewriting the whole kernel
- Main risk:
  - too many config combinations causing high tuning cost
- Read next:
  - [autotune.md](autotune.md)

### `scalar-latency-traps`

- Use when:
  - code inspection shows avoidable scalarizing constructs before a larger rewrite is justified
  - tail handling, loop address updates, or static parameters look suspicious
- Signals:
  - `%` appears in hot address expressions where a mask could preserve contiguous access
  - pointer variables use `+=` inside a loop
  - fixed sizes, strides, booleans, or modes are not `tl.constexpr`
  - single-position `tl.where`, long-axis `tl.cumsum`, or safe int32 vector arithmetic opportunities appear
- Expected benefit:
  - less scalar/control overhead
  - better vectorization and compiler simplification
- Main risk:
  - these are semantic repairs when they change wraparound, dtype range, specialization, or prefix-order behavior
- Read next:
  - [scalar-latency-traps.md](scalar-latency-traps.md)

### `cache-use`

- Use when:
  - the bottleneck is likely memory hierarchy usage rather than pure compute
  - tile or access shapes may not respect L2, L1, or UB capacity well
- Signals:
  - repeated reloads of the same data
  - working sets that could fit better into cache or UB with different tiling
- Expected benefit:
  - better data reuse across L2, L1, and UB
- Main risk:
  - overfitting tile sizes to one level of cache while hurting another
- Read next:
  - [cache_use.md](cache_use.md)

### `compile-hint`

- Use when:
  - the kernel structure is already close to good but compiler assumptions may be too weak
  - alignment or contiguity information is known but not expressed in code
- Signals:
  - `tl.dot` tiles look aligned except for one dimension
  - slices are known contiguous or aligned but the code does not declare it
- Expected benefit:
  - better code generation from stronger compiler hints
- Main risk:
  - incorrect assumptions about alignment or contiguity
- Read next:
  - [compile_hint.md](compile_hint.md)

### `layout-store-and-block-pointers`

- Use when:
  - the hot path is limited by transfer shape, store layout, or block-pointer dimensionality
  - many small stores or a store-time transpose appear in the code
  - high-dimensional contiguous tensors are accessed through flattened or strided one-dimensional offsets
- Signals:
  - several `tl.store` calls write adjacent addresses
  - accumulator layout is transposed just before store
  - inner dimensions are looped or decoded from `program_id` but could be represented in `block_shape`
  - `tl.dot` consumes `tl.trans(x).to(dtype)` instead of converting before transposition
- Expected benefit:
  - fewer scalarized stores
  - more continuous loads and stores
  - less vector-side transpose or broadcast overhead around Cube work
- Main risk:
  - axis, stride, mask, and reduction-shape mistakes
- Read next:
  - [layout-store-and-block-pointers.md](layout-store-and-block-pointers.md)

### `remove-implicit-transpose`

- Use when:
  - one matmul operand is stored as `[N, K]` but accessed as `[K, N]` via strides
  - IR shows implicit transpose markers (e.g. `MayImplicitTransposeWithLastAxis`)
  - CUBE appears starved by transform/control overhead
- Signals:
  - transpose-style stride access in code (`[N, K]` treated as `[K, N]`)
  - IR contains `MayImplicitTransposeWithLastAxis`
  - profiler shows heavy `WAIT_FLAG_DEVI` and transform-heavy matmul path
- Expected benefit:
  - simpler lowering and cheaper transform path by materializing `[K, N]` contiguous
- Main risk:
  - extra host-side transpose/copy and memory overhead if weights are not reused
- Read next:
  - [remove-implicit-transpose.md](remove-implicit-transpose.md)

### `loop-invariant-hoisting` (LICM)

- Use when:
  - an inner loop (often K loop) repeats substantial pointer math, mask construction, or bounds bookkeeping
  - scalar/control overhead appears high relative to useful compute
- Signals:
  - repeated `base + delta(loop_var)` expressions inside the loop (e.g. pointer arithmetic)
  - profiler shows scalar `LD/ST/ADD/CMP` dominating loop time
  - IR shows repeated arithmetic chains inside `scf.while` bodies
- Expected benefit:
  - lower per-iteration scalar/control cost by moving invariants out of the loop
- Main risk:
  - incorrectly hoisting a value that depends on the loop variable (or wrong broadcasting)
- Read next:
  - [loop-invariant-hoisting.md](loop-invariant-hoisting.md)

### `diagonal`

- Use when:
  - large tiled matrix-style work suffers from cache contention or bank conflicts
  - work partitioning across blocks looks too row-major or swizzled for the NPU
- Signals:
  - many kernels touch the same cache regions at once
  - large M/N grids show poor cache behavior despite otherwise reasonable tiling
- Expected benefit:
  - less cache conflict and better L2 usage through diagonal work distribution
- Main risk:
  - more complex task mapping and harder correctness debugging
- Read next:
  - [diagonal.md](diagonal.md)

### `grid-flatten-and-ub-buffering`

- Use when:
  - the logical grid greatly exceeds physical core count
  - batch or sequence partitioning causes per-core imbalance
  - after physical-core mapping, each program still performs many tiny row loads or stores
- Signals:
  - grid dimensions are logical task counts rather than physical-core bounded counts
  - per-program work is bucketed by batch or sequence length
  - gather-like code has continuous destination rows but emits row-by-row stores
  - scatter-gradient-like code loads consecutive rows one at a time
- Expected benefit:
  - better load balance
  - lower scheduling overhead
  - wider UB-staged memory transfers
- Main risk:
  - added inner loops or UB staging can hurt small shapes or exceed UB capacity
- Read next:
  - [grid-flatten-and-ub-buffering.md](grid-flatten-and-ub-buffering.md)

### `gather-load`

- Use when:
  - the kernel uses discrete or index-based global memory access
  - gather-like behavior dominates the bottleneck
- Signals:
  - index arrays drive scattered loads
  - direct non-contiguous global reads appear in hot paths
- Expected benefit:
  - faster effective gather behavior by moving discrete access to shared memory
- Main risk:
  - excessive shared-memory footprint for large source arrays
- Read next:
  - [gather-load.md](gather-load.md)

### `discrete-memory-access`

- Use when:
  - the kernel performs index-driven discrete loads and the access pattern is the central bottleneck
- Signals:
  - semantics resemble `out = x[idx]`
  - direct `tl.load` uses scattered indices into global memory
- Expected benefit:
  - cheaper discrete access by turning global scattered reads into shared-memory selection
- Main risk:
  - duplicated logic with gather-style transformations when the pattern is not the true hotspot
- Read next:
  - [discrete_memory_access.md](discrete_memory_access.md)

### `parallel`

- Use when:
  - independent vector-style work can be split across the two vector cores of one AICore
- Signals:
  - two independent elementwise or conversion subcomputations happen in sequence
  - loads are not the main bottleneck, but vector compute overlap is available
- Expected benefit:
  - better vector-core utilization
- Main risk:
  - adding parallel structure to work that is too small or not truly independent
- Read next:
  - [parallel.md](parallel.md)

### `slice-coalesce`

- Use when:
  - scatter/gather style movement dominates and batching data in UB could reduce random global access
- Signals:
  - token rearrangement, MOE-style movement, or sparse reordering
  - many small scattered writes or reads
- Expected benefit:
  - fewer expensive global memory transactions through batched slice assembly
- Main risk:
  - extra UB usage or overcomplicated data movement
- Read next:
  - [slice_coalesce.md](slice_coalesce.md)

### `slice-intermediate`

- Use when:
  - intermediate tensors threaten UB capacity even when the overall algorithm is reasonable
- Signals:
  - large fused expressions create multiple live temporaries
  - UB overflow risk comes from intermediates, not just raw input size
- Expected benefit:
  - UB-safe staged computation through slice-based processing
- Main risk:
  - too much slicing overhead if the tensor already fits comfortably
- Read next:
  - [slice_intermediate.md](slice_intermediate.md)

### `vec-cmp`

- Use when:
  - explicit integer comparisons feed masks or conditional logic
  - comparison-heavy logic appears on the hot path
- Signals:
  - repeated `i64` or `i32` comparisons outside automatic load/store mask contexts
  - vectorizable compare-heavy logic in `tl.where`-style flows
- Expected benefit:
  - better vectorized compare behavior
- Main risk:
  - unnecessary casting where the compiler already optimizes the pattern
- Read next:
  - [vec-cmp.md](vec-cmp.md)

### `attention-cv-pipeline`

- Use when:
  - a Cube+Vector fused attention-like kernel is structurally sound but vector epilogue work limits latency
  - scale, mask, softmax, bias, or saved log-sum-exp state dominates the post-dot path
- Signals:
  - repeated mask computation inside a hot loop
  - separate scale and mask operations before softmax
  - `exp2(x * log2e)` exists only to represent `exp(x)`
  - profiler evidence suggests Cube/Vector overlap or vector instruction pressure is the next bottleneck
  - target-specific A5 compile options are being considered
- Expected benefit:
  - fewer vector instructions
  - cheaper mask handling
  - better Cube/Vector overlap when architecture and numerics fit
- Main risk:
  - subtle boundary, softmax-state, backward-consistency, or architecture-gating errors
- Read next:
  - [attention-cv-pipeline.md](attention-cv-pipeline.md)

### `program-multiple-rows`

- Use when:
  - the kernel is row-wise (row reduction or row-fused epilogue) and each program currently maps **one row**
  - the per-program work looks too small and scalar/control overhead is suspicious
- Signals:
  - IR shows `tensor<1xf32>` running state plus frequent `tensor.extract`/`tensor.insert` bookkeeping
  - profiling/timeline shows scalar-heavy prologue and many tiny programs for large `B`
- Expected benefit:
  - fewer programs, wider `(BLOCK_M, BLOCK_N)` tiles, better amortization of address/mask/control work
- Main risk:
  - `BLOCK_M * BLOCK_N` too large for UB/registers; must tune and validate
  - does not fix redundant extra global memory passes by itself
- Read next:
  - [program-multiple-rows.md](program-multiple-rows.md)

## Symptom-First Shortcuts

- If the bottleneck looks memory-bandwidth or latency bound, start with:
  - `reorder-load`
  - `scalar-latency-traps`
  - `layout-store-and-block-pointers`
  - `software-pipeline`
  - `cache-use`
  - `tiling`
- If the bottleneck is structurally a manual matmul or K-reduction, start with:
  - `classic-matmul`
  - then `software-pipeline` only after the loop is already a real tiled matmul
- If the bottleneck looks gather or scatter heavy, start with:
  - `gather-load`
  - `discrete-memory-access`
  - `slice-coalesce`
  - `grid-flatten-and-ub-buffering`
- If the bottleneck looks launch-parameter or tile-parameter sensitive, start with:
  - `autotune`
- If the bottleneck looks compiler-assumption sensitive, start with:
  - `compile-hint`
- If the bottleneck looks cache-conflict or block-mapping related, start with:
  - `diagonal`
- If the bottleneck looks vector-core underutilized, start with:
  - `parallel`
- If the bottleneck is Cube+Vector fused attention-style epilogue work, start with:
  - `attention-cv-pipeline`
- If the bottleneck looks UB-limited because of intermediates, start with:
  - `slice-intermediate`

## Common Boundary Rules

- Use `classic-matmul` when the kernel should first become a standard tiled `tl.dot` loop.
- Use `software-pipeline` when that tiled loop already exists and the next issue is overlap.
- Use `tiling` when the main issue is UB footprint, block size, or live intermediate size.
- Use `scalar-latency-traps` before larger rewrites when the current code has clear scalarizing hazards such as modulo addressing, loop pointer recurrence, or missing constexpr parameters.
- Use `layout-store-and-block-pointers` when the core problem is transfer layout, store transposition, or block-pointer dimensionality rather than tile size.
- Use `grid-flatten-and-ub-buffering` after the access semantics are clear and the next issue is physical-core load balance or row-wise UB staging.
- Use `attention-cv-pipeline` only for Cube+Vector fused attention-like kernels where numerics and target architecture are explicitly checked.
- When compare-helper calls such as `tl.maximum()` or `tl.minimum()` appear in the optimized kernel, inspect all similar call sites for omitted `propagate_nan`. Add `propagate_nan=tl.PropagateNan.ALL` when the round intentionally wants explicit, consistent NaN propagation, and record that this can change NaN-input behavior.
- If tiled matmul is only good for part of the operating envelope, prefer validated dtype/shape dispatch over forcing a single implementation everywhere.
- Do not choose `software-pipeline` as a substitute for a missing structural rewrite.
- If the bottleneck looks compare or mask heavy, start with:
  - `vec-cmp`
- If the kernel is row-wise and one-row-per-program looks under-filled, start with:
  - `program-multiple-rows`

## Reading Discipline

- Prefer one primary pattern and at most one secondary pattern per round.
- If multiple patterns look relevant, choose the one that best matches the current round hypothesis.
- Return to this index between rounds instead of carrying every pattern file forward in context.
- When the current evidence is weak, gather stronger evidence before defaulting to tiling, autotune, or launch-parameter exploration.
