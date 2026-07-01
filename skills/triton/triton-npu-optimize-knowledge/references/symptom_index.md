# Symptom Index

Use this file after structured profile or IR evidence already exists.

Read this generated index first. Then read only the one or two most relevant detailed symptom cards before returning to detailed pattern references.

## Generated Symptom Summaries

### `high-scalar-overhead`

- Summary: The round spends too much time on per-program fixed work, scalar control flow, or bookkeeping relative to the amount of vector or cube work each program performs.
- Source: [high-scalar-overhead.md](symptoms/high-scalar-overhead.md)
- Evidence To Confirm:
  - Many tiny launches or very small per-program work dominate the profile.
  - Timeline or summary views suggest under-filled vector execution.
  - Code-mapping outputs show `SCALAR` or control-heavy execution in regions that should mostly feed Vector or MTE work.
  - Code inspection shows one-row-per-program structure, heavy scalar masking, explicit compare-heavy control logic, or a **flat 1D** pad/copy kernel with expensive coordinate decode on the last axis.
- Candidate Pattern Directions:
  - `program-multiple-rows`
  - `padded_row_col_copy`
  - `vec-cmp`
  - `classic-matmul`
  - `block-pointer-dimensionality`
- Common Non-Matches:
  - Scalar-looking code at the edges does not matter if the hot loop is actually cube-bound.
  - Small-shape kernels can show scalar overhead even when the better answer is dispatch or specialization rather than a local rewrite.

### `sequential-launch-loop-sync`

- Summary: A Python `for`/`while` loop drives an NPU kernel (Triton or aclnn) **per iteration** over a shrinking per-iteration work set, and each iteration also reads per-iteration state back to the host (`.item()`/`.cpu()`/`.tolist()`) to pick the next item. The cost is **launch×count + GPU→CPU sync×count**, not useful compute. Host-side structural symptom, distinct from in-kernel scalar overhead.
- Source: [sequential-launch-loop-sync.md](symptoms/sequential-launch-loop-sync.md)
- Evidence To Confirm:
  - Code: a loop body that **both** launches an NPU kernel **and** reads a scalar/small tensor back via `.item()`/`.cpu()`/`.tolist()` in the same iteration, where the read-back gates the next iteration's item selection.
  - Profile: the per-iteration NPU kernel has an **invocation count in the thousands** (≈ loop trip count) and dominates NPU time, while each invocation's useful compute is small and per-block time is nearly constant regardless of tail size (launch/DMA-setup-bound).
  - The per-iteration work is **embarrassingly vectorizable over a tail** of remaining candidates.
  - **Redundant-compute signal:** the tensor the per-iteration kernel writes is never read back (the loop reads a CPU mirror or recomputes the same quantity on CPU) → the NPU kernel is pure wasted work.
- Candidate Pattern Directions:
  - `cpu-offload-dedup` (Flavor B — **open the full file and run its Step 1 diagnostic before dismissing**)
  - `auxiliary-op-fusion` (only if the kernel output is read *after* the loop, i.e. the dependency is not actually per-iteration)
  - `grid-flatten-and-ub-buffering` (only for large/dense per-iteration work sets; a purely sequential algorithm usually regresses)
- Common Non-Matches:
  - A loop that launches an NPU kernel but does **no per-iteration host read-back** is not this symptom — it is likely collapsible; prefer `auxiliary-op-fusion`.
  - If the per-iteration compute is **not** vectorizable over a tail, CPU offload just makes a slower Python scalar loop — do not apply `cpu-offload-dedup` Flavor B.

### `high-transfer-pressure`

- Summary: The round looks dominated by data movement, staging cost, or transfer-heavy execution rather than useful cube or vector work.
- Source: [high-transfer-pressure.md](symptoms/high-transfer-pressure.md)
- Evidence To Confirm:
  - Profile summaries show transfer-heavy ratios, low compute saturation, or wait tied to memory movement.
  - Measured movement time is far above a rough moved-bytes / bandwidth lower bound, especially when the working set should fit on chip comfortably.
  - IR summaries show many transfer-dense stages or repeated data reshaping around the hot path.
  - Code structure repeatedly reloads tensors, stages many intermediates, or performs gather/scatter-like movement.
- Candidate Pattern Directions:
  - `tiling`
  - `effective-extent-tiling`
  - `cache-use`
  - `algebraic-optimization`
  - `discrete_memory_access`
  - `gather-load`
  - `sliding-window-inner-w-slab-gather`
  - `slice-coalesce`
  - `slice-intermediate`
- Common Non-Matches:
  - High transfer alone does not prove the best fix is software pipelining.
  - A kernel can be transfer-heavy because its structure is still scalarized or under-batched, not because the transfer order itself is wrong.

### `poor-locality`

- Summary: The kernel revisits data in an order that weakens reuse, causes repeated reloads, or creates cache-bank contention instead of feeding contiguous tiles efficiently.
- Source: [poor-locality.md](symptoms/poor-locality.md)
- Evidence To Confirm:
  - Repeated access to the same regions still yields weak cache behavior or surprising reload pressure.
  - IR or profile evidence suggests the working set could fit better than current performance indicates.
  - Code structure uses traversal order or scatter/gather layout that fights the hardware memory hierarchy.
- Candidate Pattern Directions:
  - `cache-use`
  - `diagonal`
  - `slice-coalesce`
  - `discrete_memory_access`
- Common Non-Matches:
  - Poor locality is not the same as pure UB overflow; if the main issue is footprint size, prefer footprint-reduction patterns first.
  - Not every gather or scatter pattern is fixable through traversal order alone.

### `weak-pipeline-overlap`

- Summary: The round appears to leave memory movement and compute insufficiently overlapped, so the kernel pays avoidable wait between loading data and using it.
- Source: [weak-pipeline-overlap.md](symptoms/weak-pipeline-overlap.md)
- Evidence To Confirm:
  - Timeline or wait-oriented profile summaries point to load-then-compute serialization.
  - IR summaries show sync-heavy or transfer-dense stages near the hot loop.
  - Code structure already looks tiled, but each loop iteration still loads, computes, and stores in a mostly serial rhythm.
- Candidate Pattern Directions:
  - `software-pipeline`
  - `reorder-load`
  - `classic-matmul`
- Common Non-Matches:
  - Weak overlap is not a license to add pipeline machinery before basic kernel structure is fixed.
  - If the kernel is still a manual reduction or scalar-heavy matmul shape, first normalize structure with a more foundational rewrite.
