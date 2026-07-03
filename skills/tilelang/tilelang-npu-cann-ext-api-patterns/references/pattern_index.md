# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

### `cv-sync`

- Summary: Split a kernel body into `T.Scope("C")` (Cube Core) and `T.Scope("V")` (Vector Core) blocks, using explicit `T.set_flag`/`T.wait_flag` pairs and `T.set_cross_flag`/`T.wait_cross_flag` to coordinate data handoffs between the Cube MMA unit and the Vector/MTP pipelines. Fundamental Expert-mode pattern — every Expert kernel starts here.
- Source: [cv-sync.md](patterns/cv-sync.md)

### `explicit-memory`

- Summary: Replace abstract `T.alloc_shared` / `T.alloc_fragment` with explicit hardware-level allocation APIs (`T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C`) to gain precise control over buffer placement, sizing, and lifetime. Required before any other Expert memory pattern.
- Source: [explicit-memory.md](patterns/explicit-memory.md)

### `double-buffer`

- Summary: Use two sets of L1/UB buffers (ping-pong) with `T.set_flag`/`T.wait_flag` synchronization to overlap MTE data prefetch with Cube compute. Also covers L0A/L0B/L0C double-buffering for per-sub-block MMA.
- Source: [double-buffer.md](patterns/double-buffer.md)

### `workspace-pipeline`

- Summary: Use GM workspace tensors as a ring buffer between Cube and Vector scopes, with READY/FREE cross-core semaphore pairs controlling a multi-task pipeline. Enables 3-task concurrent execution for maximum throughput (MTE2/Cube bound).
- Source: [workspace-pipeline.md](patterns/workspace-pipeline.md)

## Generated Pattern Summaries

### `cv-sync`

- Summary: Split a kernel body into `T.Scope("C")` (Cube Core) and `T.Scope("V")` (Vector Core) blocks, using explicit `T.set_flag`/`T.wait_flag` pairs and `T.set_cross_flag`/`T.wait_cross_flag` to coordinate data handoffs between the Cube MMA unit and the Vector/MTP pipelines.
- Priority: high
- Source: [cv-sync.md](patterns/cv-sync.md)
- Use When:
  - A kernel mixes `T.gemm_v0` (Cube) operations with element-wise math, reductions, or normalization (Vector) in the same compute flow.
  - The auto-managed Developer-mode `pass_configs` produce incorrect results or suboptimal performance due to conservative sync insertion.
  - You need explicit control over when MTE copies complete before Cube reads, or when Cube finishes before Vector consumes results.
  - The kernel benefits from overlapping Cube compute with MTE data movement, requiring precise pipeline-level synchronization.
- Avoid When:
  - The kernel is purely element-wise (no `T.gemm_v0`) — scope separation adds complexity without benefit.
  - The auto-managed `pass_configs` already produce correct and performant results.
  - The kernel structure is simple enough that `T.barrier_all()` after each major step is sufficient.

### `workspace-pipeline`

- Summary: Use GM workspace tensors as a ring buffer between Cube and Vector scopes, with READY/FREE cross-core semaphore pairs controlling a multi-task pipeline. Each workspace slot carries NR consecutive KV-block results, amortizing cross-core synchronization overhead.
- Priority: high
- Source: [workspace-pipeline.md](patterns/workspace-pipeline.md)
- Use When:
  - A kernel mixes Cube MMA and Vector element-wise/softmax in a multi-step pipeline requiring cross-core data handoff.
  - Each task produces intermediate results (S = Q·K^T, P = softmax(S), P·V) that flow Cube → Vector → Cube.
  - The Cube pipe must never drain — tasks should stay continuously in flight.
  - `T.Pipelined` cannot express cross-core data handoff.
- Avoid When:
  - The kernel has no cross-core dependency (pure Cube-only or Vector-only).
  - A single `T.barrier_all()` between scopes is sufficient.
  - The task count per tile is 1 — ring buffer overhead exceeds benefit.

### `double-buffer`

- Summary: Use two sets of L1/UB buffers (ping-pong) with `T.set_flag`/`T.wait_flag` synchronization to overlap MTE data prefetch with Cube compute, hiding memory latency behind computation. Also covers L0A/L0B/L0C double-buffering for per-sub-block MMA within a task.
- Priority: high
- Source: [double-buffer.md](patterns/double-buffer.md)
- Use When:
  - The K-loop in a GEMM kernel is memory-bound — MTE copy time dominates over Cube compute time.
  - The kernel has sufficient L1/UB capacity to hold two buffer sets.
  - `T.Pipelined` (auto software pipeline) is insufficient or you need finer control over the overlap depth.
  - You are already in Expert mode (all auto-passes OFF) with explicit memory allocation.
- Avoid When:
  - The kernel is compute-bound — double buffering won't help if Cube is the bottleneck.
  - L1/UB capacity cannot hold two complete buffer sets.
  - `T.Pipelined(num_stages=2)` already achieves the desired overlap with less code complexity.
  - The kernel is still in Developer mode with auto-passes ON.

### `explicit-memory`

- Summary: Replace abstract `T.alloc_shared` / `T.alloc_fragment` with explicit hardware-level allocation APIs (`T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C`) to gain precise control over buffer placement, sizing, and lifetime.
- Priority: high
- Source: [explicit-memory.md](patterns/explicit-memory.md)
- Use When:
  - The compiler's automatic shared→hardware mapping produces suboptimal buffer placement (e.g., a Vector buffer placed in L1 instead of UB).
  - You need to control exact buffer sizes to fit within hardware limits (UB size, L1 size, L0C size).
  - Double-buffering requires two buffer sets at specific hardware levels.
  - The kernel is hitting UB or L1 capacity limits and needs manual buffer sizing.
- Avoid When:
  - The kernel is simple and the compiler's automatic mapping is correct.
  - You are still prototyping the kernel structure — explicit memory adds complexity that makes iteration slower.
  - The performance gain from manual placement does not justify the maintenance cost.

### `layout-affinity`

- Summary: Use `T.annotate_layout` with `make_zn_layout` / `make_nz_layout` to annotate L1 buffer data layouts for optimal Cube MMA throughput. The compiler uses these hints to optimize MTE copy bursts and MMA operand ordering. Essential for resident buffers and double-buffered L1 operands in Expert-mode kernels.
- Priority: medium
- Source: [layout-affinity.md](patterns/layout-affinity.md)
- Use When:
  - A kernel uses `T.gemm_v0` with explicit L1 buffers (`T.alloc_L1`).
  - The kernel keeps a buffer resident across multiple MMA operations (e.g., Q matrix loaded once per tile and reused).
  - Performance profiling shows memory-bound behavior on MTE copies or MMA operand staging.
  - The kernel uses double-buffered L1 buffers where precise layout optimization matters.
- Avoid When:
  - The kernel is in Developer mode with `TL_ASCEND_AUTO_CV_COMBINE: True` — the compiler handles layout automatically.
  - The kernel is purely element-wise with no MMA operations.
  - The performance gain from layout annotation is marginal for the kernel's complexity.
