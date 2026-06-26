# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

- None.

## Generated Pattern Summaries

### `cv-sync`

- Summary: Split a kernel body into `T.Scope("C")` (Cube Core) and `T.Scope("V")` (Vector Core) blocks, using explicit `T.set_flag`/`T.wait_flag` pairs to coordinate data handoffs between the Cube MMA unit and the Vector/MTP pipelines.
- Source: [cv-sync.md](patterns/cv-sync.md)
- Use When:
  - A kernel mixes `T.gemm_v0` (Cube) operations with element-wise math, reductions, or normalization (Vector) in the same compute flow.
  - The auto-managed Developer-mode `pass_configs` produce incorrect results or suboptimal performance due to conservative sync insertion.
  - You need explicit control over when MTE copies complete before Cube reads, or when Cube finishes before Vector consumes results.
  - The kernel benefits from overlapping Cube compute with MTE data movement, requiring precise pipeline-level synchronization.

### `double-buffer`

- Summary: Use two sets of L1/UB buffers (ping-pong) with `T.set_flag`/`T.wait_flag` synchronization to overlap MTE data prefetch with Cube compute, hiding memory latency behind computation.
- Source: [double-buffer.md](patterns/double-buffer.md)
- Use When:
  - The K-loop in a GEMM kernel is memory-bound — MTE copy time dominates over Cube compute time.
  - The kernel has sufficient L1/UB capacity to hold two buffer sets.
  - `T.Pipelined` (auto software pipeline) is insufficient or you need finer control over the overlap depth.
  - You are already in Expert mode (all auto-passes OFF) with explicit memory allocation.

### `explicit-memory`

- Summary: Replace abstract `T.alloc_shared` / `T.alloc_fragment` with explicit hardware-level allocation APIs (`T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C`) to gain precise control over buffer placement, sizing, and lifetime.
- Source: [explicit-memory.md](patterns/explicit-memory.md)
- Use When:
  - The compiler's automatic shared→hardware mapping produces suboptimal buffer placement (e.g., a Vector buffer placed in L1 instead of UB).
  - You need to control exact buffer sizes to fit within hardware limits (UB size, L1 size, L0C size).
  - Double-buffering requires two buffer sets at specific hardware levels.
  - The kernel is hitting UB or L1 capacity limits and needs manual buffer sizing.
