# Optimization Pattern Index

Use this file to choose optimization directions before reading any detailed pattern reference.

Read this generated index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

Before scanning the full list, first analyze whether the operator matches any high-priority patterns below. If it does, try those directions first.

## High Priority Patterns

- None.

## Generated Pattern Summaries

### `al-copy-fractal`

- Summary: Handle the Ascend NZ (fractal) tensor layout for cube operands, allocate UB/L1/L0C buffers with `bl.alloc`, transfer data between cube and vector using `al.fixpipe` (L0C→UB) and `al.copy` (UB→L1), and choose the correct NZ conversion strategy for fp16 vs fp32.
- Source: [al_copy_fractal.md](patterns/al_copy_fractal.md)
- Use When:
  - A kernel uses `tl.dot` and the result must be consumed by vector code (requires NZ→ND conversion via fixpipe).
  - Softmax output (the P matrix) needs to move from UB to L1 for a subsequent cube PV matmul via `al.copy`.
  - The kernel uses `sub_vec_id` lane split with L1 subview handoff.
  - Cube operands or intermediates are in NZ format and need correct allocation sizing.

### `al-scope`

- Summary: Split a CV-fusion kernel body into `core_mode="cube"` and `core_mode="vector"` `al.scope` blocks, and dispatch softmax or element-wise work to the VF unit via `al.scope(vector_mode="simd", outline=True)`.
- Source: [al_scope.md](patterns/al_scope.md)
- Use When:
  - A kernel mixes `tl.dot` (cube) operations with element-wise math, softmax, or other vector work in the same loop body.
  - The vector softmax or element-wise computation could benefit from dedicated VF unit dispatch.
  - You need explicit control over cube/vector synchronization boundaries (`disable_auto_inject_block_sync`).
  - You are adding pipeline stages (`PIPE_STAGES > 1`) via `al.parallel` and need cube/vector scope separation.

### `al-scope-args`

- Summary: Work around the Bishengir `AnalyzeDataLayout` compiler assertion that triggers when an NZ-shaped fp32 tensor (`FRACTAL_N0=8`) is used as a block argument inside an outlined `al.scope(vector_mode="simd", outline=True)`.
- Source: [al_scope_args.md](patterns/al_scope_args.md)
- Use When:
  - An outlined VF scope triggers a compiler crash mentioning `AnalyzeDataLayout` or `collapse_shape` during compilation.
  - The kernel passes an NZ-shaped fp32 tensor (e.g., the softmax P matrix in NZ format) into an `al.scope(vector_mode="simd", outline=True)` block.
  - A single-loop (one-pass) softmax inside an outlined scope fails to compile for fp32.

### `al-sync`

- Summary: Add explicit producer-consumer synchronization between cube and vector `al.scope` blocks using `al.sync_block_set` and `al.sync_block_wait`, covering single-buffer, ping-pong, and triple-buffer (task-ring) pipeline depths.
- Source: [al_sync.md](patterns/al_sync.md)
- Use When:
  - A kernel uses separate `al.scope(core_mode="cube")` and `al.scope(core_mode="vector")` blocks that share buffer data across the boundary.
  - Data races, hangs, or incorrect results appear between cube and vector scope handoff points.
  - The kernel needs throughput improvement by adding ping-pong buffering (`PIPE_STAGES=2`).
  - The kernel uses preload or task-ring patterns (`PIPE_STAGES >= 3`) requiring doubled sync events.
  - L0C result buffers are bound before `al.fixpipe` transfer and need synchronization with vector UB consumption.

### `sub-vec-id-1to2`

- Summary: Split vector work across the two `sub_vec_id()` lanes so each lane handles half of the chosen tile axis, while keeping cube math on full tiles with full `tl.dot` semantics, using explicit staging and synchronization for lane handoff.
- Source: [sub_vec_id_1to2.md](patterns/sub_vec_id_1to2.md)
- Use When:
  - A kernel mixes vector work and `tl.dot` (cube) work in a structure where vector work can be split without changing cube math.
  - The kernel still needs full-tile `tl.dot` semantics — partial-dot approximations are not acceptable.
  - Vector utilization is below what the A5 mixed vector-plus-cube structure can support.
  - The kernel is hitting UB/CBUF/register pressure that limits tile size in a vanilla implementation.
