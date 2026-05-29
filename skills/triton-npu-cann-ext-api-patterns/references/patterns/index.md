# CANN Extension API Pattern Index

Use this file to choose an extension-API-specific optimization direction before reading any detailed pattern reference.

Read this index first. Then read only the one or two most relevant detailed pattern files for the current bottleneck.

## How To Use This Index

1. Identify the dominant symptom from code inspection, benchmark evidence, profiling evidence, or IR evidence.
2. Decide whether the bottleneck specifically suggests a CANN Triton extension API rewrite instead of a generic optimize pattern.
3. Pick the most relevant extension pattern.
4. Read only that detailed pattern file unless multiple independent extension-API bottlenecks are clearly present.
5. Record why the selected pattern is plausible for the current round.

## Pattern Selection Table

### `al-scope`

- Use when:
  - the kernel needs to split work into cube and vector `al.scope` blocks
  - the kernel uses (or should use) `al.scope(vector_mode="simd", outline=True)` for VF dispatch
  - you need to understand `al.parallel`, `no_inline`, or kernel launch flags
- Signals:
  - mixed cube+vector kernel without explicit scope boundaries
  - vector softmax or element-wise math that could benefit from VF dispatch
  - need for explicit sync with `disable_auto_inject_block_sync`
- Expected benefit:
  - correct cube/vector split and VF scope usage
  - ability to add pipeline stages via `al.parallel`
- Main risk:
  - missing `no_inline` on ping/pong alpha scopes causes live-range conflicts
  - forgetting `disable_auto_inject_block_sync` when using manual sync
- Read next:
  - [al_scope.md](al_scope.md)

### `al-sync`

- Use when:
  - the kernel uses (or should use) `al.sync_block_set` / `al.sync_block_wait` for cube-vector handoff
  - you need to add ping-pong or triple-buffer pipeline stages
  - L0C binding and fixpipe chain sync is needed
- Signals:
  - data races or deadlocks between cube and vector scopes
  - single-buffer kernel that needs throughput improvement via ping-pong
  - preload kernel with PIPE_STAGES > 2
- Expected benefit:
  - correct and complete sync protocol for any pipeline depth
  - elimination of data races and deadlocks
- Main risk:
  - swapped producer/consumer strings in set/wait
  - omitted pre-loop credit initialization
  - duplicate event IDs for the same pipeline stage
- Read next:
  - [al_sync.md](al_sync.md)

### `al-copy-fractal`

- Use when:
  - the kernel works with NZ (fractal) format tensors on Ascend
  - you need to use `al.fixpipe` (cube→UB), `al.copy` (UB→L1), or `bl.subview`
  - NZ layout conversion strategy needs to be chosen (fp16 vs fp32)
- Signals:
  - `tl.dot` output consumed by vector code (needs fixpipe)
  - softmax P matrix needs to move from UB to L1 for cube PV matmul
  - `sub_vec_id` lane split with L1 subview handoff
- Expected benefit:
  - correct NZ allocation, conversion, and transfer
  - proper handling of fp32 NZ restrictions
- Main risk:
  - fp32 NZ reshape inside outlined VF scope triggers compiler assertion
  - incorrect fractal sizing (N0=8 vs N0=16)
  - missing `is_mem_unique=True` on L0C buffers
- Read next:
  - [al_copy_fractal.md](al_copy_fractal.md)

### `al-scope-args`

- Use when:
  - an outlined VF scope (`al.scope(vector_mode="simd", outline=True)`) triggers a compiler assertion
  - the assertion message mentions `AnalyzeDataLayout` or `collapse_shape`
  - fp32 NZ-shaped tensors are passed into an outlined scope
- Signals:
  - Bishengir compiler crash during compilation of a kernel using outlined VF scopes
  - shape mismatch errors that trace to NZ reshape inside a VF scope
- Expected benefit:
  - understanding of what triggers the assertion
  - correct workaround (ND staging or two-scope structure)
- Main risk:
  - attempting single-pass softmax with fp32 inside an outlined scope
  - using `reshape` that generates `collapse_shape` on a block arg
- Read next:
  - [al_scope_args.md](al_scope_args.md)

### `sub-vec-id-1to2`

- Use when:
  - the kernel mixes vector work and cube work
  - the mixed kernel matches a vector-plus-cube structure where vector work can be split without changing cube math
  - the kernel still needs full-tile `tl.dot` semantics
- Signals:
  - vector-heavy staging or epilogue work around a full-tile cube path
  - a plausible opportunity to split vector work across the two `sub_vec_id()` lanes without changing cube math
- Expected benefit:
  - better vector-unit utilization on A5 mixed kernels
  - larger viable tiles
  - better vector-heavy tail behavior
- Main risk:
  - incorrect lane ownership
  - unsound vector-to-cube handoff
  - partial-dot rewrites that break full-dot semantics
- Read next:
  - [sub_vec_id_1to2.md](sub_vec_id_1to2.md)
