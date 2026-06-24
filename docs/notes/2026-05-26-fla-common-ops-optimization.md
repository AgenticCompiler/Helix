# FLA Common Ops Optimization Review

This note captures reusable optimization knowledge found by comparing:

- `src/kernels/fla/ops/common_origin/`
- `src/kernels/fla/ops/common_ops/`

The findings are review material for future `triton-npu-optimize-knowledge` pattern-card updates. They are intentionally not linked from `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md` yet.

## Scope

The strongest reusable evidence comes from:

- `chunk_scaled_dot_kkt.py`: layout coalescing, grid flattening, algebraic factorization, and mask simplification.
- `chunk_delta_h.py`: full-chunk recurrence hot path, tail peeling, K-specialized dispatch, and pointer-base hoisting.
- `chunk_o.py`: gate tensor layout conversion, masked pointer loads/stores for gate gradients, and launch hint sensitivity.

`chunk_h.py` appears unchanged between origin and optimized versions, so it does not provide new optimization knowledge.

## Candidate 1: Time-Axis Layout Coalescing For Gate-Like Tensors

Suggested target patterns:

- `remove-implicit-transpose.md`
- `layout-materialization-elision.md`
- `layout-store-and-block-pointers.md`
- possibly `cache_use.md` if v1 later imports that v3 pattern

### Summary

When a gate-like tensor is logically shaped `[B, T, HV]` but the kernel repeatedly loads one `(B, HV)` lane across time, materialize it as `[B, HV, T].contiguous()` at the wrapper boundary so the Triton kernel can load the `T` axis with stride 1.

This is not the same as blindly removing a transpose. The host-side layout conversion is useful when it turns a hot strided per-time access into contiguous vector/block-pointer movement and the conversion cost is amortized by the kernel work.

### Use When

- Kernel hot path repeatedly reads `g`, `beta`, or similar per-time scalars for a fixed batch/head.
- Original tensor layout is `[B, T, HV]`, so time access for one head has stride `HV`.
- Kernel access wants a contiguous time vector such as `o_t = chunk * BT + arange(0, BT)`.
- The wrapper can pay `transpose(1, 2).contiguous()` once before one or more heavy kernels.
- Profiling or code inspection suggests strided scalar/block-pointer loads on the original layout are limiting.

### Avoid When

- The tensor is only read once by a tiny kernel and the host transpose dominates total operator time.
- Downstream kernels still require `[B, T, HV]`, causing repeated layout ping-pong.
- Variable-length indexing prevents the transposed layout from giving a simple contiguous time path.
- The main bottleneck is `tl.dot` compute or UB footprint rather than gate/scalar movement.

### Implementation Sketch

```python
# Wrapper
g = g.transpose(1, 2).contiguous()
beta = beta.transpose(1, 2).contiguous()

# Kernel, fixed batch/head lane loads contiguous T.
p_g = tl.make_block_ptr(
    g + (i_b * HV + i_h) * T,
    (T,),
    (1,),
    (chunk_off * BT,),
    (BT,),
    (0,),
)
b_g = tl.load(p_g, boundary_check=(0,))
```

### Verification

- Include the wrapper transpose in the end-to-end operator benchmark when target mode is `operator`.
- Check whether multiple kernels can share the transposed layout; otherwise, the conversion may only move cost around.
- Verify varlen and fixed-length branches separately.
- Confirm output restoration is needed for gradients, such as reducing and permuting `dg` back to `[B, T, HV]`.

## Candidate 2: Chunk Recurrence Tail Peeling

Suggested target pattern:

- Extend `exact-tile-no-boundary-fast-path.md`
- Or create a new subpattern under `scalar-latency-traps.md` if it should be framed as scalar boundary-control removal.

### Summary

For chunked recurrence kernels where only the final chunk can be partial, split the loop into:

- a hot loop over `range(NT - 1)` that assumes full chunks and avoids per-iteration `min`, `tl.where`, and tail masks
- a single tail block that handles `min(NT * BT, T) - 1` and boundary masks

This is a "mostly exact tile" variant of exact-tile fast path. It does not require the whole sequence length to be divisible by `BT`; it only requires all chunks before the last one to be full.

### Use When

- The algorithm processes time in fixed chunks of `BT`.
- The recurrence loop has `NT` chunks and only the last chunk can be partial.
- The original loop computes `min(...)`, tail masks, or `tl.where(m_t, ..., 0)` in every iteration.
- Scalar/control overhead is visible or suspected in the recurrence loop.
- Duplicating the tail block does not create unmaintainable kernel drift.

### Avoid When

- Many chunks can be irregular, not just the final one.
- Boundary masks are part of algorithm semantics, not just tail protection.
- The loop body is so small that duplicating it increases instruction pressure more than it reduces scalar work.
- Correctness depends on using the same expression for full and tail chunks due to numerical sensitivity.

### Implementation Sketch

```python
# Full chunks: no scalar min/mask/where in the hot loop.
for i_t in range(NT - 1):
    last_idx = (i_t + 1) * BT - 1
    b_g_last = tl.load(g_base + last_idx)
    b_g = tl.load(...)
    b_v = b_v * exp(b_g_last - b_g)[:, None]
    ...

# Tail chunk: may be partial, keep boundary logic here.
i_t = NT - 1
last_idx = min(NT * BT, T) - 1
m_t = (i_t * BT + tl.arange(0, BT)) < T
b_g_last = tl.load(g_base + last_idx)
b_g = tl.load(...)
b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
```

### Verification

- Test `T < BT`, `T == BT`, `T % BT == 0`, and `T % BT != 0`.
- For varlen mode, verify each sequence still has the correct local `NT` and tail.
- Confirm the hot-loop IR no longer contains the removed `min` / tail `where` pattern.

## Candidate 3: Gated Pairwise Exp Factorization

Suggested target pattern:

- Add as a new case in `algebraic-optimization.md`.

### Summary

For causal/pairwise tiles that multiply by `exp(g_i - g_j)`, avoid materializing or computing a full pairwise difference matrix when the expression can be factored:

```text
exp(g_i - g_j) = exp(g_i) * exp(-g_j)
```

In a `[BT, BT]` tile, this lets the compiler handle row/column broadcasts from two `[BT]` vectors instead of building `b_g[:, None] - b_g[None, :]` as a full intermediate.

### Use When

- The kernel applies a pairwise gate such as `A_ij *= exp(g_i - g_j)`.
- `g_i` and `g_j` come from the same one-dimensional time vector within a chunk.
- The pairwise difference matrix is only used as input to `exp` and then multiplied into another tile.
- Broadcasted vector factors reduce live intermediates or dependency depth.

### Avoid When

- Numerical semantics depend on a specific evaluation order or range behavior.
- The backend already optimizes the difference matrix into equivalent broadcast factors.
- `g_i - g_j` is reused by multiple downstream expressions, making the explicit diff useful.
- Extra `exp(g)` and `exp(-g)` evaluations cost more than the removed intermediate for the active shapes.

### Implementation Sketch

```python
# Before
b_g_diff = b_g[:, None] - b_g[None, :]
b_A *= exp(b_g_diff)

# After
b_A *= exp(b_g)[:, None] * exp(-b_g)[None, :]
```

If another per-row factor exists, merge it into the row-side broadcast:

```python
b_A *= (b_beta * exp(b_g))[:, None] * exp(-b_g)[None, :]
```

### Verification

- Verify numerical tolerance on large positive/negative `g`.
- Check NaN and inf propagation if the operator has strict semantics.
- Benchmark because this trades one pairwise `exp(diff)` expression for two vector `exp` expressions and broadcast multiplication.
- Inspect IR/profile to ensure the change reduces live matrix temporaries or scalar/vector pressure.

## Candidate 4: Batch-Head Serial Loop Inside A Chunk Program

Suggested target patterns:

- Extend `grid-flatten-and-ub-buffering.md`
- Possibly cross-reference `program-multiple-rows.md`

### Summary

When each `(batch, head)` chunk has small or moderate work and launching one program per `(chunk, batch, head)` creates too many tiny programs, flatten the launch grid to the chunk axis and loop over batch/head lanes inside the program.

This is different from mapping logical tasks to a fixed physical core count. It keeps the grid semantic simple, but increases per-program work density by serializing a lightweight logical axis inside the kernel.

### Use When

- Original grid is `(NT, B * HV)`.
- Each program does one chunk for one batch/head lane and has relatively high fixed overhead.
- `B * HV` is small enough that an inner `for i_bh in range(BH)` is acceptable.
- The per-head work is independent and writes disjoint output regions.
- The extra inner loop does not destroy locality or exceed compile/code-size limits.

### Avoid When

- `B * HV` is large enough that one program becomes too long or under-parallelized.
- Per-head work is heavy Cube compute where serialization would reduce useful parallelism.
- Varlen mapping makes batch/head ownership ambiguous or expensive.
- Different heads have highly imbalanced work.

### Implementation Sketch

```python
pid_t = tl.program_id(0)

for i_bh in range(BH):
    i_b = i_bh // HV
    i_h = i_bh % HV
    # Process this chunk and batch/head lane.
```

Wrapper launch:

```python
kernel[(NT,)](..., BH=B * HV)
```

instead of:

```python
kernel[(NT, B * HV)](...)
```

### Verification

- Compare small and large `B * HV` regimes; this is likely shape-sensitive.
- Check compile time and generated code size if `BH` is large.
- Confirm no output aliasing across the serial loop.
- Use end-to-end benchmark because launch/grid effects may differ from kernel-only timing.

## Candidate 5: K-Dimension Constexpr Specialization

Suggested target patterns:

- `classic-matmul.md`
- `effective-extent-tiling.md`
- possibly a future `shape-specialized-dispatch.md`

### Summary

When a kernel has structurally different hot paths for common K sizes, split dispatch by `K` and compile specialized kernels such as `K == 64` and `K == 128` instead of keeping runtime branches like `if K > 64` inside a generic kernel.

This can remove inactive branch trees, simplify block pointers, reduce scalar control, and let the backend optimize fixed tile shapes more aggressively.

### Use When

- Common shapes concentrate around a few K values such as 64 or 128.
- Generic kernel has runtime branches or repeated `if K > ...` blocks.
- The specialized branches have meaningfully different block-pointer/load/dot structure.
- Wrapper-level dispatch can guard the specialized kernels cleanly.

### Avoid When

- K has many active values and specialization would explode compile/cache size.
- The specialized kernel duplicates too much logic and becomes hard to keep correct.
- The runtime branch is cold or already optimized away by constexpr propagation.
- The main bottleneck is unrelated to K structure.

### Implementation Sketch

```python
if K == 64:
    kernel_k64[grid](...)
elif K == 128:
    kernel_k128[grid](...)
else:
    kernel_generic[grid](...)
```

### Verification

- Validate every dispatch branch, including fallback.
- Compare compile time and first-run overhead.
- Record shape coverage: specialization should target dominant benchmark or production shapes.
- Avoid carrying experimental branches that are slower than the generic path.

## Candidate 6: Boundary Mask Simplification After Safe Boundary Loads

Suggested target patterns:

- `scalar-latency-traps.md`
- `exact-tile-no-boundary-fast-path.md`
- possibly `compile_hint.md`

### Summary

If `tl.load(..., boundary_check=...)` already zero-pads rows outside the valid extent, downstream masks may not need to repeat both row and column validity. Keep only the semantic mask and the still-needed validity dimension.

Example: for a lower-triangular causal tile, row out-of-bounds may already be zero because all row inputs were boundary-loaded as zero. The final mask can sometimes drop `m_t[:, None]` and keep only column validity plus the triangular predicate.

### Use When

- Inputs contributing to invalid rows are loaded with `boundary_check` and safe `other`/zero behavior.
- A later `tl.where` repeats row validity even though invalid rows already compute to safe zeros.
- The remaining mask still protects invalid stores or invalid columns.
- The removed mask is boundary-only, not part of causal/math semantics.

### Avoid When

- Invalid row computations can produce nonzero values through constants, bias, or reused state.
- Store still needs row protection and `boundary_check` is not present on the store.
- Removing row validity can expose invalid pointer arithmetic before the masked operation.
- NaN propagation from invalid rows could change semantics.

### Implementation Sketch

```python
# Before
m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)

# After, when row OOB is already zero-padded by prior loads.
m_A = (o_t[:, None] > o_t[None, :]) & m_t[None, :]
```

### Verification

- Test partial chunks and exact chunks.
- Verify invalid rows are truly zero before the final mask.
- Inspect NaN behavior if invalid loaded values could have used `other=nan` or if math introduces NaNs.
- Confirm store boundary checks still cover output bounds.

## Candidate 7: Launch Hint Sensitivity For Ascend Triton Kernels

Suggested target patterns:

- `compile_hint.md`
- `autotune.md`

### Summary

Ascend-specific launch options such as `multibuffer`, `set_workspace_multibuffer`, and `enable_auto_bind_sub_block` should be treated as kernel-local tuning knobs. They are not universal defaults.

The same file can have one kernel that benefits from `enable_auto_bind_sub_block=True` and another that uses `False`, depending on whether auto binding helps the active load/compute structure.

### Use When

- Kernel structure is already stable and correctness is passing.
- Profiling suggests overlap/binding behavior or workspace buffering may affect latency.
- Existing code has mixed launch hints across related kernels.
- A small bounded A/B test can compare immediate parent vs child.

### Avoid When

- Main optimization opportunity is still algorithm/layout/grid structure.
- The hint is copied from another kernel without evidence.
- Results are compared only against an old baseline, not the immediate parent.
- The option changes compile behavior so much that first-run overhead pollutes measurements.

### Implementation Sketch

```python
kernel[grid](
    ...,
    multibuffer=True,
    limit_auto_multi_buffer_of_local_buffer="no-limit",
    set_workspace_multibuffer=2,
    enable_auto_bind_sub_block=False,
)
```

### Verification

- Benchmark with enough warmup to exclude compile effects.
- Record the exact option set in the optimization round summary.
- Test options per kernel; do not apply a file-wide rule automatically.
- Prefer autotune when multiple launch/meta options interact.

## Suggested Review Order

1. Confirm whether Candidate 1 should be framed as "host-side beneficial layout materialization" rather than "remove transpose".
2. Decide whether Candidate 2 deserves a new pattern or should extend `exact-tile-no-boundary-fast-path`.
3. Add Candidate 3 as the next case in `algebraic-optimization.md` if the numerical semantics are acceptable.
4. Add Candidate 4 to `grid-flatten-and-ub-buffering.md` only if the shape gate around `BH` is clear enough.
5. Decide whether Candidate 5 belongs in existing `classic-matmul` / `effective-extent-tiling`, or needs a new shape-specialized dispatch pattern.
6. Keep Candidates 6 and 7 as smaller sections unless future evidence shows they deserve standalone pattern cards.
