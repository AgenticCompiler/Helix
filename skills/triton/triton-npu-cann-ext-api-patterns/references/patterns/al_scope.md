---
id: al-scope
---

# al.scope Cube/Vector Scope Split

## Summary

Split a CV-fusion kernel body into `core_mode="cube"` and `core_mode="vector"` `al.scope` blocks, and dispatch softmax or element-wise work to the VF unit via `al.scope(vector_mode="simd", outline=True)`.

## Use When

- A kernel mixes `tl.dot` (cube) operations with element-wise math, softmax, or other vector work in the same loop body.
- The vector softmax or element-wise computation could benefit from dedicated VF unit dispatch.
- You need explicit control over cube/vector synchronization boundaries (`disable_auto_inject_block_sync`).
- You are adding pipeline stages (`PIPE_STAGES > 1`) via `al.parallel` and need cube/vector scope separation.

## Avoid When

- The kernel is pure cube work without vector operations.
- The kernel is pure vector work without `tl.dot`.
- No cross-scope buffer handoff is needed (single execution path without staged data movement).

## Signals

### Code

- A kernel function mixes `tl.dot` calls and element-wise operations (exp, max, multiply-add) in the same loop without any scope boundary.
- Buffer allocations (`bl.alloc`) appear without corresponding `al.scope` blocks.
- The kernel uses `tl.load` / `tl.store` for cube-related data without scope separation.

## Related Patterns

- [al-sync](al_sync.md) — every `al.scope` boundary requires explicit sync events for cross-scope data handoff.
- [al-scope-args](al_scope_args.md) — fp32 NZ tensors passed as block arguments into outlined VF scopes trigger a compiler assertion.
- [al-copy-fractal](al_copy_fractal.md) — NZ layout conversion and buffer copy between scopes.

## What To Verify After Applying

- Both cube and vector scopes cover the same iteration space.
- Buffer allocations (`bl.alloc`) are hoisted before both scopes so they are visible in both.
- `tl.full` / `tl.zeros` initial values for vector-side state (`m_i`, `l_i`, `acc`) are placed in the vector scope.
- `disable_auto_inject_block_sync=True` is set when manual `al.sync_block_set/wait` is used.
- Ping/pong alpha scopes use `no_inline=True` with distinct variable names.

---

# Ascend affinity API: `al.scope` reference

## Imports

```python
import triton.language.extra.cann.extension as al
import triton.extension.buffer.language as bl
```

## 1. Core split: `core_mode="cube"` / `core_mode="vector"`

The primary use of `al.scope` in CV-fusion kernels is to split the program body into a **cube** half and a **vector** half. Each AI-core dispatches both; hardware schedules them concurrently.

```python
with al.scope(core_mode="cube"):
    for block_idx in range(start, end, step):
        ...
        _qk_matmul(...)
        _pv_matmul(...)

with al.scope(core_mode="vector"):
    for block_idx in range(start, end, step):
        ...
        _softmax(...)
        _flash_update(...)
```

Rules:
- Both scopes must cover the same iteration space (same loop bounds).
- Cube scope does `tl.load`, `tl.dot`, `al.fixpipe` (cube→UB), `al.sync_block_*`.
- Vector scope does element-wise math, `tl.store`, `al.copy` (UB→L1), `al.sync_block_*`.
- Buffers allocated with `bl.alloc` before the scopes are visible in both.
- `tl.full(...)` initial values for `m_i`, `l_i`, `acc`, `alpha` must be in the vector scope (they are vector-side state).

## 2. VF scope: `vector_mode="simd", outline=True`

Used for softmax computation dispatched to the VF (vector function) unit. The outlined scope becomes a separate function call in the compiled binary.

```python
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        row = al.extract_slice(qk, [loop, 0], [1, BLOCK_N], [1, 1])
        row = row * sm_scale
        tmp_max = al.insert_slice(tmp_max, tl.max(row, 1), [loop], [1], [1])
    m_ij = tl.maximum(m_i, tmp_max, ...)
    for loop in range(BLOCK_M // 2):
        ...
```

Constraints:
- Only `al.extract_slice` / `al.insert_slice` for sub-tensor access inside the scope.
- Do not use raw indexing `tensor[i, :]` inside outlined scopes — use `al.extract_slice`.
- Tensors that are block arguments (passed into the outlined scope from outside) must not be NZ-shaped fp32 (N0=8). See [al_scope_args.md](al_scope_args.md) for the workaround.
- Max 64 elements per `al.extract_slice` call. Unroll manually for BLOCK_N=128 (two 64-element extracts).

## 3. Two-pass softmax pattern

The two-pass structure is mandatory. Single-pass (max+exp in one loop) fails because `al.extract_slice(block_arg, [loop], ...)` inside a VF scope feeding into NZ reshape triggers a compiler assertion.

```
# Pass 1: collect row maxima
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N], [1, 1])
        qk_loop = qk_loop * sm_scale
        row_max = tl.max(qk_loop, 1, propagate_nan=True)
        tmp_max = al.insert_slice(tmp_max, row_max, [loop], [1], [1])
        qk_scale = al.insert_slice(qk_scale, qk_loop, [loop, 0], [1, BLOCK_N], [1, 1])
    m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)

# Pass 2: subtract max, exp, pack to NZ
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N], [1, 1])
        m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])
        qk_loop = qk_loop - m_ij_loop[:, None]
        p_loop = tl.math.exp(qk_loop)
        p_loop_reshape = p_loop.reshape(BLOCK_N // FRACTAL_N0, 1, FRACTAL_N0)
        p_cast_loop = p_loop_reshape.to(cast_dtype)
        p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0],
                               [BLOCK_N // FRACTAL_N0, 1, FRACTAL_N0], [1, 1, 1])
        l_ij = al.insert_slice(l_ij, tl.sum(p_loop, 1), [loop], [1], [1])
```

For BLOCK_N=128, unroll the inner loop body into two halves (`BLOCK_N_UNROLL = BLOCK_N // 2`) to stay within the 64-element extract limit.

## 4. `no_inline=True`

Prevents the compiler from inlining the outlined scope. Used in `fa_fwd_parallel` for the alpha update scopes to ensure the VF dispatch boundary is preserved:

```python
if (sid & 1) == 0:
    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
else:
    with al.scope(vector_mode="simd", outline=True, no_inline=True):
        alpha_pong = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha_pong + l_ij
        m_i = m_ij
```

The two branches must be distinct (ping/pong variables) to avoid live-range conflicts.

## 5. `al.parallel`

Used instead of `range` in preload kernels for the outer block loop. Signals that the compiler may overlap iterations:

```python
for sq_loop_idx in al.parallel(start_block, end_block, step):
    ...
```

This is required when using `PIPE_STAGES > 2` or task-ring preload patterns.

## 6. Kernel launch flags

Include these in the `@triton.jit` kernel call's `**kwargs`:

| Flag | Kernels using it | Purpose |
|---|---|---|
| `debug=True` | All | Enable debug output |
| `disable_auto_inject_block_sync=True` | All with manual sync | Disable compiler's auto sync injection (manual sync via `al.sync_block_set/wait`) |

Always set `disable_auto_inject_block_sync=True` when using `al.sync_block_set/wait` manually; otherwise the compiler may inject conflicting sync events.

## 7. `al.compile_hint`

Gives the compiler a name hint for a live variable to aid register allocation:

```python
al.compile_hint(alpha, "alpha")
al.compile_hint(alpha_pong, "alpha_pong")
```

Use when the compiler might merge two ping-pong scalars into the same register.
