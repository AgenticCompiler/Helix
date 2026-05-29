# Ascend affinity API: scope arguments and workarounds

## The problem

Tensors that are **block arguments** (captured from the outer function and passed into an outlined VF scope) must not be **NZ-shaped fp32** (N0=8). When such a tensor is used inside an outlined `al.scope(vector_mode="simd", outline=True)`, the Bishengir AnalyzeDataLayout pass asserts during compilation.

The assertion triggers when:
1. The tensor has NZ layout with `FRACTAL_N0=8` (fp32).
2. It is a block argument of the outlined scope (i.e., it was defined outside the scope and is read or written inside).
3. Any reshape operation on it produces a `collapse_shape` IR node (e.g., `[1, 16] → [2, 1, 1, 8]` is a collapse, not an expand).

For fp16/bf16 (N0=16), a reshape of the form `[1, N0] → [1, 1, N0]` or `[1, BLOCK_N] → [N1, 1, N0]` is a pure `expand_shape` and is safe.

## What is safe inside an outlined VF scope

**Safe**:
- `al.extract_slice(tensor, offsets, sizes, strides)` — reads a sub-slice. Max 64 elements per call.
- `al.insert_slice(dst, src, offsets, sizes, strides)` — writes a sub-slice into a dst tensor.
- Element-wise ops: `+`, `-`, `*`, `tl.math.exp`, `tl.max`, `tl.maximum`, `tl.sum`, `tl.where`.
- `tl.full(...)`, `tl.zeros(...)` — constants.
- `reshape` of a local (non-block-arg) tensor when it is a pure expand (adds dimensions, no collapse).
- `.to(cast_dtype)` — type cast.

**Unsafe**:
- `reshape` on a block-arg tensor that produces `collapse_shape` (any reshape that merges dims).
- `al.extract_slice(block_arg, [loop], ...)` feeding directly into an NZ reshape for fp32.
- Using `p_nz` (fp32, NZ-shaped, `[N1, M, N0]`) as a block arg of the outlined scope.

## Workaround: ND staging for fp32

Build the P matrix in plain ND (`p_temp`) inside the scope, then convert to NZ **outside** the scope.

```python
# Allocation (outside all scopes):
p_temp = bl.alloc(tl.float32, [BLOCK_M // 2, BLOCK_N], al.ascend_address_space.UB)
p_temp = bl.to_tensor(p_temp)

p_nz = bl.alloc(cast_dtype, [BLOCK_N // FRACTAL_N0, BLOCK_M // 2, FRACTAL_N0],
                al.ascend_address_space.UB)
p_nz = bl.to_tensor(p_nz)

# Inside outlined VF scope — fp32 path:
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N], [1, 1])
        m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])
        qk_loop = qk_loop - m_ij_loop[:, None]
        p_loop = tl.math.exp(qk_loop)
        p_temp = al.insert_slice(p_temp, p_loop, [loop, 0], [1, BLOCK_N], [1, 1])
        l_ij = al.insert_slice(l_ij, tl.sum(p_loop, 1), [loop], [1], [1])

# Outside — NZ conversion:
p_nz = tl.permute(
    p_temp.reshape((BLOCK_M // 2) // FRACTAL_M0, FRACTAL_M0,
                    BLOCK_N // FRACTAL_N0, FRACTAL_N0),
    (2, 0, 1, 3),
)
```

`p_temp` is never NZ-shaped, so it is always safe as a block argument.

## BLOCK_N=128 manual unroll with safe paths

For BLOCK_N=128, the max 64-element `al.extract_slice` limit requires manual unrolling. The two-pass softmax is written as:

```python
# Pass 1: collect row maxima
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        qk_loop = al.extract_slice(qk, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
        qk_loop_unroll = al.extract_slice(qk, [loop, BLOCK_N_UNROLL],
                                          [1, BLOCK_N_UNROLL], [1, 1])
        row_max = tl.maximum(
            tl.max(qk_loop * sm_scale, 1),
            tl.max(qk_loop_unroll * sm_scale, 1),
        )
        tmp_max = al.insert_slice(tmp_max, row_max, [loop], [1], [1])
    m_ij = tl.maximum(m_i, tmp_max, propagate_nan=tl.PropagateNan.ALL)

# Pass 2: exp and NZ pack
BLOCK_N_UNROLL = BLOCK_N // 2
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N_UNROLL], [1, 1])
        qk_loop_unroll = al.extract_slice(qk_scale, [loop, BLOCK_N_UNROLL],
                                          [1, BLOCK_N_UNROLL], [1, 1])
        m_ij_loop = al.extract_slice(m_ij, [loop], [1], [1])
        p_loop = tl.math.exp(qk_loop - m_ij_loop[:, None])
        p_loop_unroll = tl.math.exp(qk_loop_unroll - m_ij_loop[:, None])

        # fp16/bf16 NZ packing (FRACTAL_N0=16):
        p_nz = al.insert_slice(p_nz,
            p_loop.reshape(BLOCK_N_UNROLL // FRACTAL_N0, 1, FRACTAL_N0).to(cast_dtype),
            [0, loop, 0], [BLOCK_N_UNROLL // FRACTAL_N0, 1, FRACTAL_N0], [1, 1, 1])
        p_nz = al.insert_slice(p_nz,
            p_loop_unroll.reshape(BLOCK_N_UNROLL // FRACTAL_N0, 1, FRACTAL_N0).to(cast_dtype),
            [BLOCK_N_UNROLL // FRACTAL_N0, loop, 0],
            [BLOCK_N_UNROLL // FRACTAL_N0, 1, FRACTAL_N0], [1, 1, 1])

        l_ij = al.insert_slice(l_ij, tl.sum(p_loop + p_loop_unroll, 1), [loop], [1], [1])
```

Note: for fp32 with BLOCK_N=128, use the ND staging pattern (p_temp+permute) applied to both halves, with the NZ conversion outside the scope.

## `al.extract_slice` / `al.insert_slice` API

```python
result = al.extract_slice(src, offsets, sizes, strides)
dst    = al.insert_slice(dst, src, offsets, sizes, strides)
```

- `offsets`: list of per-dim start positions
- `sizes`: list of per-dim counts
- `strides`: list of per-dim steps (usually all 1)

All three lists must have the same length as the tensor rank.

Examples:
```python
# Extract row `loop` from [BLOCK_M//2, BLOCK_N] → [1, BLOCK_N]:
row = al.extract_slice(qk, [loop, 0], [1, BLOCK_N], [1, 1])

# Extract scalar from [BLOCK_M//2] → [1]:
m_val = al.extract_slice(m_ij, [loop], [1], [1])

# Insert [1, BLOCK_N] slice at row `loop` of [BLOCK_M//2, BLOCK_N]:
buf = al.insert_slice(buf, row, [loop, 0], [1, BLOCK_N], [1, 1])

# Insert [N1, 1, N0] NZ block at position [0, loop, 0]:
p_nz = al.insert_slice(p_nz, nz_block, [0, loop, 0], [N1, 1, N0], [1, 1, 1])
```

## Single-pass softmax: why it fails

A single-loop softmax that computes `max`, `exp`, and inserts into a `p_nz` block arg in one pass:

```python
# WRONG for fp32 — single pass inside outlined scope:
with al.scope(vector_mode="simd", outline=True):
    for loop in range(BLOCK_M // 2):
        row = al.extract_slice(qk, [loop, 0], [1, BLOCK_N], [1, 1])
        row_max = tl.max(row, 1)
        p_loop = tl.math.exp(row - row_max[:, None])
        # This insert into NZ-shaped p_nz (block arg, fp32) triggers assertion:
        p_nz = al.insert_slice(p_nz, p_loop.reshape(N1, 1, N0).to(cast_dtype), ...)
```

The `p_nz` block arg is NZ-shaped fp32; the reshape on `p_loop` is fine (it is local), but the insertion modifies `p_nz` which is the block arg — the compiler sees it as an NZ write to a block arg and asserts.

The fix is either:
1. Use ND `p_temp` inside (fp32 workaround above).
2. Use two separate outlined scopes (pass 1 produces `tmp_max`; pass 2 produces `p_nz`), where `p_nz` is **not** a block arg of either scope but is allocated before and modified via `insert_slice` that writes local data — this works for fp16/bf16 where N0=16 avoids the assertion.
