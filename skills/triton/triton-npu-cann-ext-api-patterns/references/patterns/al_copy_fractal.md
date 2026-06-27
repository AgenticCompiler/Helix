---
id: al-copy-fractal
---

# NZ Fractal Layout and Buffer Copy

## Summary

Handle the Ascend NZ (fractal) tensor layout for cube operands, allocate UB/L1/L0C buffers with `bl.alloc`, transfer data between cube and vector using `al.fixpipe` (L0C→UB) and `al.copy` (UB→L1), and choose the correct NZ conversion strategy for fp16 vs fp32.

## Use When

- A kernel uses `tl.dot` and the result must be consumed by vector code (requires NZ→ND conversion via fixpipe).
- Softmax output (the P matrix) needs to move from UB to L1 for a subsequent cube PV matmul via `al.copy`.
- The kernel uses `sub_vec_id` lane split with L1 subview handoff.
- Cube operands or intermediates are in NZ format and need correct allocation sizing.

## Avoid When

- The kernel is entirely element-wise (no `tl.dot`, no cube involvement).
- The kernel uses only global memory (HBM) without UB/L1 staging.

## Signals

### Code

- `tl.dot` appears but no `al.fixpipe` follows to move the NZ result to a UB buffer for vector consumption.
- Softmax output is written directly to global memory instead of being staged to L1 for a cube matmul.
- `bl.alloc` is called without specifying the correct address space (`UB`, `L1`, or `L0C`).
- NZ-shaped tensor allocations use incorrect fractal sizing (e.g., `N0=16` for fp32 or `N0=8` for fp16).

### Profile

- Kernel performance is limited by unnecessary global-memory round-trips between cube and vector work.
- Buffer-related compiler errors mention `AnalyzeDataLayout` or incorrect NZ shape.

## Related Patterns

- [al-scope](al_scope.md) — `al.fixpipe` and `al.copy` bridge cube and vector scopes.
- [al-sync](al_sync.md) — every `al.fixpipe` or `al.copy` transfer needs matching sync events.
- [al-scope-args](al_scope_args.md) — fp32 NZ conversion inside outlined VF scopes requires the ND staging workaround.
- [sub-vec-id-1to2](sub_vec_id_1to2.md) — `bl.subview` with `sub_vec_id` for lane-split L1 handoff.

## What To Verify After Applying

- Fractal `N0` and `M0` match the dtype (`N0=8` for fp32, `N0=16` for fp16/bf16; `M0=16` always).
- `al.fixpipe` destination is a UB buffer with correct ND shape.
- `al.copy` source is a UB buffer and destination is an L1 (sub)buffer.
- `is_mem_unique=True` is set on all L0C allocations and on UB buffers that must not alias.
- For fp32 within outlined VF scopes: ND staging (`p_temp`) is used instead of direct NZ `insert_slice`.

---

# Ascend affinity API: copy and fractal format reference

## NZ (fractal NZ) layout

The cube unit on Ascend operates on NZ (fractal) format tensors, not row-major ND tensors.

**Shape**: 4D `[N1, M1, M0, N0]`

| Dtype | N0 | M0 | Fractal size |
|---|---|---|---|
| fp16 / bf16 | 16 | 16 | 16×16×2B = 512B |
| fp32 | 8 | 16 | 16×8×4B = 512B |

These are set at kernel launch:
```python
if q.dtype == torch.float32:
    fractal_m0, fractal_n0 = 16, 8
else:
    fractal_m0, fractal_n0 = 16, 16
```

For a logical `[BLOCK_M, BLOCK_N]` tile in NZ format:
- Physical shape: `[BLOCK_N // N0, BLOCK_M // M0, M0, N0]`
- Example fp16 `[32, 64]` → `[4, 2, 16, 16]`
- Example fp32 `[32, 64]` → `[8, 2, 16, 8]`

L1 allocation for NZ P matrix:
```python
p_l1 = bl.alloc(
    cast_dtype,
    [BLOCK_N // FRACTAL_N0, BLOCK_M // FRACTAL_M0, FRACTAL_M0, FRACTAL_N0],
    al.ascend_address_space.L1,
)
```

## `bl.alloc` — buffer allocation

```python
buf = bl.alloc(dtype, shape, address_space, is_mem_unique=False)
```

- `dtype`: `tl.float32`, `tl.float16`, `tl.bfloat16`, etc.
- `shape`: list of integers (all dims)
- `address_space`: `al.ascend_address_space.UB`, `.L1`, or `.L0C`
- `is_mem_unique=True`: required for L0C buffers and for UB buffers that the compiler must not alias (e.g., `pv_ub`, `acc_buffer` in preload kernels)

## `bl.to_tensor` / `bl.to_buffer`

```python
tensor = bl.to_tensor(buf)                    # buffer → tensor (readable by triton ops)
tensor = bl.to_tensor(buf, target_shape=[M, N])  # reshape on conversion (for NZ→2D)
buf    = bl.to_buffer(tensor, address_space)  # tensor → buffer (for fixpipe, al.copy)
buf    = bl.to_buffer(tensor, bind_buffer=other_buf)  # bind alias
```

The `bind_buffer=` form is used in the preload kernel to bind the L0C cube output alias before fixpipe:
```python
bl.to_buffer(qk, bind_buffer=qk_l0c)   # qk is the tl.dot result; qk_l0c is the L0C buf
```

## `al.fixpipe` — cube L0C → vector UB

Transfers a cube NZ result to a vector UB buffer, optionally halving rows for `sub_vec_id` split:

```python
al.fixpipe(src_nz_tensor, dst_ub_buffer, dma_mode, dual_dst_mode)
```

- `src_nz_tensor`: the `tl.dot` result tensor (in L0C NZ format)
- `dst_ub_buffer`: `bl.to_buffer(tensor, al.ascend_address_space.UB)` — target UB buffer
- `dma_mode`: `al.FixpipeDMAMode.NZ2ND` (always for converting NZ→ND)
- `dual_dst_mode`: `al.FixpipeDualDstMode.ROW_SPLIT` — each sub_vec lane gets half the rows

Example from `_qk_matmul`:
```python
qk = tl.dot(q, trans_k)  # result is NZ in L0C
al.fixpipe(
    qk,
    bl.to_buffer(bl.to_tensor(qk_ub), al.ascend_address_space.UB),
    al.FixpipeDMAMode.NZ2ND,
    al.FixpipeDualDstMode.ROW_SPLIT,
)
```

After fixpipe, `qk_ub` (a `[BLOCK_M//2, BLOCK_N]` UB buffer) holds the lane's half of the QK result.

## `al.copy` — UB → L1

Transfers from a UB buffer to an L1 sub-buffer. Used to move the softmax P matrix from vector UB to L1 for the cube PV matmul.

```python
al.copy(src_ub_buffer, dst_l1_subview)
```

Example from the vector scope:
```python
al.copy(
    bl.to_buffer(p_nz.reshape(BLOCK_N // FRACTAL_N0,
                              (BLOCK_M // 2) // FRACTAL_M0,
                              FRACTAL_M0, FRACTAL_N0),
                 al.ascend_address_space.UB),
    p_l1_ping_sub,   # bl.subview of p_l1_ping for this lane's half
)
```

## `bl.subview` — sub-buffer view

Creates a view into a buffer covering a contiguous sub-region. Used to address a sub_vec lane's half of the L1 P matrix:

```python
p_l1_sub = bl.subview(
    p_l1,
    offsets=[0, sub_vec_id * ((BLOCK_M // 2) // FRACTAL_M0), 0, 0],
    sizes=[BLOCK_N // FRACTAL_N0, (BLOCK_M // 2) // FRACTAL_M0, FRACTAL_M0, FRACTAL_N0],
    strides=[1, 1, 1, 1],
)
```

The subview is then passed to `al.copy` as destination. Both vector lanes write to disjoint sub-regions of the same `p_l1` L1 buffer, which the cube reads as a full `[BLOCK_M, BLOCK_N]` tile.

Full NZ tile in cube via `bl.to_tensor` with `target_shape`:
```python
p_l1_tensor = bl.to_tensor(p_l1, target_shape=[BLOCK_M, BLOCK_N])
pv = tl.dot(p_l1_tensor, v)
```

## NZ layout conversion (vector side)

### fp16 / bf16 (FRACTAL_N0=16): direct reshape + permute

For fp16/bf16, the reshape `[1, BLOCK_N] → [N1, 1, N0]` is a pure expand_shape and is safe inside an outlined VF scope:

```python
p_loop_reshape = p_loop.reshape(BLOCK_N // FRACTAL_N0, 1, FRACTAL_N0)
p_cast_loop = p_loop_reshape.to(cast_dtype)
p_nz = al.insert_slice(p_nz, p_cast_loop, [0, loop, 0],
                       [BLOCK_N // FRACTAL_N0, 1, FRACTAL_N0], [1, 1, 1])
```

This builds `p_nz` with shape `[N1, BLOCK_M//2, N0]` inside the scope.

After the scope (in the vector core path, outside any `al.scope`), reshape to 4D before `al.copy`:
```python
p_nz.reshape(BLOCK_N // FRACTAL_N0, (BLOCK_M // 2) // FRACTAL_M0, FRACTAL_M0, FRACTAL_N0)
```

### fp32 (FRACTAL_N0=8): p_temp + permute outside scope

For fp32, inserting a `[1, N0]` slice into an NZ-shaped tensor that is a block arg of an outlined scope triggers the Bishengir AnalyzeDataLayout assertion. The workaround: build in ND (`p_temp`) inside the scope, convert to NZ outside.

```python
# Inside outlined VF scope:
for loop in range(BLOCK_M // 2):
    qk_loop = al.extract_slice(qk_scale, [loop, 0], [1, BLOCK_N], [1, 1])
    p_loop = tl.math.exp(qk_loop - m_ij_loop[:, None])
    p_temp = al.insert_slice(p_temp, p_loop, [loop, 0], [1, BLOCK_N], [1, 1])
    ...

# Outside the scope — NZ conversion:
p_nz = tl.permute(
    p_temp.reshape(
        (BLOCK_M // 2) // FRACTAL_M0, FRACTAL_M0,
        BLOCK_N // FRACTAL_N0, FRACTAL_N0,
    ),
    (2, 0, 1, 3),
)
```

`p_temp` shape: `[BLOCK_M//2, BLOCK_N]` (ND, fp32).  
After permute: `[N1, M1, M0, N0]` fp32 in NZ layout.

Do NOT attempt `[1, BLOCK_N] → [N1, 1, 1, N0]` reshape inside the scope for fp32; it generates a `collapse_shape` on the block arg and fails the AnalyzeDataLayout pass.

### `fa_fwd_Affinity` style (simple, fp16 only, outside any outlined scope)

In the non-outlined single-buffer kernel, the full reshape+permute can be done directly in the vector core path:

```python
p_fractal = p_cast.reshape(
    (BLOCK_M // 2) // FRACTAL_M0, FRACTAL_M0,
    BLOCK_N // FRACTAL_N0, FRACTAL_N0,
)
p_nz = tl.permute(p_fractal, (2, 0, 1, 3))
```

This works because the vector scope is not an outlined VF scope — there is no block-arg restriction.

## Summary: choosing the conversion strategy

| Context | Dtype | Method |
|---|---|---|
| Outlined VF scope | fp16/bf16 | `reshape([N//N0, 1, N0])` + `insert_slice` inside scope; reshape to 4D outside |
| Outlined VF scope | fp32 | Build ND `p_temp` inside scope; `permute(reshape(...), (2,0,1,3))` outside |
| Non-outlined vector scope | Any | Direct `reshape` + `tl.permute` outside any scope |
