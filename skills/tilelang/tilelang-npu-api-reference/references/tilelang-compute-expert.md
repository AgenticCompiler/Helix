# TileLang Compute — Expert Mode

> **Layer**: Expert extensions (Layer 2 compute, Layer 3 sync model).
> **Source**: [TileLang-Ascend Expert API Reference](../../../TileLang-Ascend%20Expert%20API%20Reference.md)
> **Prerequisite**: [tilelang-compute-developer.md](tilelang-compute-developer.md)

Two parts:
- **Part 1 — Extended compute** `T.tile.*`: use in convert when base primitives are insufficient (e.g., sort, compare, cast). Stay within auto-managed `pass_configs`.
- **Part 2 — Sync primitives**: Expert programming model. Prefer auto-managed sync; use manual sync when precise pipeline control is needed.

---

## Part 1: Extended Compute `T.tile.*`

> **Convert rule**: prefer `T.Parallel` + symbolic math from [tilelang-compute-developer.md](tilelang-compute-developer.md). Use `T.tile.*` only when the base primitives cannot express the operation. Keep the auto-managed `pass_configs` from Developer mode.

### Math

```python
T.tile.add(dst, src0, src1)          # dst = src0 + src1
T.tile.sub(dst, src0, src1)          # dst = src0 - src1
T.tile.mul(dst, src0, src1)          # dst = src0 * src1
T.tile.div(dst, src0, src1)          # dst = src0 / src1
T.tile.max(dst, src0, src1)          # dst = max(src0, src1)
T.tile.min(dst, src0, src1)          # dst = min(src0, src1)
```

`src0`/`src1` can be buffers or scalars:

```python
T.tile.add(c_ub, a_ub, b_ub)     # buffer + buffer
T.tile.add(c_ub, a_ub, 2)        # buffer + scalar
```

### Unary Math

```python
T.tile.exp(dst, src)              # dst = exp(src)
T.tile.ln(dst, src)               # dst = ln(src)
T.tile.sqrt(dst, src)             # dst = sqrt(src)
T.tile.rsqrt(dst, src)            # dst = 1/sqrt(src)
T.tile.reciprocal(dst, src)       # dst = 1/src
T.tile.abs(dst, src)              # dst = abs(src)
T.tile.relu(dst, src)             # dst = max(0, src)
T.tile.leaky_relu(dst, src, s)    # dst = src if >=0 else src*s
T.tile.axpy(dst, src, scalar)     # dst = scalar*src + dst (in-place)
T.tile.sin(dst, src)              # dst = sin(src)
T.tile.cos(dst, src)              # dst = cos(src)
```

### Bitwise

```python
T.tile.bitwise_and(dst, src0, src1)       # dst = src0 & src1
T.tile.bitwise_or(dst, src0, src1)        # dst = src0 | src1
T.tile.bitwise_not(dst, src)              # dst = ~src
T.tile.bitwise_xor(dst, src0, src1)       # dst = src0 ^ src1
T.tile.bitwise_lshift(dst, src, scalar)   # dst = src << scalar
T.tile.bitwise_rshift(dst, src, scalar)   # dst = src >> scalar
```

### Compare

```python
T.tile.compare(dst, src0, src1, mode)
```

Mode: `"EQ"`, `"NE"`, `"GT"`, `"GE"`, `"LT"`, `"LE"`. Result: bit 1 where true, 0 otherwise.

```python
T.tile.compare(c_ub, a_ub, b_ub, "EQ")   # a == b
T.tile.compare(c_ub, a_ub, 1.0, "GT")    # a > 1.0
```

### Select

```python
T.tile.select(dst, selMask, src0, src1, selMode)
```

Picks from src0 (mask bit=1) or src1 (mask bit=0).

| selMode | When |
|---------|------|
| `"VSEL_CMPMASK_SPR"` | Between two tensors, mask has element limit |
| `"VSEL_TENSOR_SCALAR_MODE"` | Between tensor and scalar |
| `"VSEL_TENSOR_TENSOR_MODE"` | Between two tensors, no mask limit |

### Cast

```python
T.tile.cast(dst, src, mode, count)
```

Mode: `"CAST_NONE"`, `"CAST_RINT"`, `"CAST_FLOOR"`, `"CAST_CEIL"`, `"CAST_ROUND"`, `"CAST_TRUNC"`, `"CAST_ODD"`.

### Data Manipulation

```python
T.tile.transpose(dst, src)               # 16×16 block transpose
T.tile.fill(buffer, value)               # Fill with constant
T.tile.clear(buffer)                     # Zero-fill
T.tile.createvecindex(dst, first)        # Increasing indices from `first`
T.tile.arith_progression(buf, first, diff, count)  # Arithmetic sequence
```

### Gather / Mask

```python
T.tile.gather_mask(dst, src, pattern)
```

Fixed patterns: `"P0101"` (even indices), `"P1010"` (odd), `"P0001"`, `"P0010"`, `"P0100"`, `"P1000"`, `"P1111"` (all).

Custom: pass an index list.

```python
T.tile.gather_mask(b_ub, a_ub, "P0101")
T.tile.gather_mask(b_ub, a_ub, [0, 1, 3, 4, 5, 7, 8, 9])
```

### Gather (by offset)

```python
T.tile.gather(dst, src, src_offset, src_base_addr)
```

Gathers elements from src according to offset/index buffer.

```python
T.tile.gather(c_ub, a_ub, b_ub, 0)
```

### Sort / TopK

```python
T.tile.sort(dst, src, actual_num)                  # Descending sort
T.tile.merge_sort(dst, src0, src1, src2=None, src3=None)  # 2/3/4-way merge
T.tile.topk(dst, src, K, actual_num)               # Top-K descending
```

`src` size must be a multiple of 32 (auto-padded). Output format: `(val0, idx0, val1, idx1, ...)`.

```python
T.tile.sort(dst, src, 131)                       # 131 elements, padded to 160
T.tile.topk(topk_dst, sorted_src, K=10, actual_num=41)
```

### Atomic Add

```python
T.tile.atomic_add(dst_gm, src_local)
```

Atomic accumulate local tensor (UB or L0C) into GM. Zero GM output before calling.

```python
src_ub = T.alloc_ub((tile_n,), "float32")
T.tile.fill(src_ub, 1.0)
T.tile.atomic_add(C[0], src_ub)
```

---

## Part 2: Sync Primitives

> ⚠️ **Expert programming model** — requires manual sync. In Developer mode, sync is handled automatically by `pass_configs`. Prefer the auto-managed approach; use these only when precise control is needed.

### Barriers

```python
T.barrier_all()              # All pipelines synchronized
T.pipe_barrier(pipe)         # Specific pipeline barrier
T.sync_all()                 # All compute units synchronized
```

Pipeline names: `"fix"`, `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`, `"s"`, `"ALL"`

### Intra-core Pipeline Sync

```python
T.set_flag(src, dst, eventId)      # Set sync flag
T.wait_flag(src, dst, eventId)     # Wait for sync flag
```

Pipeline values: `"fix"`, `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`, `"s"`, `"ALL"`

```python
# MTE3 notifies Cube that copy is done
T.set_flag("mte3", "m", 0)
T.wait_flag("mte3", "m", 0)
```

### Cross-core Sync (Cube ↔ Vector)

```python
T.set_cross_flag(pipe, flag)     # Set cross-core flag
T.wait_cross_flag(flag)          # Wait for cross-core flag
```

```python
T.set_cross_flag("MTE3", 0)
T.wait_cross_flag(0)
```

### Expert Model GEMM — Double Buffering + Manual Sync

> MTE prefetches the next K-tile into a second L1 pair while the Cube core computes the
> current tile — all controlled by fine-grained `set_flag`/`wait_flag` synchronization.
> Developer-mode auto-sync cannot express this overlap pattern.

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def expert_gemm(A: T.Tensor((M, K), "float16"),
                B: T.Tensor((K, N), "float16"),
                C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (cid, vid):
        A_L1_0 = T.alloc_L1((BLOCK_M, BLOCK_K), "float16")
        A_L1_1 = T.alloc_L1((BLOCK_M, BLOCK_K), "float16")
        B_L1_0 = T.alloc_L1((BLOCK_K, BLOCK_N), "float16")
        B_L1_1 = T.alloc_L1((BLOCK_K, BLOCK_N), "float16")
        C_L0C  = T.alloc_L0C((BLOCK_M, BLOCK_N), "float16")

        A_L1 = [A_L1_0, A_L1_1]
        B_L1 = [B_L1_0, B_L1_1]

        num_k = T.ceildiv(K, BLOCK_K)

        for m in T.serial(T.ceildiv(M, BLOCK_M)):
            # Prime: prefetch tile 0
            T.copy(A[m * BLOCK_M:, 0:BLOCK_K], A_L1_0)
            T.copy(B[0:BLOCK_K, cid * BLOCK_N:], B_L1_0)
            T.set_flag("mte3", "m", 0)

            for k in T.serial(num_k):
                cur = k % 2
                nxt = 1 - cur

                T.wait_flag("mte3", "m", cur)
                T.gemm_v0(A_L1[cur], B_L1[cur], C_L0C, init=(k == 0))

                if k + 1 < num_k:
                    T.copy(A[m * BLOCK_M:, (k + 1) * BLOCK_K:], A_L1[nxt])
                    T.copy(B[(k + 1) * BLOCK_K:, cid * BLOCK_N:], B_L1[nxt])
                    T.set_flag("mte3", "m", nxt)

            T.copy(C_L0C, C[m * BLOCK_M:, cid * BLOCK_N:])
```

## Relationship to Developer Mode

Expert mode extends — not replaces — Developer mode:

| Category | Developer (base) | Expert (extension) |
|----------|-----------------|-------------------|
| Memory alloc | `T.alloc_shared` / `T.alloc_fragment` / `T.alloc_var` | `T.alloc_ub` / `T.alloc_L1` / `T.alloc_L0*` |
| Data movement | `T.copy` | Reused |
| Matrix multiply | `T.gemm_v0` | Reused |
| Reduce | `T.reduce_*` | Reused |
| Element-wise | `T.Parallel` + symbolic math | `T.tile.add/mul/exp/...` |
| Advanced compute | — | `T.tile.sort/topk/compare/cast/...` |
| Sync | Auto (pass_configs) | `T.barrier_all` / `T.set_flag` / `T.wait_flag` / cross-core |
| Scheduling | `T.Parallel` / `T.Pipelined` / `T.Persistent` | Reused |
