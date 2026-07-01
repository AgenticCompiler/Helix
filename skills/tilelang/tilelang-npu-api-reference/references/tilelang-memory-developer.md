# TileLang Memory — Developer Mode

> **Layer**: Base (Layer 1) — used by both convert and optimize workflows.
> **Source**: [TileLang-Ascend Developer API Reference](../../../TileLang-Ascend%20Developer%20API%20Reference.md)

Developer mode uses abstract memory tiers. The compiler maps them to hardware units automatically. This is the default recommendation for convert tasks.

## Memory Tier Abstraction

```
Global Memory (GM / HBM)
    │  T.copy
    ├──► Shared (on-chip) ──► L1 Buffer (Cube) or UB (Vector) — compiler picks
    │    T.alloc_shared
    └──► Fragment (register-level) ──► L0C Buffer
         T.alloc_fragment
```

## Allocation APIs

### `T.alloc_shared`

```python
T.alloc_shared(shape, dtype, scope="shared.dyn")
```

Allocates on-chip shared memory. On Ascend:
- GEMM input → L1 Buffer
- Vector compute → Unified Buffer

The compiler infers the correct hardware unit from context.

```python
A_L1 = T.alloc_shared((block_M, block_K), "float16")    # GEMM input → L1
c_ub = T.alloc_shared((block_M, block_N), "float16")    # Vector compute → UB
```

### `T.alloc_fragment`

```python
T.alloc_fragment(shape, dtype, scope="local.fragment")
```

Allocates register-level storage. On Ascend maps to L0C Buffer (MMA accumulator).

```python
C_L0 = T.alloc_fragment((block_M, block_N), "float16")  # accumulator → L0C
```

### `T.alloc_var`

```python
T.alloc_var(dtype, init=None, scope="local.var")
```

Allocates a scalar variable. Use for flags, counters, temporary scalars.

```python
flag    = T.alloc_var("bool", init=False)
counter = T.alloc_var("int32", init=1)
```

## Data Movement: `T.copy`

```python
T.copy(src, dst, coalesced_width=None)
```

Moves data between memory tiers. Transfer size is inferred from buffer shapes.

| src | dst | Description |
|-----|-----|-------------|
| GM | shared | Global → L1 or UB (compiler decides) |
| L1 | L0A | L1 → L0A (Cube left operand) |
| L1 | L0B | L1 → L0B (Cube right operand) |
| fragment | GM | L0C → Global |
| shared | GM | UB → Global |

```python
# Full buffer copy
T.copy(A[bx * block_M, k * block_K], A_L1)

# Partial region copy
T.copy(A[bx * block_M : bx * block_M + block_M, :], A_L1)
```

## Complete Example: Element-wise Add

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

M, N = 1024, 1024
block_M, block_N = 128, 128

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def tile_add(M, N, block_M, block_N, dtype="float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def add_kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            a_ub = T.alloc_shared((block_M, block_N), dtype)
            b_ub = T.alloc_shared((block_M, block_N), dtype)
            c_ub = T.alloc_shared((block_M, block_N), dtype)

            T.copy(A[bx * block_M, by * block_N], a_ub)
            T.copy(B[bx * block_M, by * block_N], b_ub)

            for i, j in T.Parallel(block_M, block_N):
                c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

            T.copy(c_ub, C[bx * block_M, by * block_N])

    return add_kernel

func = tile_add(M, N, block_M, block_N)
```

## Complete Example: GEMM

```python
M, N, K = 1024, 1024, 1024
block_M, block_N, K_L1 = 128, 256, 64

@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def gemm(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.serial(T.ceildiv(K, K_L1)):
                T.copy(A[bx * block_M, k * K_L1], A_L1)
                T.copy(B[k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C[bx * block_M, by * block_N])

    return main
```

## Helper APIs

```python
T.reshape(buffer, new_shape)   # Fixed total element count
T.view(buffer, new_shape)      # Different shape, same data
```

## Next

For explicit hardware-level memory control (`T.alloc_ub` / `T.alloc_L1` / `T.alloc_L0*`), see [tilelang-memory-expert.md](tilelang-memory-expert.md).
