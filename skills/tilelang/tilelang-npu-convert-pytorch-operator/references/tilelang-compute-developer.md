# TileLang Compute — Developer Mode

> **Layer**: Base (Layer 1) — used by both convert and optimize workflows.
> **Source**: [TileLang-Ascend Developer API Reference](../../../TileLang-Ascend%20Developer%20API%20Reference.md)

Developer mode compute primitives: matrix multiply via `T.gemm_v0`, reductions via `T.reduce_*`, and element-wise via `T.Parallel` with symbolic math. The compiler handles CV splitting, sync insertion, and memory planning automatically.

## 1. Matrix Multiply: `T.gemm_v0`

```python
T.gemm_v0(A, B, C, transpose_A=False, transpose_B=False, init=False)
```

Computes `C += A * B`.

- `A`: left input (shared tier, Ascend = L1)
- `B`: right input (shared tier, Ascend = L1)
- `C`: accumulator output (fragment tier, Ascend = L0C)
- `transpose_A` / `transpose_B`: whether to transpose
- `init`: zero C before accumulation — `True` on first k-step

### Standard K-loop pattern

```python
for k in T.serial(T.ceildiv(K, block_K)):
    T.copy(A[bx * block_M, k * block_K], A_L1)
    T.copy(B[k * block_K, by * block_N], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
```

### Complete GEMM

```python
import tilelang
import tilelang.language as T

M, N, K = 1024, 1024, 1024
block_M, block_N, K_L1 = 128, 256, 64

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

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

func = gemm(M, N, K, block_M, block_N, K_L1)
```

## 2. Reductions

```python
T.reduce_sum(src, dst, dim=-1, clear=True, real_shape=None)
T.reduce_max(src, dst, dim=-1, clear=True, real_shape=None)
T.reduce_min(src, dst, dim=-1, clear=True, real_shape=None)
T.reduce_abssum(src, dst, dim=-1)
T.reduce_absmax(src, dst, dim=-1, clear=True)
```

- `dim`: reduction axis. Supports `0`/`1`/`-1`/`-2` depending on buffer rank.
- `out` can use compressed shape (`[M]`) or keepdim shape (`[M, 1]`).
- `clear=True`: zero dst before accumulating (default).
- `clear=False`: merge into existing dst values.

| Reduce | `clear=False` merge rule |
|--------|--------------------------|
| `reduce_sum` | `new = old + result` |
| `reduce_max` | `new = max(old, result)` |
| `reduce_min` | `new = min(old, result)` |

```python
T.reduce_sum(src_ub, dst_ub, dim=-1, clear=True)
T.reduce_max(src_ub, dst_ub, dim=-1, clear=False)
```

## 3. Element-wise: `T.Parallel` + Symbolic Math

`T.Parallel` expresses data-parallel element-wise compute. Inside the loop body, use Python symbolic math operators. The compiler auto-vectorizes.

### Syntax

```python
# 1D
for j in T.Parallel(block_N):
    c_ub[j] = a_ub[j] + b_ub[j]

# 2D
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i, j]
```

### Float operators

| Op | Syntax |
|----|--------|
| add, sub, mul, div | `+`, `-`, `*`, `/` |
| abs | `T.abs(x)` |
| exp | `T.exp(x)` |
| log | `T.log(x)` |
| sqrt | `T.sqrt(x)` |
| rsqrt | `T.rsqrt(x)` |
| max | `T.max(a, b)` |
| min | `T.min(a, b)` |
| relu | `T.max(x, 0)` |

### Integer operators

| Op | Syntax |
|----|--------|
| bitwise and, or, not | `&`, `\|`, `~` |
| left/right shift | `<<`, `>>` |

### Broadcasting

```python
# Row broadcast
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i]   # b_ub broadcast along dim i

# Complex expression — compiler auto-splits into steps
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = a_ub[i, j] * b_ub[i, j] + a_ub[i, j] / b_ub[i, j]
```

## 4. Complete Example: RMSNorm

```python
import tilelang
import tilelang.language as T

def create_rmsnorm_kernel(BLOCK_M=128, EPS=1e-5):
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    }

    @tilelang.jit(out_idx=[1], pass_configs=pass_configs)
    def rmsnorm_kernel(X: T.Tensor((N, D), "float16"),
                        Y: T.Tensor((N, D), "float16")):
        with T.Kernel(T.ceildiv(N, BLOCK_M), threads=2, is_npu=True) as (cid,):
            x_ub   = T.alloc_shared((BLOCK_M, D), "float16")
            y_ub   = T.alloc_shared((BLOCK_M, D), "float16")
            ss_ub  = T.alloc_shared((BLOCK_M,), "float16")
            rms_ub = T.alloc_shared((BLOCK_M,), "float16")

            T.copy(X[cid * BLOCK_M:, :], x_ub)

            # square and sum
            for i, j in T.Parallel(BLOCK_M, D):
                y_ub[i, j] = x_ub[i, j] * x_ub[i, j]
            T.reduce_sum(y_ub, ss_ub, dim=1)

            # rms = rsqrt(mean_sq + eps)
            for i in T.Parallel(BLOCK_M):
                rms_ub[i] = T.rsqrt(ss_ub[i] / D + EPS)

            # normalize
            for i, j in T.Parallel(BLOCK_M, D):
                y_ub[i, j] = x_ub[i, j] * rms_ub[i]

            T.copy(y_ub, Y[cid * BLOCK_M:, :])

    return rmsnorm_kernel()
```

## 5. Other Utilities

```python
T.any_of(buffer)              # True if any element is non-zero
T.all_of(buffer)              # True if all elements are non-zero
T.clamp(dst, min_val, max_val)    # Clamp to [min_val, max_val]
# Ternary: use Python syntax  val = t if cond else f  (cond is a TIR expression)
T.cumsum(src, dst=None, dim=0, reverse=False)  # Prefix sum along dim
```

## Next

When base primitives cannot express an operation (e.g., sort, topk, compare, cast), use `T.tile.*` extensions from [tilelang-compute-expert.md](tilelang-compute-expert.md) — but stay within the auto-managed pass_configs framework.

For manual synchronization and the full Expert programming model, see [tilelang-compute-expert.md](tilelang-compute-expert.md) sync section and [tilelang-memory-expert.md](tilelang-memory-expert.md).
