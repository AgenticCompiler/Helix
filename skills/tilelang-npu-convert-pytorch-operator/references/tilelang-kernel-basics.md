# TileLang Kernel Basics

## Required imports

```python
import tilelang
import tilelang.language as T
```

## Kernel definition pattern

A kernel is defined with `@T.prim_func` inside a factory function decorated with `@tilelang.jit`.

`out_idx` marks output parameters (negative = from end). `workspace_idx` marks workspace buffers auto-managed by the compiler:

```python
@tilelang.jit(out_idx=[-1])                   # last param is output
@tilelang.jit(out_idx=[-2, -1])               # last two params are outputs
@tilelang.jit(out_idx=[4,5,6,7], workspace_idx=[8])  # multi-output + workspace

def tile_operator(M: int, N: int, block_M: int, block_N: int, dtype: str = "float16"):
    @T.prim_func
    def kernel(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, vid):
            # ... compute ...
    return kernel
```

## Kernel launch variants

```python
# Default: dual vector units (cid, vid)
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    ...

# Vid reduction: single vector unit (cid only)
with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
    ...
```

## Scope contexts

```python
with T.Scope("V"):   # Vector unit scope (element-wise ops)
    ...
with T.Scope("C"):   # Cube unit scope (gemm ops)
    ...
```

## Loop constructs

| Construct | Use |
|-----------|-----|
| `T.serial(N)` | Sequential iteration |
| `T.Parallel(M, N)` | Element-wise data-parallel |
| `T.Pipelined(iters, num_stages=N)` | Compute/memory overlap |
| `T.Persistent(domain, core_num, cid)` | Cache-friendly core scheduling |
| `T.unroll(N)` | Loop unrolling hint |
| `T.ceildiv(a, b)` | Ceil-division for loop bounds |

## Autotune

TileLang supports `@tilelang.autotune` for automatic configuration search:

```python
@tilelang.autotune(
    configs=[{"block_M": 128, "block_N": 128, "K_L1": 64},
             {"block_M": 256, "block_N": 128, "K_L1": 64}],
    ref_prog=lambda A, B: A @ B,       # reference implementation
    supply_prog=lambda params: [a, b],  # input tensors
    atol=1e-2, rtol=1e-2,
)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    ...
```

## JIT cache

```python
tilelang.cache.clear_cache()  # Clear JIT cache before compilation
```

## Kernel inspection

```python
kernel = tilelang.engine.lower(func)  # Lower to kernel source without JIT
print(kernel.kernel_source)           # Inspect generated AscendC code
```

## Data types

Supported: `float16`, `float32` (`"float"`), `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`.

For gemm accumulator, prefer `float32`; on A5 camodel, `"float"` is required.

## JIT pass configs

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

For CV separation and workspace reduction:
```python
"tl.ascend_auto_cv_combine": True,
"tl.ascend_auto_cross_core_sync": True,
```
