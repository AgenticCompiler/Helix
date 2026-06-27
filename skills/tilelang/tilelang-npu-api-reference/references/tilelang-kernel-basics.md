# TileLang Kernel Basics

> **Layer**: Shared infrastructure — used by both Developer and Expert modes.

Shared infrastructure: `@tilelang.jit`, `T.Kernel`, `@T.prim_func`, loop constructs, `pass_configs`, JIT cache, autotune, and debugging.

## Minimal Runnable Skeleton

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
def tile_operator(M: int, N: int, block_M: int, block_N: int, dtype: str = "float16"):
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def kernel(
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

    return kernel


func = tile_operator(M, N, block_M, block_N)
```

## Compilation: `@tilelang.jit` vs `tilelang.compile`

| | `@tilelang.jit` | `tilelang.compile` |
|---|---|---|
| Input | Factory function returning `PrimFunc` | `PrimFunc` directly |
| Use when | Full kernel definition in one factory | Already have a `PrimFunc` |
| Compile timing | At call site (module load) | Explicit call |
| Returns | Callable kernel object | `JITKernel` instance |

```python
# @tilelang.jit — factory returns PrimFunc, auto-compiles at call
@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def my_kernel_factory(M, N, block_M, block_N, dtype="float16"):
    @T.prim_func
    def kernel(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
        ...
    return kernel

func = my_kernel_factory(1024, 1024, 128, 128)  # compiles here

# tilelang.compile — explicit path
func = tilelang.compile(prim_func, out_idx=[-1], target="ascend")
```

### `@tilelang.jit` Full Signature

```python
tilelang.jit(
    func=None,
    *,
    out_idx: list[int] | int | None = None,
    workspace_idx: list[int] | int | None = None,
    target: str | Target = "auto",
    target_host: str | Target | None = None,
    platform: str = "auto",
    execution_backend: str = "cython",
    verbose: bool = False,
    pass_configs: dict | None = None,
    debug_root_path: str | None = None,
) -> JITImpl
```

## Parameter Marking: `out_idx`

```python
@tilelang.jit(out_idx=[-1])                              # last param is output
@tilelang.jit(out_idx=[-2, -1])                          # last two params are outputs
@tilelang.jit(out_idx=[4, 5, 6, 7], workspace_idx=[8])   # multi-output + workspace
```

`out_idx` marks kernel output parameters (negative = from end). `workspace_idx` marks workspace buffers auto-managed by the compiler.

> **Critical**: When `out_idx` marks a parameter as output, the compiled `@tilelang.jit` kernel **returns** a new tensor for each output index — it does **not** modify the passed-in Python tensor in-place. Always capture the return value:
> ```python
> # CORRECT
> y = kernel(x, y)
> # WRONG — y stays uninitialized / zeros
> kernel(x, y)
> ```
> This is the single most common convert bug. If you pass a pre-allocated tensor and do not capture the return, the kernel allocates a fresh output and returns it — silently discarding your buffer.

## Kernel Launch Variants

```python
# Legacy: two-index launch (cid, vid) — vid is a block-internal partition index
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    ...

# Vid elimination (threads=2): single index (cid only) — recommended
with T.Kernel(m_num * n_num, threads=2, is_npu=True) as (cid):
    ...
```

`T.Kernel(n_blocks: int, is_npu: bool = False, ...)`:
- `n_blocks` — total block count
- `is_npu` — `True` for Ascend NPU (default: `False`)
- `threads` — vector core parallelism. Ascend: `1` or `2`
- Yields `(cid, vid)` or `(cid,)` depending on thread count

> **Syntax trap**: `as (cid,)` with a trailing comma tries to unpack the return value as a tuple, causing `TypeError: cannot unpack non-iterable Var object`. Use `as cid` (simple variable) or `as (cid)` (single-element tuple without trailing comma). When `threads=1` (no vid), the context manager yields a single value — same syntax applies.

### Vid Elimination

When `threads=2`, vid is eliminated — the compiler handles vector core partitioning automatically:

```python
# Before vid elimination
with T.Kernel(n_blocks, is_npu=True) as (cid, vid):
    a_ub = T.alloc_shared((block_M // VEC_NUM, block_N), dtype)
    T.copy(A[bx * block_M + vid * block_M // VEC_NUM, ...], a_ub)

# After vid elimination (threads=2) — recommended
with T.Kernel(n_blocks, threads=2, is_npu=True) as (cid):
    a_ub = T.alloc_shared((block_M, block_N), dtype)
    T.copy(A[bx * block_M, ...], a_ub)
```

> **Vid elimination only splits `T.Parallel`**: the compiler auto-splits `T.Parallel` iteration spaces across the two cores, but does **not** split `T.serial`. Both cores independently execute the identical `T.serial` body — redundant work, wasting half the vector compute. Simultaneous writes from both cores to the same GM address are also a hardware-level data race:
> 
> | Loop on vid-split dim | Safe? |
> |---|---|
> | `T.Parallel(block_M, ...)` | ✅ compiler auto-splits |
> | `T.serial(block_M)` | ❌ both cores do identical work, race on output |
> | `T.serial(num_k)` on K dim | ✅ K is not the split dim |
> 
> **Fix**: rewrite `T.serial` on the split dimension as `T.Parallel`, or use `threads=1`.

## Loop Constructs

| Construct | Use Case |
|-----------|----------|
| `T.serial(N)` | Sequential iteration (k-loop in gemm) |
| `T.Parallel(M, N)` | Element-wise data-parallel, auto-vectorized |
| `T.Pipelined(N, num_stages=2)` | Overlap compute and data movement |
| `T.Persistent(domain, wave_size, idx)` | Cache-friendly core scheduling |
| `T.ceildiv(a, b)` | Ceil-division for loop bounds |
| `T.loop_break()` | Early loop exit |

```python
# Serial k-loop
for k in T.serial(T.ceildiv(K, K_L1)):
    T.copy(A[bx * block_M, k * K_L1], A_L1)
    T.copy(B[k * K_L1, by * block_N], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

# Pipelined — num_stages controls overlap depth
for k in T.Pipelined(loop_k, num_stages=3):
    T.copy(A[...], A_L1)
    T.copy(B[...], B_L1)
    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

# Element-wise parallel
for i, j in T.Parallel(block_M, block_N):
    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
```

> `T.WarpSpecialize` and `T.ws` are NVIDIA Hopper-specific — do NOT use for Ascend NPU.

## PassConfigKey Reference

```python
pass_configs = {
    # Split Cube (matrix) and Vector (element-wise) ops across cores automatically
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    # Auto-insert barrier / set_flag / wait_flag between copies and compute
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    # Analyze buffer lifetimes, reuse freed memory to reduce peak footprint
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    # Auto-insert set_cross_flag / wait_cross_flag between Cube and Vector cores
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}
```

All four auto passes enabled — the compiler handles partitioning, synchronization, memory reuse, and cross-core coordination. This is the recommended configuration for convert tasks.

### Additional Advanced Keys

| Key | Type | Effect |
|-----|------|--------|
| `tir.disable_vectorize` | `bool` | Disable vectorization |
| `tl.disable_tma_lower` | `bool` | Disable TMA lower (GPU only, not applicable to Ascend) |
| `tl.disable_warp_specialized` | `bool` | Disable warp specialization (GPU only) |
| `tl.config_index_bitwidth` | `int` | Config index bitwidth |
| `tl.disable_dynamic_tail_split` | `bool` | Disable dynamic tail split |
| `tl.disable_safe_memory_legalize` | `bool` | Disable safe memory legalize |

## JIT Cache

```python
tilelang.set_cache_dir(path)        # Set cache directory
tilelang.get_cache_dir() -> str     # Get cache directory
tilelang.enable_cache()             # Enable caching
tilelang.disable_cache()            # Disable caching
tilelang.is_cache_enabled() -> bool # Check cache status
```

## Autotune

```python
@tilelang.autotune(
    configs=[
        {"block_M": 128, "block_N": 128, "K_L1": 64},
        {"block_M": 256, "block_N": 128, "K_L1": 64},
        {"block_M": 128, "block_N": 256, "K_L1": 64},
    ],
    ref_prog=lambda A, B: A @ B,          # reference for correctness
    supply_prog=lambda params: [a, b],     # input tensor supplier
    atol=1e-2,
    rtol=1e-2,
)
def matmul(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    ...
```

## Kernel Inspection & Debugging

```python
# Lower to kernel source without full JIT
kernel = tilelang.engine.lower(func)
print(kernel.kernel_source)

# From a compiled JITKernel
source = kernel.get_kernel_source()  # Ascend C source
tir    = kernel.get_tir()            # TIR representation

# Set log level
tilelang.set_log_level("DEBUG")      # or "INFO", "WARNING", "ERROR"
```

## Data Types

| Category | Types |
|----------|-------|
| Float | `"float16"`, `"bfloat16"`, `"float32"` (`"float"`) |
| Int (signed) | `"int8"`, `"int16"`, `"int32"`, `"int64"` |

> `"float64"` is available on CUDA/CPU backends but **not on Ascend NPU** — the Ascend `_dtype` codegen map does not include it.
| Int (unsigned) | `"uint8"`, `"uint16"`, `"uint32"`, `"uint64"` |

Rules:
- GEMM accumulator: prefer `"float32"` (`"float"`)
- GEMM inputs: typically `"float16"` for performance

## Top-Level Utilities

```python
tilelang.cdiv(a, b)            # Ceil-division: ceil(a / b)
tilelang.next_power_of_2(x)    # Next power of 2: 3→4, 5→8, 9→16
tilelang.TensorSupplyType      # Enum: Auto / Random / Zeros / Ones / Incremental
tilelang.set_log_level(level)  # Set module log level
```

## Reference Files

For detailed API documentation organized by layer:

| File | Contents |
|------|----------|
| [tilelang-memory-developer.md](tilelang-memory-developer.md) | Base memory: `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_var`, `T.copy` |
| [tilelang-memory-expert.md](tilelang-memory-expert.md) | Explicit hardware memory: `T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0*` |
| [tilelang-compute-developer.md](tilelang-compute-developer.md) | Base compute: `T.gemm_v0`, `T.reduce_*`, `T.Parallel` + symbolic math |
| [tilelang-compute-expert.md](tilelang-compute-expert.md) | Extended compute `T.tile.*` and sync primitives |
