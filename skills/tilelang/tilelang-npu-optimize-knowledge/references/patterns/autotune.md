---
priority: high
---

# TileLang-Ascend Autotune Decision Pattern

## Summary

Use TileLang autotune as the default way to search block sizes, K tile dimensions, and pass config options when the kernel structure is already reasonable and the main open question is parameter choice.

Treat this pattern as a routing rule: try direct config lists first, generate candidates programmatically when the search space is large, and fall back to hand-picked configurations only when the space must be constrained tightly.

## Use When

- The kernel structure already looks semantically correct, and the likely headroom is in `block_M`, `block_N`, `K_L1`, or `pass_configs` options.
- The current optimization loop is drifting toward repeated manual tiling edits without strong evidence that a structural rewrite is needed first.
- The factory function exposes free tuning parameters (block sizes, K tile) that are not hard-coded.
- The operator is compute-bound (GEMM-heavy) or a mixed compute/memory pattern where tiling trade-offs matter.

## Avoid When

- The real problem is structural — a manual element-wise loop that should first become a proper `T.gemm_v0` with serial K-loop.
- All relevant tuning parameters are already fixed at module level, so the factory function exposes no meaningful search space.
- A semantic constraint fixes one block dimension so tightly that generated candidates would mostly be invalid (e.g., `block_N` must equal a feature dimension).
- The kernel is correctness-fragile under repeated benchmarking without proper result validation hooks.

## What To Verify After Applying

- Verify the chosen config list is the least manual one that still produces meaningful candidates for the kernel shape.
- Verify the reference program (`ref_prog`) produces the correct expected output — it is the correctness oracle during autotune.
- Verify `supply_prog` generates inputs matching the kernel's expected shapes and dtypes.
- Verify `atol` and `rtol` are set to reasonable values for the operator's precision requirements.
- Verify the searched parameters are Ascend-relevant: `block_M`, `block_N`, `K_L1`, and optionally `pass_configs` directives.
- Verify `tilelang.cache.clear_cache()` is called between distinct autotune runs if re-compilation is needed.

## Route 1: Direct Config List

Use a static list of config dicts when the search space is small and the tuning dimensions are well-understood.

Typical signals:

- 2–3 tuning dimensions with a constrained set of values each
- The factory function signature clearly separates block sizes from shape parameters
- You already have a rough idea of the good region from manual experimentation

```python
@tilelang.autotune(
    configs=[
        {"block_M": 128, "block_N": 128, "K_L1": 64},
        {"block_M": 256, "block_N": 128, "K_L1": 64},
        {"block_M": 128, "block_N": 256, "K_L1": 64},
        {"block_M": 256, "block_N": 256, "K_L1": 32},
        {"block_M": 128, "block_N": 128, "K_L1": 32},
    ],
    ref_prog=lambda A, B: A @ B,
    supply_prog=lambda params: [
        torch.randn(M, K, dtype=torch.float16, device="npu"),
        torch.randn(K, N, dtype=torch.float16, device="npu"),
    ],
    atol=1e-2,
    rtol=1e-2,
)
def gemm(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    ...
```

## Route 2: Programmatic Config Generation

Generate configs programmatically when the search space is larger or follows a pattern.

Typical signals:

- The tuning space is combinatorial (e.g., all combinations of 3+ dimensions)
- Block sizes must satisfy divisibility constraints with the problem dimensions
- You want to sweep a range rather than enumerate specific points

```python
def get_configs():
    configs = []
    for block_M in [256, 128, 64]:
        for block_N in [256, 128, 64]:
            for K_L1 in [64, 32, 16]:
                if block_M * K_L1 <= L1_BYTE_LIMIT and block_N * K_L1 <= L1_BYTE_LIMIT:
                    configs.append({"block_M": block_M, "block_N": block_N, "K_L1": K_L1})
    return configs


@tilelang.autotune(
    configs=get_configs(),
    ref_prog=lambda A, B: A @ B,
    supply_prog=lambda params: [
        torch.randn(M, K, dtype=torch.float16, device="npu"),
        torch.randn(K, N, dtype=torch.float16, device="npu"),
    ],
    atol=1e-2,
    rtol=1e-2,
)
def gemm(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    ...
```

## Route 3: Tuning `pass_configs`

When block sizes are settled but pass-configuration trade-offs remain (e.g., pipelining depth, CV combine behavior), parametrize the pass configs themselves.

Typical signals:

- The kernel already has stable block sizes but pipeline depth or CV mode may matter
- You want to compare `T.Pipelined` with different `num_stages` values
- CV combine on/off significantly changes performance for mixed Cube/Vector kernels

```python
def get_configs():
    return [
        {"num_stages": ns, "auto_cv_combine": cv}
        for ns in [2, 3]
        for cv in [True, False]
    ]


def make_kernel(block_M, block_N, K_L1, num_stages, auto_cv_combine):
    pass_configs = {
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: auto_cv_combine,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    }

    @tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
    def kernel_factory(M, N, K, dtype="float16", accum_dtype="float"):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), threads=2, is_npu=True) as (cid):
                ...
                for k in T.Pipelined(T.ceildiv(K, K_L1), num_stages=num_stages):
                    T.copy(...)
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                ...
        return main
    return kernel_factory(...)


@tilelang.autotune(
    configs=get_configs(),
    ref_prog=lambda A, B: A @ B,
    supply_prog=lambda params: [
        torch.randn(M, K, dtype=torch.float16, device="npu"),
        torch.randn(K, N, dtype=torch.float16, device="npu"),
    ],
    atol=1e-2,
    rtol=1e-2,
)
def gemm(M, N, K, block_M, block_N, K_L1, num_stages, auto_cv_combine, dtype="float16", accum_dtype="float"):
    return make_kernel(block_M, block_N, K_L1, num_stages, auto_cv_combine)
```

## Ascend-Specific Notes

- Default tuning space for GEMM-heavy kernels: `block_M`, `block_N`, `K_L1`. These control L1 tile sizes and K-loop trip count — the dominant Ascend tuning knobs.
- `K_L1` directly affects L1 buffer pressure and K-loop overhead. Smaller `K_L1` reduces L1 usage but increases loop trip count. Typical range: 16–128.
- For mixed Cube/Vector kernels, also consider tuning `TL_ASCEND_AUTO_CV_COMBINE` — enabling it lets the compiler merge Cube/Vector scopes, which can reduce sync overhead at the cost of some control.
- On A3, the Cube core is the primary compute unit. Focus tuning on GEMM tile shapes and K-loop structure rather than Vector-core micro-optimizations.
- Call `tilelang.cache.clear_cache()` before re-running autotune with different pass_configs to avoid stale compiled artifacts.

## When Automatic Tuning Is Not Enough

Prefer explicit config lists when:

- The kernel has few meaningful free parameters (e.g., only one block size)
- A semantic rule ties one parameter to another (e.g., `block_N == D // 2`)
- The search space is small enough that manual enumeration is faster than setting up autotune infrastructure
- The reference program is expensive to run and you want to minimize evaluation count

## Related Patterns

- `tiling`: use it first when the kernel still needs a better tiled structure before any search space should be explored.
- `software-pipeline`: use it when the tile structure is already good and the next issue is overlap quality rather than parameter choice.
