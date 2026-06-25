# TileLang Memory — Expert Mode

> **Layer**: Expert model (Layer 3).
> **Prerequisite**: [tilelang-memory-developer.md](tilelang-memory-developer.md)

Expert mode exposes explicit hardware memory tiers. Use when you need precise control over buffer placement and lifecycle — when the compiler's automatic mapping is insufficient.

## Explicit Hardware Memory

```
Global Memory (GM / HBM)
    │  T.copy
    ├──► L1 Buffer (Cube L1 cache)  ← T.alloc_L1
    │    ├──► L0A (MMA left)   ← T.alloc_L0A
    │    ├──► L0B (MMA right)  ← T.alloc_L0B
    │    └──► L0C (MMA accum)  ← T.alloc_L0C
    └──► UB (Unified Buffer)   ← T.alloc_ub
```

## Allocation APIs

| API | Hardware | Use |
|-----|----------|-----|
| `T.alloc_ub(shape, dtype)` | Unified Buffer | Vector Core workspace |
| `T.alloc_L1(shape, dtype)` | L1 Cache | GEMM operand tiles |
| `T.alloc_L0A(shape, dtype)` | L0A | MMA left-matrix input |
| `T.alloc_L0B(shape, dtype)` | L0B | MMA right-matrix input |
| `T.alloc_L0C(shape, dtype)` | L0C | MMA accumulator |

```python
# GEMM operands in L1
A_L1  = T.alloc_L1((BLOCK_M, BLOCK_K), "float16")
B_L1  = T.alloc_L1((BLOCK_N, BLOCK_K), "float16")

# MMA accumulator in L0C
C_L0C = T.alloc_L0C((BLOCK_M, BLOCK_N), "float16")

# Vector workspace in UB
a_ub = T.alloc_ub((block_M, block_N), "float16")
```

## Developer ↔ Expert Mapping

| Developer (abstract) | Expert (explicit) | Hardware |
|---------------------|-------------------|----------|
| `T.alloc_shared(shape, dtype)` | `T.alloc_L1(shape, dtype)` | L1 Buffer |
| `T.alloc_shared(shape, dtype)` | `T.alloc_ub(shape, dtype)` | Unified Buffer |
| `T.alloc_fragment(shape, dtype)` | `T.alloc_L0A(shape, dtype)` | L0A Buffer |
| `T.alloc_fragment(shape, dtype)` | `T.alloc_L0B(shape, dtype)` | L0B Buffer |
| `T.alloc_fragment(shape, dtype)` | `T.alloc_L0C(shape, dtype)` | L0C Buffer |

## When to Upgrade

| Scenario | Recommendation |
|----------|---------------|
| Convert / daily operator dev | Developer mode — compiler handles mapping |
| Manual double-buffering | Expert mode — precise buffer placement |
| Fine-grained memory reuse | Expert mode — manual buffer lifecycle |
| Performance tuning | Expert mode — disable auto passes, full control |

When upgrading, synchronization must also switch:
- Developer mode: auto-sync via `pass_configs`
- Expert mode: manual `T.barrier_all` / `T.set_flag` / `T.wait_flag`

See [tilelang-compute-expert.md](tilelang-compute-expert.md) for sync primitives.

## Expert pass_configs

```python
pass_configs = {
    # Expert model: disable all automatic passes
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
```

> In practice, `TL_ASCEND_MEMORY_PLANNING` and `TL_ASCEND_AUTO_SYNC` are sometimes kept `True` to reduce manual sync burden. Adjust per kernel.

## Expert GEMM Example — Double Buffering

> Double-buffered L1 + manual `set_flag`/`wait_flag` sync: MTE prefetches the next K-tile
> into one L1 pair while the Cube core computes the current tile from the other pair.

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,      # manual sync
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}

@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def expert_gemm(A: T.Tensor((M, K), "float16"),
                B: T.Tensor((K, N), "float16"),
                C: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (cid, vid):
        # Double-buffered L1 — two pairs so MTE and Cube can overlap
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

                # Wait for current tile to arrive, then compute
                T.wait_flag("mte3", "m", cur)
                T.gemm_v0(A_L1[cur], B_L1[cur], C_L0C, init=(k == 0))

                # Prefetch next tile while Cube computes current one
                if k + 1 < num_k:
                    T.copy(A[m * BLOCK_M:, (k + 1) * BLOCK_K:], A_L1[nxt])
                    T.copy(B[(k + 1) * BLOCK_K:, cid * BLOCK_N:], B_L1[nxt])
                    T.set_flag("mte3", "m", nxt)

            T.copy(C_L0C, C[m * BLOCK_M:, cid * BLOCK_N:])
```
