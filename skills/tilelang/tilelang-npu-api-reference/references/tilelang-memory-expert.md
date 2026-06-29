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

## Affinity / Layout Annotation

```python
from tilelang.intrinsics import make_zn_layout, make_nz_layout
```

### `T.annotate_layout`

Annotate buffer layouts for data locality on Ascend NPU. The compiler uses these hints to optimize MTE copy bursts and MMA operand ordering. Critical for resident buffers and double-buffered L1 operands in Expert-mode kernels.

```python
T.annotate_layout(
    {
        q_l1: make_zn_layout(q_l1),    # (M, K) → ZN: contiguous in inner dim
        k_l1: make_nz_layout(k_l1),    # (N, K) → NZ: contiguous in outer dim
        p_l1: make_zn_layout(p_l1),    # (M, N) → ZN: optimal for MMA left
        v_l1: make_zn_layout(v_l1),    # (N, K) → ZN: optimal for MMA right (as left operand in V·P)
    }
)
```

### Layout Factory Functions

| Function | Hardware mapping | When to use |
|----------|-----------------|-------------|
| `make_zn_layout(buffer)` | ZN (fractal) layout — contiguous along inner dimension | MMA left operand (A, P), GEMM accum input, UB workspace |
| `make_nz_layout(buffer)` | NZ (fractal) layout — contiguous along outer dimension | MMA right operand (B, K, V), transpose-friendly layouts |

### Rules of thumb

- **A matrix (M×K)**: `make_zn_layout` — the Cube reads left operands in ZN format.
- **B matrix (N×K) or K matrix (N×K)**: `make_nz_layout` — optimal for MMA right operand / transpose copies.
- **Workspace (M×N)**: `make_zn_layout` — intermediate tensors written by FIX pipe, read by MTE2.
- Place `T.annotate_layout` once **after all `T.alloc_L1` / `T.alloc_ub` calls** and **before** entering any `T.Scope`.

### Example from flash_attn_opt.py

```python
from tilelang.intrinsics import make_zn_layout, make_nz_layout

q_l1 = T.alloc_L1([block_M, dim], dtype)
k_l1 = T.alloc_L1([block_N, dim], dtype)
v_l1 = T.alloc_L1([block_N, dim], dtype)
p_l1 = T.alloc_L1([block_M, block_N], dtype)

T.annotate_layout(
    {
        q_l1: make_zn_layout(q_l1),
        k_l1: make_nz_layout(k_l1),
        p_l1: make_zn_layout(p_l1),
        v_l1: make_zn_layout(v_l1),
    }
)
```

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
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # required — manual T.Scope
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,         # recommended
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # either
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,      # recommended
}
```

> Only `AUTO_CV_COMBINE` is **required** to be `False` — the compiler must not interfere with manual `T.Scope`. The other three are recommendations for full manual control; leaving them `True` adds redundant compiler passes but will not break correctness.

## Workspace Tensors (Cross-Core Communication)

When Cube and Vector scopes need to exchange data (no direct L0C → UB path), use workspace tensors allocated as GM `T.Tensor` parameters and staged through the `workspace_idx` decorator parameter.

```python
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def flash_attention_fwd(
    ...
    workspace_1: T.Tensor([NUM_CORES, RING, NR, block_M, block_N], dtype),  # S
    workspace_2: T.Tensor([NUM_CORES, RING, NR, block_M, block_N], dtype),  # P
    workspace_3: T.Tensor([NUM_CORES, RING, NR, block_M, dim], dtype),      # P·V
):
```

### Workspace tensor layout

| Dimension | Meaning | Typical value |
|-----------|---------|---------------|
| `NUM_CORES` | Core count for core-local addressing | 24 (910B) |
| `RING` | Pipeline depth (concurrent tasks) | 2–4 |
| `NR` | Big-block sub-blocks per task | 1–16 |
| `block_M × block_N` | MMA tile dimensions | 128 × 128 |

### Why workspace tensors?

L0C (Cube accumulator) cannot be `T.copy`'d directly to UB (Vector buffer). The data path is:

```
L0C → (FIX pipe) → workspace GM → (MTE2 pipe) → UB
```

Workspace tensors provide this intermediate staging area, with `T.set_cross_flag`/`T.wait_cross_flag` coordinating the producer (Cube/FIX) and consumer (Vector/MTE2).

### Cross-core flag naming convention

```python
# Producer (Cube scope) — signals data is ready
T.set_cross_flag("FIX", SEM_WS1_READY)    # FIX pipe sets READY for ws1
# Consumer (Vector scope) — waits for data, signals slot free after use
T.wait_cross_flag(SEM_WS1_READY)          # Vector waits for ws1 data
T.set_cross_flag("MTE2", SEM_WS1_FREE)    # MTE2 pipe signals ws1 slot reusable
```

Pipe qualifiers on `set_cross_flag`:
- `"MTE2"` — MTE2 engine writes to workspace (vector scope output)
- `"MTE3"` — MTE3 engine writes to workspace
- `"FIX"` — Fixed-function unit writes (L0C → GM path)

## Expert GEMM Example — Double Buffering

> Double-buffered L1 + manual `set_flag`/`wait_flag` sync: MTE prefetches the next K-tile
> into one L1 pair while the Cube core computes the current tile from the other pair.

```python
import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
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
