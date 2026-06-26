---
priority: high
---

# Explicit Hardware Memory Allocation

## Summary

Replace abstract `T.alloc_shared` / `T.alloc_fragment` with explicit hardware-level allocation APIs (`T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C`) to gain precise control over buffer placement, sizing, and lifetime.

## Use When

- The compiler's automatic shared→hardware mapping produces suboptimal buffer placement (e.g., a Vector buffer placed in L1 instead of UB).
- You need to control exact buffer sizes to fit within hardware limits (UB size, L1 size, L0C size).
- Double-buffering requires two buffer sets at specific hardware levels.
- The kernel is hitting UB or L1 capacity limits and needs manual buffer sizing.

## Avoid When

- The kernel is simple and the compiler's automatic mapping is correct.
- You are still prototyping the kernel structure — explicit memory adds complexity that makes iteration slower.
- The performance gain from manual placement does not justify the maintenance cost.

## Pattern

### Memory-to-hardware mapping

| Developer API | Expert API | Hardware | Typical Use |
|---|---|---|---|
| `T.alloc_shared` | `T.alloc_L1` | L1 Buffer (Cube cache) | GEMM operand tiles |
| `T.alloc_shared` | `T.alloc_ub` | Unified Buffer (Vector) | Element-wise workspace |
| `T.alloc_fragment` | `T.alloc_L0A` | L0A Buffer | MMA left operand |
| `T.alloc_fragment` | `T.alloc_L0B` | L0B Buffer | MMA right operand |
| `T.alloc_fragment` | `T.alloc_L0C` | L0C Buffer | MMA accumulator |

### GEMM with explicit memory

```python
with T.Scope("C"):
    A_L1  = T.alloc_L1((block_M, K_L1), "float16")     # L1 for A tiles
    B_L1  = T.alloc_L1((K_L1, block_N), "float16")     # L1 for B tiles
    C_L0C = T.alloc_L0C((block_M, block_N), "float")   # L0C accumulator

    for k in T.serial(T.ceildiv(K, K_L1)):
        T.copy(A[bx * block_M, k * K_L1], A_L1)
        T.copy(B[k * K_L1, by * block_N], B_L1)
        T.gemm_v0(A_L1, B_L1, C_L0C, init=(k == 0))

    T.copy(C_L0C, C[bx * block_M, by * block_N])
```

### Vector element-wise with explicit memory

```python
with T.Scope("V"):
    a_ub = T.alloc_ub((block_M, block_N), "float16")   # UB workspace
    b_ub = T.alloc_ub((block_M, block_N), "float16")
    c_ub = T.alloc_ub((block_M, block_N), "float16")

    T.copy(A[bx * block_M, by * block_N], a_ub)
    T.copy(B[bx * block_M, by * block_N], b_ub)

    for i, j in T.Parallel(block_M, block_N):
        c_ub[i, j] = a_ub[i, j] + b_ub[i, j]

    T.copy(c_ub, C[bx * block_M, by * block_N])
```

### Mixed Cube/Vector with explicit memory

```python
# Buffers allocated before scopes — visible in both
A_L1 = T.alloc_L1((block_M, K_L1), "float16")
B_L1 = T.alloc_L1((K_L1, block_N), "float16")
C_L0 = T.alloc_L0C((block_M, block_N), "float")
c_ub = T.alloc_ub((block_M, block_N), "float16")

with T.Scope("C"):
    for k in T.serial(T.ceildiv(K, K_L1)):
        T.copy(A[...], A_L1)
        T.copy(B[...], B_L1)
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
    # L0C → GM workspace (no direct L0C → UB path)
    T.copy(C_L0, C_workspace)
    T.set_cross_flag("mte3", 0)

with T.Scope("V"):
    T.wait_cross_flag(0)
    T.copy(C_workspace, c_ub)
    for i, j in T.Parallel(block_M, block_N):
        c_ub[i, j] = T.max(c_ub[i, j], 0)
    T.copy(c_ub, C[...])
```

## What To Verify After Applying

- Each `T.alloc_*` call targets the correct hardware level for its use: L1 for GEMM operands, UB for Vector workspace, L0C for accumulator.
- Buffer sizes fit within hardware limits. L1 is typically ~64KB per block; UB is ~192KB per core.
- When using `T.Scope("C")` and `T.Scope("V")`, shared buffers are allocated before both scopes.

## Related Patterns

- `cv-sync`: use explicit sync with explicit memory for full Expert-mode control.
- `double-buffer`: allocate two sets of explicit buffers for compute/memory overlap.
