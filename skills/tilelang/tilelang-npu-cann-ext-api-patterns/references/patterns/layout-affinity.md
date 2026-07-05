---
priority: medium
---

# Layout Affinity for MMA Operands

## Summary

Use `T.annotate_layout` with `make_zn_layout` / `make_nz_layout` to annotate L1 buffer data layouts for optimal Cube MMA throughput. The compiler uses these hints to optimize MTE copy bursts, MMA operand ordering, and buffer reuse. Essential for resident buffers and double-buffered L1 operands in Expert-mode kernels.

## Use When

- A kernel uses `T.gemm_v0` with explicit L1 buffers (`T.alloc_L1`).
- The kernel keeps a buffer resident across multiple MMA operations (e.g., Q matrix loaded once per tile and reused).
- Performance profiling shows memory-bound behavior on MTE copies or MMA operand staging.
- The kernel uses double-buffered L1 buffers where precise layout optimization matters.

## Avoid When

- The kernel is in Developer mode with `TL_ASCEND_AUTO_CV_COMBINE: True` — the compiler handles layout automatically.
- The kernel is purely element-wise with no MMA operations.
- The performance gain from layout annotation is marginal for the kernel's complexity.

## Pattern

### Step 1: Import layout intrinsics

```python
from tilelang.intrinsics import make_zn_layout, make_nz_layout
```

### Step 2: Allocate L1 buffers

```python
q_l1 = T.alloc_L1([block_M, dim], dtype)     # Q: (M, D)
k_l1 = T.alloc_L1([block_N, dim], dtype)     # K: (N, D) — note: outer dim is block_N
v_l1 = T.alloc_L1([block_N, dim], dtype)     # V: (N, D)
p_l1 = T.alloc_L1([block_M, block_N], dtype) # P: (M, N)
```

### Step 3: Annotate layouts

Place annotations after all `T.alloc_L1` calls, before entering any `T.Scope`:

```python
T.annotate_layout(
    {
        q_l1: make_zn_layout(q_l1),
        k_l1: make_nz_layout(k_l1),
        p_l1: make_zn_layout(p_l1),
        v_l1: make_zn_layout(v_l1),
    }
)
```

### Layout selection rules

#### `make_zn_layout` — ZN (fractal) layout

Contiguous along the **inner** dimension. Use for:
- **MMA left operand (A, Q, P)**: `(M, K)` or `(M, N)` shapes — the Cube reads left operands in ZN format for optimal dot-product accumulation.
- **L1 buffers written by MTE and read by Cube**: the MTE→L0A copy path prefers ZN.
- **UB workspace read by MTE3**: intermediate results written by Vector, consumed by MTE copy to GM.

```python
q_l1: make_zn_layout(q_l1)   # Q(M, D) as MMA left operand
p_l1: make_zn_layout(p_l1)   # P(M, N) as MMA left operand (P·V)
v_l1: make_zn_layout(v_l1)   # V(N, D) as MMA right when accessed as left in transpose
```

#### `make_nz_layout` — NZ (fractal) layout

Contiguous along the **outer** dimension. Use for:
- **MMA right operand (B, K)**: `(N, K)` shape — the Cube reads right operands in NZ format, and `T.copy(..., transpose=True)` from NZ to L0B is efficient.
- **L1 buffers that will be transposed on copy to L0B**: `T.copy(k_l1, l0b, transpose=True)` is optimized for NZ source.

```python
k_l1: make_nz_layout(k_l1)   # K(N, D) as MMA right operand with transpose
```

### Complete example

```python
from tilelang.intrinsics import make_zn_layout, make_nz_layout

with T.Kernel(NUM_CORES, is_npu=True) as (cid, vid):
    q_l1 = T.alloc_L1([block_M, dim], "float16")
    k_l1 = T.alloc_L1([block_N, dim], "float16")
    p_l1 = T.alloc_L1([block_M, block_N], "float16")
    v_l1 = T.alloc_L1([block_N, dim], "float16")

    T.annotate_layout(
        {
            q_l1: make_zn_layout(q_l1),
            k_l1: make_nz_layout(k_l1),
            p_l1: make_zn_layout(p_l1),
            v_l1: make_zn_layout(v_l1),
        }
    )

    # ... rest of kernel with T.Scope("C") / T.Scope("V")
```

## What To Verify After Applying

- `make_zn_layout` is used for MMA left operands (Q, P) and buffers loaded with default (non-transpose) copies.
- `make_nz_layout` is used for MMA right operands (K) that will be transposed on copy to L0B.
- The annotation call is placed **after** all `T.alloc_L1` calls and **before** entering `T.Scope`.
- The kernel compiles without layout-related errors.
- Run correctness check: incorrect layout annotations can silently produce wrong MMA results.
- Run `python3 scripts/tl_sync_lint.py --tier1 --tier2 --tier3 --tier4 <kernel>.py` if the kernel uses manual sync.

## Related Patterns

- `explicit-memory`: use `T.alloc_L1` for the buffers being annotated.
- `double-buffer`: annotate both buffer pairs — each copy of the double-buffered buffer gets the same layout.
- `workspace-pipeline`: layout-annotated L1 buffers for Q, K, P, V operands in cross-core pipeline kernels.
