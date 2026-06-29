---
priority: high
---

# CV Scope Separation with Manual Synchronization

## Summary

Split a kernel body into `T.Scope("C")` (Cube Core) and `T.Scope("V")` (Vector Core) blocks, using explicit `T.set_flag`/`T.wait_flag` pairs to coordinate data handoffs between the Cube MMA unit and the Vector/MTP pipelines.

## Use When

- A kernel mixes `T.gemm_v0` (Cube) operations with element-wise math, reductions, or normalization (Vector) in the same compute flow.
- The auto-managed Developer-mode `pass_configs` produce incorrect results or suboptimal performance due to conservative sync insertion.
- You need explicit control over when MTE copies complete before Cube reads, or when Cube finishes before Vector consumes results.
- The kernel benefits from overlapping Cube compute with MTE data movement, requiring precise pipeline-level synchronization.

## Avoid When

- The kernel is purely element-wise (no `T.gemm_v0`) — scope separation adds complexity without benefit.
- The auto-managed `pass_configs` already produce correct and performant results.
- The kernel structure is simple enough that `T.barrier_all()` after each major step is sufficient.

## Pattern

### Step 1: Disable auto-passes

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
```

Keep `TL_ASCEND_MEMORY_PLANNING: True` unless you need manual buffer reuse.

### Step 2: Allocate buffers before scopes

Buffers shared between scopes must be allocated outside both scopes:

```python
A_L1 = T.alloc_L1((block_M, K_L1), dtype)
B_L1 = T.alloc_L1((K_L1, block_N), dtype)
C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
```

### Step 3: Split into Cube and Vector scopes

```python
with T.Scope("C"):
    for k in T.serial(T.ceildiv(K, K_L1)):
        T.copy(A[bx * block_M, k * K_L1], A_L1)
        T.copy(B[k * K_L1, by * block_N], B_L1)
        T.barrier_all()
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
        T.barrier_all()

with T.Scope("V"):
    T.copy(C_L0, c_ub)
    for i, j in T.Parallel(block_M, block_N):
        c_ub[i, j] = T.max(c_ub[i, j], 0)
    T.copy(c_ub, C[bx * block_M, by * block_N])
```

### Step 4: Add pipeline-level sync for finer control

Replace `T.barrier_all()` with targeted `set_flag`/`wait_flag` when you need to overlap MTE and Cube:

```python
with T.Scope("C"):
    for k in T.serial(T.ceildiv(K, K_L1)):
        T.copy(A[...], A_L1)
        T.copy(B[...], B_L1)
        T.set_flag("mte3", "m", 0)        # signal: data ready

        T.wait_flag("mte3", "m", 0)       # wait: data arrived
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
        T.set_flag("m", "mte3", 1)        # signal: compute done

T.wait_flag("m", "mte3", 1)               # wait: final result ready
T.copy(C_L0, C[...])
```

### Pipeline naming convention

| Pipe | Hardware | Use |
|------|----------|-----|
| `"mte1"` | MTE engine 1 | GM → L1 copy |
| `"mte2"` | MTE engine 2 | GM → UB copy, UB → GM copy |
| `"mte3"` | MTE engine 3 | GM → L1 copy (alt path) |
| `"m"` | Cube MMA | Matrix multiply (T.gemm_v0) |
| `"v"` | Vector core | Element-wise, reduce |
| `"fix"` | Fixed-function | L0C → GM, scalar, control |
| `"ALL"` | All pipelines | `T.barrier_all()` equivalent |

### Step 5: Prime/destroy for init-direction flags

When `T.set_flag` is issued in the init direction (e.g., `"MTE1" → "MTE2"`), the flag must be **consumed** (via `wait_flag`) after the loop to prevent resource exhaustion. This is the "prime/destroy" pattern:

```python
with T.Scope("C"):
    # Prime: set flags so consumer can proceed on first iteration
    T.set_flag("MTE1", "MTE2", SIG_K_L1)
    T.set_flag("MTE1", "MTE2", SIG_V_L1)
    T.set_flag("M", "MTE1", SIG_L0AB)
    T.set_flag("M", "MTE1", SIG_L0AB + 1)

    for g in T.serial(GT):
        # ... loop body that flips each flag with set/wait pairs ...

    # Destroy: consume outstanding init-direction flags
    T.wait_flag("MTE1", "MTE2", SIG_K_L1)
    T.wait_flag("MTE1", "MTE2", SIG_V_L1)
    T.wait_flag("M", "MTE1", SIG_L0AB)
    T.wait_flag("M", "MTE1", SIG_L0AB + 1)
```

Without the destroy step, the compiler reports `[ERROR] Pipe ... has extra set event` — the init-direction `set_flag` has no matching `wait_flag` at the scope boundary.

### Step 6: Cross-core flag communication

When Cube and Vector scopes exchange data via workspace tensors (no direct L0C → UB path), use `T.set_cross_flag`/`T.wait_cross_flag`:

```python
# Cross-core semaphore IDs
SEM_WS_READY = 0  # producer -> consumer: data ready
SEM_WS_FREE  = 1  # consumer -> producer: slot free

with T.Scope("C"):
    T.wait_cross_flag(SEM_WS_FREE)
    # ... compute, write to workspace ...
    T.set_cross_flag("FIX", SEM_WS_READY)  # FIX pipe signals data ready

with T.Scope("V"):
    T.wait_cross_flag(SEM_WS_READY)
    # ... read from workspace, compute ...
    T.set_cross_flag("MTE2", SEM_WS_FREE)  # MTE2 pipe signals slot free
```

**Pipe qualifiers for `set_cross_flag`**:
- `"MTE2"`: MTE2 engine writes to workspace GM (Vector scope output)
- `"MTE3"`: MTE3 engine writes to workspace GM
- `"FIX"`: Fixed-function unit writes to workspace GM (L0C → GM path from Cube scope)

`wait_cross_flag` takes only the semaphore index (no pipe qualifier) — the consumer inherently waits from its own pipeline context.

**Prime FREE flags before the loop**: the consumer must prime FREE semaphores so the producer can write the first slot:

```python
with T.Scope("V"):
    T.set_cross_flag("MTE2", SEM_WS_FREE)   # prime: slot 0 free
    T.set_cross_flag("MTE2", SEM_WS_FREE)   # prime: slot 1 free (if ring depth 2)
    # ... loop ...
```

## What To Verify After Applying

- Run `python3 scripts/tl_sync_lint.py --tier1 --tier2 --tier3 --tier4 <kernel>.py` to catch flag imbalances, cross-core deadlocks, and prime underflows before any on-device test.
- The kernel compiles and produces correct results — manual sync errors often manifest as silent data corruption, not compile errors.
- Each `set_flag` has a matching `wait_flag` with the same event ID.
- The pipeline direction is correct: producer sets, consumer waits.
- Cross-scope buffers are allocated before both scopes.
- Init-direction flags are set before the loop (prime) and consumed after the loop (destroy).
- Every `wait_cross_flag` has a matching `set_cross_flag` in the opposite scope.
- `T.barrier_all()` is still used at the boundaries where full pipeline synchronization is needed.

## Related Patterns

- `double-buffer`: extends this pattern with ping-pong buffering for compute/memory overlap.
- `explicit-memory`: use `T.alloc_ub`/`T.alloc_L1`/`T.alloc_L0*` for precise buffer placement.
- `workspace-pipeline`: full cross-core ring-buffer pipeline with multi-task scheduling — read after mastering CV scope separation.
- `layout-affinity`: add `T.annotate_layout` to L1 buffers used across scopes for optimal MMA throughput.
