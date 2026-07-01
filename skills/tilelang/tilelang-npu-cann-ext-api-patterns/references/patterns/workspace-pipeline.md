---
priority: high
---

# Cross-Core Workspace Pipeline (Ring Buffer)

## Summary

Use GM workspace tensors as a ring buffer between Cube and Vector scopes, with `T.set_cross_flag`/`T.wait_cross_flag` pairs controlling a multi-task pipeline. Each workspace slot carries NR consecutive KV-block results, amortizing cross-core synchronization overhead. Pattern used in arch22 FlashAttention — three tasks kept in flight simultaneously to keep the Cube continuously fed (MTE2/Cube bound).

## Use When

- A kernel mixes `T.gemm_v0` (Cube) and element-wise / softmax (Vector) in a multi-step pipeline.
- Each task produces intermediate results (S = Q·K^T, P = softmax(S), P·V) that must flow Cube → Vector → Cube across cores.
- `T.Pipelined` cannot express cross-core data handoff.
- The kernel needs maximum throughput — the Cube pipe must never drain between tasks.

## Avoid When

- The kernel has no cross-core dependency (pure Cube-only or Vector-only).
- A single `T.barrier_all()` between scopes is sufficient.
- The task count per tile is 1 — ring buffer overhead exceeds benefit.

## Pattern

### Step 1: Design the pipeline depth and workspace layout

Choose `RING` (pipeline depth, typically 3 for 3-task schedules) and `NR` (KV blocks per task, the big-block tiling factor):

```python
NUM_CORES = 24
RING = 3
NR = 8   # S2 = 128 * NR KV tokens per task
block_M, block_N = 128, 128
```

### Step 2: Allocate workspace tensors

Workspace tensors are GM tensors passed as kernel parameters and marked with `workspace_idx`:

```python
@tilelang.jit(out_idx=[3], workspace_idx=[4, 5, 6], pass_configs=pass_configs)
def kernel(
    ...,
    workspace_1: T.Tensor([NUM_CORES, RING, NR, block_M, block_N], dtype),  # S = Q·K^T
    workspace_2: T.Tensor([NUM_CORES, RING, NR, block_M, block_N], dtype),  # P = softmax(S)
    workspace_3: T.Tensor([NUM_CORES, RING, NR, block_M, dim], dtype),      # O = P·V
):
```

### Step 3: Define cross-core semaphores

Use READY/FREE pairs — one pair per workspace per ring slot:

```python
SEM_WS1_READY = 0  # C -> V : ws1 has data
SEM_WS1_FREE  = 1  # V -> C : ws1 slot free
SEM_WS2_READY = 2  # V -> C : ws2 has data
SEM_WS2_FREE  = 3  # C -> V : ws2 slot free
SEM_WS3_READY = 4  # C -> V : ws3 has data
SEM_WS3_FREE  = 5  # V -> C : ws3 slot free
```

### Step 4: Prime FREE flags in the consumer scope

The consumer (Vector) starts by priming FREE flags for all ring slots, so the producer (Cube) can write immediately:

```python
with T.Scope("V"):
    T.set_cross_flag("MTE2", SEM_WS1_FREE)
    T.set_cross_flag("MTE2", SEM_WS1_FREE)
    T.set_cross_flag("MTE2", SEM_WS1_FREE)
```

### Step 5: Ring-indexed task loop

Each global task index `g` maps to a ring slot `g % RING`:

```python
for g in T.serial(GT + 1):
    # ===== MM1(g): S = Q·K^T -> ws1[cid, g%3, :, :] =====
    if g < GT:
        r1 = g % RING
        T.wait_cross_flag(SEM_WS1_FREE)  # slot available?

        for nr in T.serial(NR):
            # ... copy K tile, MMA S = Q·K^T, write to workspace_1[cid, r1, nr, :, :]

        T.set_cross_flag("FIX", SEM_WS1_READY)  # ws1[r1] (all NR) ready

    # ===== MM2(g-1): O_partial = P·V -> ws3[cid, (g-1)%3, :, :] =====
    if g >= 1:
        r2 = (g - 1) % RING
        T.wait_cross_flag(SEM_WS2_READY)  # P ready?

        for nr in T.serial(NR):
            # ... copy V tile, read P from workspace_2, MMA P·V, write to workspace_3[cid, r2, nr, :, :]

        T.set_cross_flag("FIX", SEM_WS3_READY)  # ws3[r2] ready
        T.set_cross_flag("MTE2", SEM_WS2_FREE)  # ws2[r2] slot free
```

### Step 6: Mirror the pattern in the Vector scope

```python
with T.Scope("V"):
    for g in T.serial(GT + 1):
        # ===== Vec1(g): softmax ws1[r1] -> ws2[r1] =====
        if g < GT:
            r1 = g % RING
            T.wait_cross_flag(SEM_WS1_READY)  # S ready?

            for nr in T.serial(NR):
                # ... softmax on workspace_1[cid, r1, nr, ...], write P to workspace_2[cid, r1, nr, ...]

            T.set_cross_flag("MTE2", SEM_WS1_FREE)   # ws1[r1] slot free
            T.set_cross_flag("MTE3", SEM_WS2_READY)  # P ready

        # ===== Vec2(g-1): accumulate P·V =====
        if g >= 1:
            r2 = (g - 1) % RING
            T.wait_cross_flag(SEM_WS3_READY)

            for nr in T.serial(NR):
                # ... accumulate O = O * alpha + P·V from workspace_3[cid, r2, nr, ...]

            T.set_cross_flag("MTE2", SEM_WS3_FREE)  # ws3[r2] slot free
```

### Pipeline timing diagram (3-task ring, NR=1 simplified)

```
Task  g=0      g=1      g=2      g=3
C MM1: [=S0=]  [=S1=]  [=S2=]  [=S3=]
C MM2:          [=P0·V=][=P1·V=][=P2·V=]
V Vec1:         [soft0] [soft1] [soft2]
V Vec2:                  [acc0]  [acc1]
Time  ---->
```

Cube dual-issues MM1(g) and MM2(g-1); Vector dual-issues Vec1(g) and Vec2(g-1). Three tasks are always in flight.

### Complete example: 3-task ring pipeline skeleton

See `DLBlas/tilelang/fa/flash_attn_opt.py` for a production implementation with nRatio big-block tiling, resident Q reuse, and online softmax threading.

## What To Verify After Applying

- Run `python3 scripts/tl_sync_lint.py --tier1 --tier2 --tier3 --tier4 <kernel>.py` before any on-device test. Check for:
  - `CROSS_DEADLOCK`: every `wait_cross_flag` has a matching `set_cross_flag` in the opposite scope.
  - `FLAG_IMBALANCE`: every `set_flag`/`wait_flag` pair has balanced multiplicity.
  - `PRIME_UNDERFLOW`: FREE flags are primed before the loop, READY flags are not waited before being set.
- Verify `NUM_CORES × RING × NR × 128 × 128` workspace size fits within HBM budget (~2-4 GB total for 3 workspaces at typical sizes).
- The ring depth `RING` is >= the pipeline depth (number of concurrent tasks overlapping).
- FREE flags are primed (multiple times) in the consumer scope before the task loop.
- Pipe qualifiers match: `"FIX"` for L0C → GM writes, `"MTE2"` for UB → GM writes, `"MTE3"` for UB → GM writes.
- The `g < GT` / `g >= 1` guard structure matches the pipeline prologue/epilogue (g=0: only MM1; g=GT: only MM2).

## Related Patterns

- `cv-sync`: basic CV scope separation — read first before this pattern.
- `double-buffer`: intra-core L1/L0 double buffering — complements this pattern for MTE/Cube overlap within each task.
- `layout-affinity`: use `T.annotate_layout` + `make_zn_layout`/`make_nz_layout` on L1 buffers for optimal MMA throughput.
