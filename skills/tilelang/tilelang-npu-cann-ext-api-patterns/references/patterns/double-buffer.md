---
priority: high
---

# Double Buffering with Manual Synchronization

## Summary

Use two sets of L1/UB buffers (ping-pong) with `T.set_flag`/`T.wait_flag` synchronization to overlap MTE data prefetch with Cube compute, hiding memory latency behind computation.

## Use When

- The K-loop in a GEMM kernel is memory-bound — MTE copy time dominates over Cube compute time.
- The kernel has sufficient L1/UB capacity to hold two buffer sets.
- `T.Pipelined` (auto software pipeline) is insufficient or you need finer control over the overlap depth.
- You are already in Expert mode (all auto-passes OFF) with explicit memory allocation.

## Avoid When

- The kernel is compute-bound — double buffering won't help if Cube is the bottleneck.
- L1/UB capacity cannot hold two complete buffer sets.
- `T.Pipelined(num_stages=2)` already achieves the desired overlap with less code complexity.
- The kernel is still in Developer mode with auto-passes ON.

## Pattern

### Step 1: Disable auto-passes, allocate double buffers

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
}
```

```python
# Double-buffered L1 — two pairs
A_L1_0 = T.alloc_L1((block_M, K_L1), "float16")
A_L1_1 = T.alloc_L1((block_M, K_L1), "float16")
B_L1_0 = T.alloc_L1((block_N, K_L1), "float16")
B_L1_1 = T.alloc_L1((block_N, K_L1), "float16")
C_L0C  = T.alloc_L0C((block_M, block_N), "float16")

A_L1 = [A_L1_0, A_L1_1]
B_L1 = [B_L1_0, B_L1_1]
```

### Step 2: Prime the first tile

Prefetch tile 0 before entering the K-loop so the Cube core has data to work on immediately:

```python
# Prime: prefetch tile 0
T.copy(A[m * block_M:, 0:K_L1], A_L1_0)
T.copy(B[0:K_L1, cid * block_N:], B_L1_0)
T.set_flag("mte3", "m", 0)    # signal: tile 0 ready
```

### Step 3: K-loop with ping-pong

On each iteration: wait for the current tile, compute, then prefetch the next tile while Cube is busy:

```python
for k in T.serial(num_k):
    cur = k % 2
    nxt = 1 - cur

    # Wait for current tile to arrive, then compute
    T.wait_flag("mte3", "m", cur)
    T.gemm_v0(A_L1[cur], B_L1[cur], C_L0C, init=(k == 0))

    # Prefetch next tile while Cube computes current one
    if k + 1 < num_k:
        T.copy(A[m * block_M:, (k + 1) * K_L1:], A_L1[nxt])
        T.copy(B[(k + 1) * K_L1:, cid * block_N:], B_L1[nxt])
        T.set_flag("mte3", "m", nxt)
```

### Step 4: Write back

```python
T.copy(C_L0C, C[m * block_M:, cid * block_N:])
```

### Complete example

```python
@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def double_buffered_gemm(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((N, K), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, BLOCK_N), is_npu=True) as (cid, vid):
        A_L1_0 = T.alloc_L1((BLOCK_M, BLOCK_K), "float16")
        A_L1_1 = T.alloc_L1((BLOCK_M, BLOCK_K), "float16")
        B_L1_0 = T.alloc_L1((BLOCK_N, BLOCK_K), "float16")
        B_L1_1 = T.alloc_L1((BLOCK_N, BLOCK_K), "float16")
        C_L0C = T.alloc_L0C((BLOCK_M, BLOCK_N), "float16")

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

                T.wait_flag("mte3", "m", cur)
                T.gemm_v0(A_L1[cur], B_L1[cur], C_L0C, transpose_B=True, init=(k == 0))

                if k + 1 < num_k:
                    T.copy(A[m * BLOCK_M:, (k + 1) * BLOCK_K:], A_L1[nxt])
                    T.copy(B[(k + 1) * BLOCK_K:, cid * BLOCK_N:], B_L1[nxt])
                    T.set_flag("mte3", "m", nxt)

            T.copy(C_L0C, C[m * BLOCK_M:, cid * BLOCK_N:])
```

## Timing diagram (num_stages=2 equivalent)

| Time | MTE Copy | Cube Compute |
|------|----------|-------------|
| t₀ | copy tile 0 (A_L1_0, B_L1_0) | |
| t₁ | copy tile 1 (A_L1_1, B_L1_1) | gemm tile 0 |
| t₂ | copy tile 2 (A_L1_0, B_L1_0) | gemm tile 1 |
| t₃ | | gemm tile 2 |

## What To Verify After Applying

- The kernel compiles without sync-related errors.
- Results match the reference — off-by-one in event IDs or missing `init=(k==0)` can silently corrupt the accumulator.
- The first iteration's `init=True` correctly zeros the accumulator.
- L1 capacity is sufficient for two complete buffer pairs.
- `T.set_flag` uses `"mte3"` → `"m"` direction (MTE → Cube), matching the producer-consumer relationship.

## Related Patterns

- `cv-sync`: basic CV scope separation with manual sync — use this before upgrading to double buffering.
- `explicit-memory`: use `T.alloc_L1`/`T.alloc_L0C` for the double-buffered buffers.
