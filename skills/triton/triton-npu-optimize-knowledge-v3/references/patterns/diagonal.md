# Diagonal Block Traversal Pattern

## Summary

When tiled matrix-style kernels are already structurally sound but still suffer cache contention, change traversal order rather than math. Diagonal/grouped block traversal reduces simultaneous pressure on the same cache regions and can improve effective L2 reuse.

This pattern is a scheduling-order optimization (program-to-block mapping), not a replacement for first-order tiling/layout fixes.

## Use When

- Large matrix block grids still show locality/conflict issues after basic tiling is reasonable.
- Many concurrent programs touch similar matrix regions under row-major/horizontal ordering.
- Both `M` and `N` span enough blocks that traversal order materially changes reuse behavior.

## Avoid When

- Problem size is too small for traversal order to matter.
- Kernel still lacks fundamental tile/layout correctness.
- Mapping overhead outweighs locality benefit.

## Signals

### Code

- Work assignment is row-major and causes synchronized reuse conflicts.
- Kernel math is stable; only block-order decisions remain unsettled.
- One operand footprint is large enough that naive ordering increases eviction/reload traffic.

### Profile

- Throughput varies significantly with block scheduling order despite identical arithmetic.
- Memory contention or reuse loss persists after tile-size tuning.

## Optimization Strategy

1. Keep kernel math unchanged.
2. Replace naive block order with diagonal/grouped progression.
3. Gate diagonal mode by block-grid thresholds.
4. Tune diagonal partition size (for example 8x8 regions) with benchmarks.
5. Validate against immediate parent and fallback row-major mode.

## Detail

### Diagonal Matmul (concept)

Traditional horizontal task numbering:

```
[0,  1,  2,  3,  4,  5,  6,  7]
[8,  9, 10, 11, 12, 13, 14, 15]
...
```

Example 8x8 diagonal numbering:

```
[0,  8, 16, 24, 32, 40, 48, 56]
[57, 1,  9, 17, 25, 33, 41, 49]
...
```

For large shapes, diagonal progression can:

- lower same-region concurrent access conflicts,
- keep right-hand-side tile reuse healthier before eviction.

### Why diagonal helps

For large 2D block grids, plain row-major progression can create two practical problems:

1. Many cores touch similar left-matrix cache regions at once, increasing contention.
2. Right-matrix tiles for one row band may be consumed, evicted, and reloaded before neighboring rows use them.

Thresholded diagonal partitioning changes concurrent access patterns so reuse windows are less synchronized and can be more cache-friendly. This is why the pattern is usually gated by `num_blocks_m >= threshold` and `num_blocks_n >= threshold` instead of applied unconditionally.

In plain horizontal striping, each core often receives tasks like `pid`, `pid + num_cores`, `pid + 2*num_cores` and so on. On large grids this can make many cores march through similarly aligned regions in lockstep, which increases contention and reuse collapse.

### Full reference implementation (autotuned)

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_TRESHHOLD": 8}),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    mat_a, mat_b, mat_c,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, num_cores: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_blocks_m = triton.cdiv(M, BLOCK_M)
    num_blocks_n = triton.cdiv(N, BLOCK_N)
    num_blocks = num_blocks_m * num_blocks_n

    if num_blocks_m >= BLOCK_TRESHHOLD and num_blocks_n >= BLOCK_TRESHHOLD:
        for block_idx in range(pid, num_blocks, num_cores):
            cur_m = BLOCK_TRESHHOLD if block_idx < (num_blocks_m // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * num_blocks_n else num_blocks_m % BLOCK_TRESHHOLD
            cur_mn = cur_m * BLOCK_TRESHHOLD
            cur_n = BLOCK_TRESHHOLD if block_idx % (num_blocks_n * BLOCK_TRESHHOLD) < (cur_m * num_blocks_n) // cur_mn * cur_mn else num_blocks_n % BLOCK_TRESHHOLD
            local = block_idx % (BLOCK_TRESHHOLD * num_blocks_n) % (BLOCK_TRESHHOLD * cur_m)
            m_idx = local % cur_m + block_idx // (BLOCK_TRESHHOLD * num_blocks_n) * BLOCK_TRESHHOLD

            if cur_m > cur_n:
                x, y = cur_m, cur_n
            else:
                x, y = cur_n, cur_m
            while y != 0:
                x, y = y, x % y
            lcm = cur_m * cur_n // x
            n_idx = (local + (local // lcm)) % cur_n + block_idx % (BLOCK_TRESHHOLD * num_blocks_n) // cur_mn * BLOCK_TRESHHOLD

            m_start = m_idx * BLOCK_M
            n_start = n_idx * BLOCK_N
            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (k_start + tl.arange(0, BLOCK_K))[None, :]
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((k_start + tl.arange(0, BLOCK_K)) < K)[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
                mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (n_start + tl.arange(0, BLOCK_N))[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & ((n_start + tl.arange(0, BLOCK_N)) < N)[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask=mat_c_mask)
    else:
        for block_idx in range(pid, num_blocks, num_cores):
            m_idx = block_idx // num_blocks_n
            n_idx = block_idx % num_blocks_n
            m_start = m_idx * BLOCK_M
            n_start = n_idx * BLOCK_N
            # same matmul tile load / dot / store body as above, with row-major m_idx/n_idx
```

### Simplified code sketch

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  BLOCK_THRESHOLD: tl.constexpr):
    pid = tl.program_id(0)
    num_m = triton.cdiv(M, BLOCK_M)
    num_n = triton.cdiv(N, BLOCK_N)
    num_blocks = num_m * num_n

    for block_idx in range(pid, num_blocks, tl.num_programs(0)):
        # Fallback row-major mapping.
        m_idx = block_idx // num_n
        n_idx = block_idx % num_n

        # Thresholded diagonal remap.
        if num_m >= BLOCK_THRESHOLD and num_n >= BLOCK_THRESHOLD:
            local = block_idx % (BLOCK_THRESHOLD * BLOCK_THRESHOLD)
            m_idx = (block_idx // num_n + local) % num_m
            n_idx = (block_idx // num_m + local * 2) % num_n

        m_start = m_idx * BLOCK_M
        n_start = n_idx * BLOCK_N
        # ... unchanged tile load / tl.dot / store ...
```

## Failure Modes And Anti-signals

- Over-complicated mapping adds scalar overhead without locality gains.
- Group/diagonal width is treated as monotonic and drifts into regressions.
- Traversal changes are mixed with many other edits, hiding true impact.

## Risks

- Incorrect block mapping can create coverage gaps/overlaps.
- Extra mapping logic can regress small and medium shapes.
- Benefit is workload-dependent and can be narrow.

## What To Verify After Applying

- Numerical results remain unchanged.
- Block mapping covers full output domain exactly once.
- Large-shape benchmarks improve over row-major parent.
- Small/medium-shape regressions remain acceptable or are dispatch-guarded.

## Related Patterns

- `tiling`
- `layout-store-and-block-pointers`
- `software-pipeline`
- `compile_hint`
