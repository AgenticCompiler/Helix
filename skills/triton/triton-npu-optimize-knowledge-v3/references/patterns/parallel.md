# Dual Vector-Core Parallel Pattern

## Summary

Use `tl.parallel` to run independent vector-side work concurrently across the two vector cores in one AICore. This helps when compute branches are independent and substantial enough to amortize parallel-control overhead.

The core constraint is branch independence.

## Use When

- Two compute-side substeps are independent but currently sequential.
- Candidate work is vector compute (casts, scales, elementwise transforms), not shared-bandwidth loads.
- Branch work is large enough that `tl.parallel` overhead is small relative to useful work.

## Signals

### Code

- Independent operand transforms exist before a shared consumer (for example `tl.dot`).
- Natural split across inputs or independent epilogue branches.

### Performance

- Vector compute remains material after layout/tiling fixes.
- Serial vector-side phases still dominate portions of loop runtime.

## Avoid When

- Branches have real data dependencies.
- Dominant bottleneck is memory bandwidth.
- Candidate work is too small/fine-grained.

## Optimization Strategy

1. Identify branch-independent compute in the hot loop.
2. Parallelize compute-side branches with `tl.parallel`.
3. Keep loads out of the parallel branches unless proven beneficial.
4. Join at dependent consumer and compare against immediate parent.

## Detail

### Good split example (independent A/B scaling)

```python
for core_id in tl.parallel(0, 2, bind_sub_block=True):
    if core_id == 0:
        scaled_a = a_s * a.to(tl.float16)
    else:
        scaled_b = b_s * b.to(tl.float16)

acc = tl.dot(scaled_a, scaled_b, acc=acc)
```

### Quantized matmul-style reference

```python
for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
    k_remaining = K - k * BLOCK_SIZE_K
    a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
    b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    k_group_idx = (k * BLOCK_SIZE_K) // group_k
    a_scale_indices = k_group_idx + k_group_offset[None, :]
    b_scale_indices = k_group_idx + k_group_offset[:, None]
    a_s = tl.load(As + offs_am[:, None] * stride_As_m + a_scale_indices * stride_As_k, mask=a_mask, other=1.0)
    b_s = tl.load(Bs + b_scale_indices * stride_Bs_k + offs_bsn[None, :] * stride_Bs_n, mask=b_mask, other=1.0)

    scaled_a = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    scaled_b = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float32)
    for core_id in tl.parallel(0, 2, bind_sub_block=True):
        if core_id == 0:
            scaled_a = a_s * a.to(tl.float16)  # vector core 0
        else:
            scaled_b = b_s * b.to(tl.float16)  # vector core 1

    acc = tl.dot(scaled_a, scaled_b, acc=acc, allow_tf32=False)
    a_ptrs += BLOCK_SIZE_K * stride_ak
    b_ptrs += BLOCK_SIZE_K * stride_bk
```

### Anti-pattern: parallelizing loads

```python
for core_id in tl.parallel(0, 2):
    if core_id == 0:
        a = tl.load(a_ptrs)  # usually shared-bandwidth limited
```

## What To Verify After Applying

- Branches are truly independent across all active regimes.
- Correctness matches sequential version.
- End-to-end benchmark improves vs immediate parent.
- Profile shows reduced serialized vector compute (not unchanged memory stalls).

### Why this split usually works

- A and B scaling paths are independent (no cross-branch dependency until `tl.dot` consume point).
- Each branch operates on contiguous vector-friendly tiles.
- Work split is compute-heavy enough to amortize `tl.parallel` control overhead.

## Related Patterns

- `program-multiple-rows`
- `tiling`
- `software-pipeline`
- `cache_use`
