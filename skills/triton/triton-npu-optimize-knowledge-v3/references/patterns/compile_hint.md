# Compiler Hint Pattern

## Summary

Use compile hints to communicate layout facts the compiler cannot always infer safely:

- `tl.compile_hint(x, "dot_pad_only_k")`
- `tl.multiple_of(ptr_or_idx, N)`
- `tl.max_contiguous(ptr_or_idx, N)`

This is a late-stage refinement pattern. It works best after layout, launch policy, and main kernel structure are already stable.

## Use When

- The kernel is structurally sound, but lowering still appears conservative.
- You can prove stronger alignment/contiguity facts than current code expresses.
- Dot kernels are stable and only need targeted lowering guidance.
- Parent comparisons are close enough that IR/lowering improvements can matter.

## Avoid When

- Core bottleneck is still structural (wrong tiling, launch shape, decomposition).
- Alignment/contiguity assumptions are shape-conditional but not dispatch-guarded.
- Hints are being used to compensate for invalid pointer/index math.

## Signals

### Code

- Dot inputs where only `K` is a true padding edge.
- Mostly full-tile contiguous slice accesses with conservative masks.
- Pointer/index expressions whose alignment is guaranteed by host contracts.

### Profile

- Strong parent kernel with small remaining inefficiencies.
- Hint-only rounds produce mixed outcomes (some wins, some regressions), indicating sensitivity.

## Optimization Strategy

1. Stabilize structure first.
2. Add the smallest hint set on the proven hot path.
3. Guard shape-dependent assumptions with dispatch.
4. Compare to immediate parent on same harness.
5. Keep hints that win; narrow/revert flat or regressive hints.

## Common Repairs

### `dot_pad_only_k` on true dot inputs

Use when `M` and `N` are already aligned by construction and only `K` needs padding semantics.

### `multiple_of` and `max_contiguous` on proven slices

Apply only where alignment/contiguity are guaranteed for the active branch.

### Narrow hint scope

If a hint helps only one dtype/shape branch, scope it there instead of applying globally.

### Roll back parent regressions

If a hint beats historical baseline but loses immediate parent, treat it as failed and revert/narrow.

## Failure Modes And Anti-signals

- Invalid `multiple_of` assertions on boundary/index-driven paths.
- Hint stacking on fragile fast paths regresses scheduling or occupancy.
- Baseline-only optimism hides parent regression.
- Applying hints before structure fixes creates noisy, non-transferable results.

## Risks

- Overstated assumptions can create subtle correctness issues.
- Hint-heavy code is harder to audit/maintain.
- Benefits can be narrow to specific shape mixes and unstable across regimes.

## What To Verify After Applying

- Asserted alignment/contiguity facts hold for every dispatched regime.
- Boundary/tail correctness where masks and hints interact.
- Parent-vs-child performance on the same benchmark mix.
- Profile/IR evidence reflects intended lowering changes.
- No new regressions in small-shape/index-heavy fallback paths.

## Related Patterns

- `layout-store-and-block-pointers`
- `tiling`
- `program-multiple-rows`
- `scalar-latency-traps`

## Detail

### dot_pad_only_k

Try using `dot_pad_only_k` to express that in a Cube operation only the `k` direction needs padding (`m` and `n` are already aligned). For example:

```python
for k_start in range(0, K, BLOCK_K):
    mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
        k_start + tl.arange(0, BLOCK_K)
    )[None, :]
    mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
        (k_start + tl.arange(0, BLOCK_K)) < K
    )[None, :]
    mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
    tl.compile_hint(mat_a_block, "dot_pad_only_k")
    mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
        n_start + tl.arange(0, BLOCK_N)
    )[None, :]
    mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
        (n_start + tl.arange(0, BLOCK_N)) < N
    )[None, :]
    mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
    tl.compile_hint(mat_b_block, "dot_pad_only_k")
    mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
```

### max_contiguous and multiple_of

Use `tl.max_contiguous` to declare contiguous accesses and `tl.multiple_of` to declare alignment. For example:

```python
@triton.jit
def write_req_to_token_pool_triton_optimize(
    req_to_token_ptr,
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)
    extend_len = seq_len - pre_len

    cumsum_start = 0
    for i in range(pid_batch):
        cumsum_start += tl.load(extend_lens + i)

    token_start = pid_token * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    actual_offset = token_start + offset
    mask = actual_offset < extend_len

    src_ptr = out_cache_loc + cumsum_start + actual_offset
    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)
    value = tl.load(src_ptr, mask=mask)
    dst_ptr = (
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride
        + actual_offset
        + pre_len
    )
    dst_ptr = tl.max_contiguous(tl.multiple_of(dst_ptr, BLOCK_SIZE), BLOCK_SIZE)

    tl.store(dst_ptr, value, mask=mask)
```
