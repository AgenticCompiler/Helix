# Compiler Hint Pattern

## Summary

Use compile hints as a late-stage Triton NPU refinement when the kernel structure is already sound, but lowering is conservative because the code has not expressed a fact that is true for the active branch.

Relevant hints:

- `tl.compile_hint(x, "dot_pad_only_k")`
- `tl.multiple_of(ptr_or_idx, N)`
- `tl.max_contiguous(ptr_or_idx, N)`

This is not a profiler-only pattern. `report.txt` helps rule out stronger structural bottlenecks and confirms the hot path, but compile hints are selected from source facts plus immediate parent-vs-child performance.

------

## Use When

- The kernel is structurally sound, but lowering still appears conservative.
- A hot `tl.dot` path loads operands with conservative masks, and shape guards prove that only K can require padding.
- Offset tensors such as `offs_m`, `offs_n`, `offs_k`, `idx_hw`, or `base + tl.arange(...)` are provably contiguous, but this fact is hidden behind dynamic starts, masks, or pointer expressions.
- Pointer or offset alignment is guaranteed by layout, block size, branch guard, or shape contract.
- `report.txt` shows normal CUBE/MMA, VECTOR, or MTE work without a stronger flat-index decode signature.
- Parent comparisons are close enough that a small IR/lowering improvement can matter.

## Avoid When

- Core bottleneck is still structural: wrong tiling, wrong grid, wrong decomposition, redundant kernels, manual dot, or scattered indexing.
- Source has `offs // HW`, `(offsets // inner) % C`, `pid // G`, or similar scalar coordinate recovery with `DIV`/`REM`/`SIGNEXT` pressure. Use `discrete_memory_access` first.
- Alignment/contiguity assumptions are shape-conditional but not dispatch-guarded.
- Hints are being used to compensate for invalid pointer/index math.
- The hint may touch padding, tails, non-contiguous tensors, or alternate dtype paths that do not satisfy the asserted fact.
- A hint beats an old baseline but loses to the immediate parent. Treat this as a failed round and revert or narrow it.

------

## Signal Matching Decision Guide

Read from `report.txt` and the hot source path:

1. Rule out structural signals first. If `[TRACE Events]` is dominated by `DIV`, `REM`, `SIGNEXT`, and source has flat index recovery, choose `discrete_memory_access`.
2. Check whether the hot path is already a valid `tl.dot` kernel. If operands are loaded with masks and shape guards prove M/N are full or already guarded while only K is padded, choose `dot_pad_only_k`.
3. Check contiguous lane offsets. If the load/store/dot uses `base + tl.arange(...)`, `offs_m`, `offs_n`, `offs_k`, `idx_hw`, or similar contiguous vectors, and `report.txt` does not show a stronger structural bottleneck, choose `max_contiguous`.
4. Check alignment facts. If layout, block size, or a dispatch branch proves the active lanes are aligned, combine `multiple_of` with `max_contiguous`.
5. Benchmark the immediate parent and child on the same harness. Compile hints often produce small or mixed simulator deltas; the perf text is the deciding evidence.

------

## Simulation Signature

| Metric | Signal | report.txt section | Interpretation |
| ------ | ------ | ------------------ | -------------- |
| CUBE/MMA activity | Present on the hot dot path | `[Pipe Distribution]`, `[Source Code Info]` | The kernel structure is already a dot/lowering problem rather than a rewrite problem. |
| Flat decode events | Not dominant | `[TRACE Events]` | If `DIV`/`REM`/`SIGNEXT` dominate with source `//` or `%`, choose `discrete_memory_access` instead. |
| Trace/event delta | May be mixed | `[TRACE Events]`, `[Pipeline Flows]` | Hint-only rounds can increase simulator events; keep only when immediate perf improves. |
| Parent-vs-child perf | Child wins on same harness | perf text / compare result | This is the decisive signal for compile hints. |

### Matching Rule

Fire this pattern when the hot source path has a provable dot-padding, contiguity, or alignment fact, no stronger structural pattern is present, and the immediate child wins after adding the smallest valid hint set.

------

## Signals

### Code

- Dot inputs where only `K` is a true padding edge.
- Mostly full-tile contiguous slice accesses with conservative masks.
- Pointer/index expressions whose alignment is guaranteed by host contracts.
- Contiguous `tl.arange` addressing where the active lanes are valid but lowering cannot infer the lane order or alignment.

### Profile

- Strong parent kernel with small remaining inefficiencies.
- CUBE/MMA or normal VECTOR/MTE activity is present; the profile does not point to coordinate-decode as the main bottleneck.
- Hint-only rounds produce mixed outcomes, indicating sensitivity.
- `report.txt` may improve slightly, stay mixed, or even show more trace events; keep the change only when immediate parent-vs-child perf improves.

------

## Optimization Strategy

1. Stabilize structure first.
2. Add the smallest hint set on the proven hot path.
3. Guard shape-dependent assumptions with dispatch.
4. Compare to the immediate parent on the same harness.
5. Keep hints that win; narrow or revert flat/regressive hints.

------

## Common Repairs

### `dot_pad_only_k` on true dot inputs

Use when `M` and `N` are already aligned by construction or safely guarded, and only `K` needs padding semantics.

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
tl.compile_hint(a, "dot_pad_only_k")
tl.compile_hint(b, "dot_pad_only_k")
acc += tl.dot(a, b)
```

Hint the loaded tensors immediately before `tl.dot`, not the pointer expression.

### `multiple_of` and `max_contiguous` on proven slices

Apply only where alignment and contiguity are guaranteed for every active lane in the current branch.

```python
offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
offs_k = tl.max_contiguous(offs_k, BLOCK_K)
```

### Narrow hint scope

If a hint helps only one dtype, shape, or dispatch branch, scope it there instead of applying globally.

### Roll back parent regressions

If a hint beats a historical baseline but loses the immediate parent, treat it as failed and revert or narrow it.

------

## Detail

### Original `dot_pad_only_k` example

Use `dot_pad_only_k` to express that in a Cube operation only the `k` direction needs padding (`m` and `n` are already aligned or safely guarded):

```python
for k_start in range(0, K, BLOCK_K):
    mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
        k_start + tl.arange(0, BLOCK_K)
    )[None, :]
    mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
        (k_start + tl.arange(0, BLOCK_K)) < K
    )[None, :]
    mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
    tl.compile_hint(mat_a_block, "dot_pad_only_k")

    mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
        n_start + tl.arange(0, BLOCK_N)
    )[None, :]
    mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
        (n_start + tl.arange(0, BLOCK_N)) < N
    )[None, :]
    mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
    tl.compile_hint(mat_b_block, "dot_pad_only_k")

    mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
```

### Original `max_contiguous` and `multiple_of` example

Use `tl.max_contiguous` to declare contiguous accesses and `tl.multiple_of` to declare alignment where the active branch proves those facts:

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

------

## Failure Modes And Anti-signals

- Invalid `multiple_of` assertions on boundary/index-driven paths.
- Hint stacking on fragile fast paths regresses scheduling or occupancy.
- Baseline-only optimism hides parent regression.
- Applying hints before structure fixes creates noisy, non-transferable results.
- Adding hints to a path whose real issue is flat-index decode or scattered global loads.

## Risks

- Overstated assumptions can create subtle correctness issues.
- Hint-heavy code is harder to audit and maintain.
- Benefits can be narrow to specific shape mixes and unstable across regimes.

## What To Verify After Applying

- Asserted alignment/contiguity facts hold for every dispatched regime.
- Boundary/tail correctness where masks and hints interact.
- Parent-vs-child performance on the same benchmark mix.
- Profile/IR evidence reflects intended lowering changes.
- No new regressions in small-shape or index-heavy fallback paths.

------

## Worked Examples

### `l1_3_Batched_matrix_multiplication`

Code signal: masked operands feed a stable `tl.dot`, and the useful hint is only about dot padding semantics.

Repair:

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
tl.compile_hint(a, "dot_pad_only_k")
tl.compile_hint(b, "dot_pad_only_k")
acc += tl.dot(a, b)
```

Observed result: `17621.687 us -> 17482.979 us`. This is a small win, so treat it as a low-risk keep rather than a major rewrite.

### `l1_18_Matmul_with_transposed_both`

Code signal: source remains a `tl.dot(tl.trans(a_km), b_kn)` kernel; the useful change is communicating contiguous/aligned tile facts to lowering.

Repair:

```python
offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
offs_k = tl.max_contiguous(offs_k, BLOCK_K)
```

Observed result:

- Perf: `1097221.645 us -> 1141.975 us`
- Total cycles: `24395 -> 8425`
- Trace events: `1233 -> 509`
- `WAIT_FLAG`: `33 -> 15`
- `BAR`: `5 -> 4`

### `l2_19_ConvTranspose2d_GELU_GroupNorm`

Code signal: groupnorm spans use contiguous HW lanes, but lowering benefits from explicit contiguity/alignment facts.

Repair:

```python
i = tl.max_contiguous(tl.multiple_of(i, 16), 16)
idx_hw = tl.max_contiguous(tl.multiple_of(idx_hw, 16), 16)
```

Observed result:

- Perf: `25103.004 us -> 7616.934 us`
- Total cycles: `5131179 -> 207365`
- Trace events: `66348 -> 5022`
- `BAR`: `5524 -> 276`
- `WAIT_FLAG`: `736 -> 128`

This is a mixed example. Hints are valid because the HW lanes are contiguous, but do not attribute the full gain to hints when unrolling or specialization is also present.

------

## Related Patterns

- `layout-store-and-block-pointers`
- `tiling`
- `program-multiple-rows`
- `scalar-latency-traps`
- `discrete_memory_access`
