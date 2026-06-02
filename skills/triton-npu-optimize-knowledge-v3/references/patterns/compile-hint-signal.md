# Compile Hint Signal

## Summary

Use this skill for late-stage Triton NPU refinements when the kernel structure is already plausible but the compiler is missing a fact the code can prove. The relevant hints are:

- `tl.compile_hint(tensor, "dot_pad_only_k")`
- `tl.max_contiguous(index_or_ptr, N)`
- `tl.multiple_of(index_or_ptr, N)`

This is not a profiler-only pattern. `report.txt` helps rule out larger structural bottlenecks and confirms the hot path, but compile hints are selected from source-code facts plus immediate parent-vs-child performance.

------

## Required Data Extraction

If raw simulator profiling data is available, run the current extractor first:

```shell
python D:\workspace\code\BitfunProfilingTool\feature_extraction\extract_profile_bin_data.py <visualize_data.bin> <output_dir>
```

The current extractor writes `report.txt` when TRACE and API instruction blocks are present. Use it first.

If `report.txt` is missing, do not use `extract_ai_problem_features.py`. Fall back to:

- `dataType_2_TRACE.json`, `dataType_4_API_INSTR.json`, and `flows.json`.
- Detail JSON files if present.
- Source diff and parent-vs-child perf text.

The agent using this skill should run extraction itself when given `visualize_data.bin`; it should not require pre-extracted data.

------

## Use When

- A hot `tl.dot` path loads operands with conservative masks, and shape guards prove that only K can require padding. Add `tl.compile_hint(a, "dot_pad_only_k")` and `tl.compile_hint(b, "dot_pad_only_k")` immediately before `tl.dot`.
- Offset tensors are provably contiguous, such as `offs_m`, `offs_n`, `offs_k`, `idx_hw`, or `i = base + tl.arange(...)`, but this fact is hidden behind dynamic starts or masks. Add `tl.max_contiguous`.
- Pointer or offset alignment is guaranteed by layout, block size, branch guard, or shape contract. Add `tl.multiple_of` only when the proof is valid for every active lane.
- `report.txt` shows CUBE/MMA or normal vector/MTE work, but does not show a stronger flat-index decode signature.
- Benchmarking the immediate parent and child shows a win, even if `report.txt` changes only slightly.

## Avoid When

- Source has `offs // HW`, `(offsets // inner) % C`, `pid // G`, or similar scalar coordinate recovery with `DIV`/`REM`/`SIGNEXT` pressure. Use `discrete_memory_access` first.
- The main problem is wrong tiling, wrong grid, manual dot implemented with scalar/vector reductions, redundant kernels, or scattered indexing.
- The contiguity/alignment fact is shape-conditional and no branch or assertion proves the condition.
- The hint is applied to a pointer expression that may include padding, tails, non-contiguous tensors, or alternate dtype paths.
- A hint beats an old baseline but loses to the immediate parent. Keep compile hints only when the local diff wins.

------

## Signal Matching Decision Guide

1. Run `extract_profile_bin_data.py` and open `report.txt` if available.
2. Rule out structural bottlenecks. If `[TRACE Events]` is dominated by `DIV`, `REM`, `SIGNEXT`, and source has flat index recovery, choose `discrete_memory_access`.
3. Inspect the hot source path.
   - If it is `tl.load -> tl.dot` with operand masks, check whether M/N are full and only K tails.
   - If it is contiguous `tl.arange` addressing, check whether the active lanes are contiguous and aligned.
4. Add the smallest valid hint set.
   - For dot padding, hint the loaded operands, not the pointer expression.
   - For contiguity, wrap the offset tensor used by the load/store/dot.
   - For alignment, use `tl.multiple_of` only when all active lanes satisfy it.
5. Benchmark the immediate parent and child. Compile hints often produce small or mixed simulator deltas; the perf text is the deciding evidence.

------

## Hint Recipes

### `dot_pad_only_k`

Detect:

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
acc += tl.dot(a, b)
```

Use when M/N tiles are full or guarded, and only K can be padded:

```python
a = tl.load(a_ptrs, mask=a_mask, other=0.0)
b = tl.load(b_ptrs, mask=b_mask, other=0.0)
tl.compile_hint(a, "dot_pad_only_k")
tl.compile_hint(b, "dot_pad_only_k")
acc += tl.dot(a, b)
```

### `max_contiguous`

Detect:

```python
offs = block_start + tl.arange(0, BLOCK)
x = tl.load(ptr + offs, mask=mask, other=0.0)
```

Use when `offs` is a contiguous lane vector:

```python
offs = tl.max_contiguous(offs, BLOCK)
x = tl.load(ptr + offs, mask=mask, other=0.0)
```

### `multiple_of`

Detect a proven alignment fact:

```python
offs_m = tl.arange(0, BLOCK_M)
offs_n = tl.arange(0, BLOCK_N)
```

Apply only when the value is aligned for every active lane:

```python
offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
```

------

## New Data Examples

All examples were extracted on the new `zrt` container under `/mnt/data01/zrt/features/skill_reextract_20260601` with the current `extract_profile_bin_data.py`.

### `l1_3_Batched_matrix_multiplication`

Change: inserted `tl.compile_hint(a, "dot_pad_only_k")` and `tl.compile_hint(b, "dot_pad_only_k")` immediately before `tl.dot`.

Observed signals:

- Perf: `17621.687 us -> 17482.979 us`; small win, so this is a low-risk keep rather than a major rewrite.
- Both reports show a dot kernel with CUBE/MMA present.
- Origin report had total trace events `907`; opt had `1750`. This mixed simulator change is acceptable only because the immediate benchmark improved.
- No flat memory-access rewrite was attempted; source still has the same dot structure.

### `l1_18_Matmul_with_transposed_both`

Change: added contiguous/alignment hints:

```python
offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
offs_k = tl.max_contiguous(offs_k, BLOCK_K)
```

Observed signals:

- Perf: `1097221.645 us -> 1141.975 us`.
- Total cycles: `24395 -> 8425`.
- Trace events: `1233 -> 509`.
- `WAIT_FLAG`: `33 -> 15`; `BAR`: `5 -> 4`.
- Source remained a `tl.dot(tl.trans(a_km), b_kn)` kernel; the useful change was communicating contiguous/aligned tile facts to lowering.

### `l2_19_ConvTranspose2d_GELU_GroupNorm`

Change: added `tl.max_contiguous(tl.multiple_of(i, 16), 16)` and `tl.max_contiguous(tl.multiple_of(idx_hw, 16), 16)` on groupnorm spans, along with channel unrolling.

Observed signals:

- Perf: `25103.004 us -> 7616.934 us`.
- Total cycles: `5131179 -> 207365`.
- Trace events: `66348 -> 5022`.
- `BAR`: `5524 -> 276`; `WAIT_FLAG`: `736 -> 128`.
- This is a mixed example: hints help because the HW lanes are contiguous, but do not attribute the full gain to hints when unrolling or specialization is also present.

### Fallback Examples Without `report.txt`

Some profiles extracted TRACE/API JSON but did not generate `report.txt`. These can still support this skill when the source and perf are clear:

- `l1_17_Matmul_with_transposed_B`: added two `dot_pad_only_k` hints; perf `1770.448 us -> 613.903 us`.
- `l1_8_Matmul_with_irregular_shapes_`: added two `dot_pad_only_k` hints plus alignment hints; perf `8286.630 us -> 4591.142 us`.

------

## Expected Result

A good compile-hint change is usually narrow:

- Source diff adds hints near the exact load/dot or contiguous offset expression.
- It does not rewrite the algorithm or hide a structural bottleneck.
- `report.txt` may improve, stay mixed, or even show slightly more simulator events.
- The decisive signal is parent-vs-child benchmark improvement on the same harness.
- If the win disappears after a neighboring structural rewrite, remove the hint or re-measure it in isolation.
