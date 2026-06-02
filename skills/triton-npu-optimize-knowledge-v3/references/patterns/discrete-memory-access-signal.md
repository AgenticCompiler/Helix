# Discrete Memory Access Signal

## Summary

Use this skill when a Triton NPU kernel spends too much work rediscovering multidimensional coordinates from a flat lane offset, then uses those coordinates to pick per-channel, per-plane, per-group, or per-window data. The usual fix is to retile the program so one axis is the discrete outer dimension and the inner axis is a contiguous span.

Do not make every lane compute `offs // HW`, `offs % C`, `pid // G`, or boundary-crossing channel recovery. Give the program explicit `row`, `plane`, `group`, `channel`, or `inner` axes, load the small discrete value once, then broadcast it across a contiguous tile.

------

## Required Data Extraction

If the input is raw simulator profiling data, run the current extractor first:

```shell
python D:\workspace\code\BitfunProfilingTool\feature_extraction\extract_profile_bin_data.py <visualize_data.bin> <output_dir>
```

The current extractor writes `report.txt` when TRACE and API instruction blocks are present. Use `report.txt` as the primary signal source.

If `report.txt` is missing, fall back to:

- `dataType_2_TRACE.json` for event names such as `DIV`, `REM`, `SIGNEXT`, `ADD`, `SUB`, `MADD`.
- `dataType_4_API_INSTR.json` for pipe instruction distribution.
- `flows.json` for pipeline flow counts.
- `dataType_5_DETAILS_BASE_INFO.json`, `dataType_7_DETAILS_COMPUTE_LOAD_TABLE.json`, and `dataType_9_DETAILS_MEMORY_TABLE.json` if present.
- The origin/opt perf text and source diff.

The agent using this skill should run extraction itself when given `visualize_data.bin`; it should not require pre-extracted data.

------

## Use When

- Source code reconstructs discrete dimensions from a flat offset, such as `c_idx = (offs // hw) % channels`, `plane_idx = offsets // HW`, `pid // G`, `pid % G`, or a branch that falls back to `(offsets // inner) % C`.
- `report.txt` shows high scalar pressure from coordinate decode: large `SIGNEXT`, `DIV`, `REM`, `ADD`, `ADD_IMM`, `SUB`, or `MADD` counts, often with arithmetic events above 10% of complete trace events.
- `[Source Code Info]` points to a hot source line that is mostly SCALAR and the line loads a per-channel/per-plane value using a recovered index.
- `[Pipeline Flows]`, `WAIT_FLAG`, and `BAR` counts are much larger than the useful work would suggest, especially when `SCALARToVECTOR` or `MTE2ToVECTOR` flow counts are in the thousands.
- The operation can be expressed as an outer discrete axis plus an inner contiguous span: `(N*C, HW)`, `(N*C, D*H*W)`, `(rows, group)`, `(plane, hw)`, or `(inner, channel-plane)`.

## Avoid When

- The source is already a flat contiguous elementwise pass and has no per-lane dimension recovery.
- The hot code is a `tl.dot` path whose main opportunity is padding/contiguity hints; use `compile_hint` instead.
- The profiler shows CUBE/MTE/tile size issues without scalar coordinate-decode evidence.
- The discrete axis is genuinely irregular and cannot be made contiguous or broadcast cheaply.
- The only change needed is `tl.max_contiguous` or `tl.multiple_of`; that is a compile-hint refinement, not this structural rewrite.

------

## Signal Matching Decision Guide

1. Run `extract_profile_bin_data.py` and open `report.txt`.
2. Check `[TRACE Events]`. Strong matches have many `SIGNEXT`, `DIV`, `REM`, `ADD`, `ADD_IMM`, `SUB`, or `MADD` events. `DIV`/`REM` are the clearest flat-index decode signal.
3. Check `[Pipe Distribution]` and `[Key Ratios]`. SCALAR cycles above 25% are enough when a hot source line confirms decode; SCALAR cycles above 70% is a very strong match.
4. Check `[Source Code Info]`. If a hot line maps to a per-channel/per-plane load or bias load and is mostly SCALAR, inspect the source for `//`, `%`, or flat offset recovery.
5. Confirm an alternate tiling exists. Prefer grids that never cross the discrete boundary, such as `(ceil(inner/BLOCK), N*C)` or programs over rows with an inner contiguous column loop.
6. After rewriting, expect trace events, `DIV`/`REM`, arithmetic events, `SCALARToVECTOR` flows, `WAIT_FLAG`, and `BAR` to drop. VECTOR or MTE cycles may become a larger percentage because scalar decode has stopped dominating.

------

## Optimization Recipe

### Pattern A: Channel Bias/Scale From Flat Offsets

Detect:

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
c_idx = (offs // HW) % C
scale = tl.load(scale_ptr + c_idx, mask=mask, other=1.0)
```

Rewrite:

```python
pid_hw = tl.program_id(0)
pid_plane = tl.program_id(1) * BLOCK_PLANES + tl.arange(0, BLOCK_PLANES)
hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
scale = tl.load(scale_ptr + (pid_plane % C), mask=pid_plane < N * C, other=1.0)
base = pid_plane[:, None] * HW + hw[None, :]
x = tl.load(x_ptr + base, mask=mask, other=0.0)
```

### Pattern B: Bias/Channel Add After Convolution

Detect:

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
c_idx = (offs // hw) % channels
bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
```

Rewrite:

```python
rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)   # rows are N*C
cols = tl.arange(0, BLOCK_N)                   # cols are HW
channel_idx = rows % channels
bias = tl.load(bias_ptr + channel_idx, mask=rows < n_rows, other=0.0)[:, None]
offsets = rows[:, None] * n_cols + cols[None, :]
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### Pattern C: Boundary-Crossing Fast Path

Detect:

```python
group0 = block_start // inner
group1 = (block_start + BLOCK_SIZE - 1) // inner
if group0 == group1:
    bias = tl.load(sum_ptr + (group0 % C))
else:
    c_idx = (offsets // inner) % C
    bias = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)
```

Rewrite:

```python
pid_inner = tl.program_id(0)
pid_nc = tl.program_id(1)
inner_offsets = pid_inner * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
base = pid_nc * inner + inner_offsets
bias = tl.load(sum_ptr + (pid_nc % C))
x = tl.load(x_ptr + base, mask=inner_offsets < inner, other=0.0)
```

### Pattern D: Group Or Row Decode From `pid`

Detect:

```python
pid = tl.program_id(0)
row = pid // G
group = pid % G
```

Rewrite:

```python
pid_row = tl.program_id(0)
group = tl.program_id(1)
rows = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
```

------

## New Data Examples

All examples were extracted on the new `zrt` container under `/mnt/data01/zrt/features/skill_reextract_20260601` with the current `extract_profile_bin_data.py`.

### `l2_31_Conv2d_Min_Add_Multiply`

Origin code used `c_idx = (offs // hw) % channels` to load bias. The optimized code retiled into `rows = N*C` and `cols = HW`, loads one `bias[:, None]`, and loops over contiguous columns.

Observed signals:

- Perf: `76380.982 us -> 2150.098 us`.
- Total cycles: `8834883 -> 146994`.
- SCALAR cycles: `71.4% -> 19.7%`.
- Trace events: `727648 -> 10811`.
- Arithmetic events: `148480 -> 824`; origin had `DIV=33792` and `REM=33792`, opt had `DIV=2`.
- `SCALARToVECTOR` flows: `5120 -> 67`; `MTE2ToVECTOR` flows: `1024 -> 32`.
- Hot source line was the bias load and was `SCALAR: 100.00%` in origin.

### `l2_54_Conv2d_Multiply_LeakyReLU_GELU`

Origin code computed `plane_idx = offsets // HW` and `c_idx = plane_idx % C` for every lane. The optimized code uses separate plane and HW axes, loads scale per plane, and processes contiguous HW chunks with `tl.max_contiguous(tl.multiple_of(hw_offsets, BLOCK_HW), BLOCK_HW)`.

Observed signals:

- Perf: `78482.103 us -> 3239.817 us`.
- Total cycles: `17735022 -> 510744`.
- SCALAR cycles: `38.8% -> 6.2%`.
- Trace events: `1022112 -> 19506`.
- Arithmetic events: `148480 -> 816`; origin had `DIV=33792`, opt had `DIV=66`.
- `SCALARToVECTOR` flows: `5120 -> 67`; `WAIT_FLAG`: `11264 -> 210`; `BAR`: `45056 -> 1182`.
- Origin hot scale-load line was `SCALAR: 100.00%`.

### `l2_90_Conv3d_LeakyReLU_Sum_Clamp_GELU`

Origin had a fast path for blocks contained in one `inner` span, but fell back to per-lane `c_idx = (offsets // inner) % C` when a block crossed the boundary. The optimized code uses a 2D grid `(inner block, N*C)` so a program never crosses the boundary.

Observed signals:

- Perf: `22.836 us -> 11.591 us`.
- Origin `report.txt` was not generated, so the source diff and extracted TRACE/API JSON were used.
- Code markers changed from `//` count `3 -> 0` and `%` count `2 -> 1`.
- Optimized report still has many events because the simulator case is large, but arithmetic percentage is low (`3.8%`) and no per-lane channel recovery remains.

### `l2_88_Gemm_GroupNorm_Swish_Multiply_Swish`

Origin decoded `row = pid // G` and `group = pid % G`. The optimized code uses `(row block, group)` grid axes and vectorizes multiple rows per program. This is the same structural signal even when the origin perf file contains `NA`; the source and origin report show the flat-`pid` decode.

------

## Expected Result

A successful rewrite usually does not just lower one metric. It changes the shape of the kernel:

- Per-lane `//` and `%` disappear or move to one scalar per row/plane/group.
- `DIV` and `REM` trace events collapse.
- Source hotspots move from scalar index/bias load lines to vector math or MTE movement.
- `SCALARToVECTOR`, `WAIT_FLAG`, and `BAR` counts drop sharply.
- Runtime improves even if VECTOR or MTE cycles become a higher percentage of the remaining work.
