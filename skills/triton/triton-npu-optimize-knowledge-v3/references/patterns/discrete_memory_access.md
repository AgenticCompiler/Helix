# Discrete Memory Access Staging Pattern

## Summary

Use this pattern when a Triton NPU kernel spends hot-path work on discrete memory access or on rediscovering a discrete dimension from a flat lane offset.

There are two related forms:

- Index-driven global loads, such as `out = x[idx]`: stage contiguous source spans first, then select locally with `tl.gather` or equivalent local indexing.
- Flat-offset coordinate decode, such as `c_idx = (offs // hw) % channels`: retile the program so one axis is the discrete outer dimension and the inner axis is a contiguous span.

Both forms replace per-lane discrete global/index work with contiguous movement plus local selection or broadcast.

------

## Use When

- The central bottleneck is discrete indexed access or flat-offset coordinate recovery rather than arithmetic.
- Hot loops repeatedly execute direct indexed global loads (`x[idx]` style).
- Source reconstructs discrete dimensions from a flat offset, such as `c_idx = (offs // hw) % channels`, `plane_idx = offsets // HW`, `pid // G`, `pid % G`, or `(offsets // inner) % C`.
- `report.txt` shows high scalar pressure from coordinate decode: large `SIGNEXT`, `DIV`, `REM`, `ADD`, `ADD_IMM`, `SUB`, or `MADD` counts.
- `[Source Code Info]` points to a hot source line that is mostly SCALAR and the line loads a per-channel, per-plane, per-group, or per-window value using a recovered index.
- `[Pipeline Flows]`, `WAIT_FLAG`, and `BAR` counts are much larger than the useful work would suggest, especially when `SCALARToVECTOR` or `MTE2ToVECTOR` flow counts are in the thousands.
- One program could own contiguous rows/spans, but current mapping is fully elementwise or crosses discrete boundaries.

## Avoid When

- Source spans are too large to stage efficiently.
- Access is already mostly contiguous and indexing is not the bottleneck.
- The source is a flat contiguous elementwise pass and has no per-lane dimension recovery.
- The hot code is a `tl.dot` path whose main opportunity is padding/contiguity hints; use `compile_hint` instead.
- The primary issue is launch geometry, decomposition, CUBE/MTE tile shape, or simple hinting rather than access shape.
- The discrete axis is genuinely irregular and cannot be made contiguous, staged, or broadcast cheaply.

------

## Signal Matching Decision Guide

Read from `report.txt` and the hot source path:

1. Check `[TRACE Events]`. Strong flat-decode matches have many `DIV`, `REM`, `SIGNEXT`, `ADD`, `ADD_IMM`, `SUB`, or `MADD` events. `DIV` and `REM` are the clearest signal.
2. Check `[Pipe Distribution]` and `[Key Ratios]`. SCALAR cycles above 25% are enough when a hot source line confirms decode; SCALAR cycles above 70% is a very strong match.
3. Check `[Source Code Info]`. If a hot line maps to a per-channel/per-plane/per-group load and is mostly SCALAR, inspect the source for `//`, `%`, or flat offset recovery.
4. Check `[Pipeline Flows]`. Large `SCALARToVECTOR`, `MTE2ToVECTOR`, `WAIT_FLAG`, or `BAR` counts support this pattern when tied to address/index work.
5. Confirm an alternate mapping exists. Prefer grids that never cross the discrete boundary, such as `(ceil(inner/BLOCK), N*C)` or programs over rows with an inner contiguous column loop.
6. Treat metric changes as post-change validation, not as part of the initial match.

------

## Simulation Signature

| Metric | Threshold / signal | report.txt section | Interpretation |
| ------ | ------------------ | ------------------ | -------------- |
| Trace event count | High, often > 10,000 | `[TRACE Events]` | The kernel is doing too much scalar coordinate or index work. |
| Top event names | `DIV`, `REM`, `SIGNEXT`, `ADD`, `SUB`, `MADD` dominate | `[TRACE Events]` | Flat offset decode or address reconstruction is likely on the hot path. |
| SCALAR cycles | > 25% with source confirmation; > 70% is very strong | `[Pipe Distribution]`, `[Key Ratios]` | SCALAR is bottlenecking useful vector/MTE work. |
| Source attribution | Hot line mostly SCALAR | `[Source Code Info]` | If the line loads bias/scale/group data by recovered index, this pattern is a strong match. |
| Flow pressure | Large `SCALARToVECTOR`, `WAIT_FLAG`, or `BAR` | `[Pipeline Flows]`, trace events | Address/index work is creating pipeline bubbles or extra synchronization. |

### Matching Rule

Fire this pattern when report-level scalar/index pressure and source-level discrete access agree. Source confirmation is required: do not choose this pattern from a high SCALAR ratio alone if the hot code is a normal dot, pure tiling issue, or simple compile-hint opportunity.

------

## Signals

### Code

- Hot loops repeatedly execute direct indexed global loads (`x[idx]` style).
- Per-lane index decode (`//`, `%`, address reconstruction) dominates surrounding math.
- One program could own contiguous rows/spans but current mapping is fully elementwise.
- Bias, scale, sum, group, row, or channel values are loaded by a recovered lane index.

### Profile

- High SCALAR pipe ratio or high SCALAR cycles tied to address/index work.
- `[TRACE Events]` dominated by `DIV`, `REM`, `SIGNEXT`, `ADD`, `SUB`, or `MADD`.
- Large `SCALARToVECTOR`, `WAIT_FLAG`, or `BAR` counts tied to the indexed access or coordinate decode.

------

## Optimization Strategy

1. Reframe indexing into contiguous views where possible.
2. Stage contiguous spans from global memory, then select indexed values locally.
3. For flat-offset decode, retile into explicit outer discrete axes plus contiguous inner spans.
4. Load per-channel, per-plane, per-group, or per-window values once per program/block and broadcast locally.
5. Repair launch mapping if widened per-program work creates grid-limit pressure.
6. Validate parent-vs-child and baseline correctness/perf.

------

## Code Manifestations

### Manifestation A: Direct indexed global load

Detect:

```python
idx = tl.load(idx_ptr + rn * stride_idx)
mask = idx < M
val = tl.load(x_ptr + idx * stride_x, mask=mask)
tl.store(out_ptr + rn, val, mask=mask)
```

Transform: stage a contiguous source span first, then select locally.

```python
rm = tl.arange(0, M)
rn = tl.arange(0, N)

idx = tl.load(idx_ptr + rn * stride_idx)
mask = idx < M

x_shared = tl.load(x_ptr + rm * stride_x)
val = tl.gather(x_shared, idx, axis=0)
tl.store(out_ptr + rn, val, mask=mask)
```

### Manifestation B: Channel bias/scale from flat offsets

Detect:

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offs, mask=mask, other=0.0)
c_idx = (offs // HW) % C
scale = tl.load(scale_ptr + c_idx, mask=mask, other=1.0)
```

Transform: split plane/channel from the inner contiguous span.

```python
pid_hw = tl.program_id(0)
pid_plane = tl.program_id(1) * BLOCK_PLANES + tl.arange(0, BLOCK_PLANES)
hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

scale = tl.load(scale_ptr + (pid_plane % C), mask=pid_plane < N * C, other=1.0)
base = pid_plane[:, None] * HW + hw[None, :]
x = tl.load(x_ptr + base, mask=mask, other=0.0)
```

### Manifestation C: Bias/channel lookup from a flattened channel-major span

Detect:

```python
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
c_idx = (offs // hw) % channels
bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
```

Transform: use explicit rows as the discrete channel-major axis and columns as the inner contiguous span.

```python
rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
cols = tl.arange(0, BLOCK_N)

channel_idx = rows % channels
bias = tl.load(bias_ptr + channel_idx, mask=rows < n_rows, other=0.0)[:, None]
offsets = rows[:, None] * n_cols + cols[None, :]
x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
```

### Manifestation D: Boundary-crossing fallback decode

Detect:

```python
group0 = block_start // inner
group1 = (block_start + BLOCK_SIZE - 1) // inner
if group0 == group1:
    value = tl.load(value_ptr + (group0 % C))
else:
    c_idx = (offsets // inner) % C
    value = tl.load(value_ptr + c_idx, mask=mask, other=0.0)
```

Transform: use a grid that cannot cross the inner boundary.

```python
pid_inner = tl.program_id(0)
pid_outer = tl.program_id(1)

inner_offsets = pid_inner * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
base = pid_outer * inner + inner_offsets
value = tl.load(value_ptr + (pid_outer % C))
x = tl.load(x_ptr + base, mask=inner_offsets < inner, other=0.0)
```

### Manifestation E: Group or row decode from `pid`

Detect:

```python
pid = tl.program_id(0)
row = pid // G
group = pid % G
```

Transform: make row and group independent grid axes.

```python
pid_row = tl.program_id(0)
group = tl.program_id(1)
rows = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)
```

------

## Failure Modes And Anti-signals

- Over-staging large ranges hurts occupancy or on-chip footprint.
- Initial contiguous remap can violate launch limits before grid repair.
- Wrong-priority application is likely when the bottleneck is elsewhere.
- A 2D retile can create too many programs for very small inner spans; benchmark small shapes.
- Replacing `//`/`%` with a branchy boundary fix can be slower if the program still crosses many boundaries.

## What To Verify After Applying

- Boundary and index-extreme correctness.
- Launch geometry remains valid for target hardware.
- Parent-vs-child and baseline performance on the same harness.
- Profile and source are checked for remaining scalar decode pressure or expensive scattered loads.
- Per-lane `//` and `%` disappear or move to one scalar per row/plane/group.

------

## Related Patterns

- `gather-load`
- `layout-store-and-block-pointers`
- `scalar-latency-traps`
- `program-multiple-rows`
- `compile_hint`
