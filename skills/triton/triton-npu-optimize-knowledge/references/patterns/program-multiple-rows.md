# Program Multiple Rows Pattern

## Summary

Map multiple logical rows to one Triton program (`BLOCK_M > 1`) to amortize per-program overhead and improve vector utilization in row-structured kernels.

This is a program-granularity pattern: fewer, heavier programs doing more row work per launch.

## Use When

- Kernel is naturally row-wise (row reductions, row-wise fused epilogues, row-major transforms).
- Current launch maps one row per program and profiling shows many thin programs or scalar-heavy overhead.
- Inner-dimension streaming over `N` can remain single-pass while widening row count.
- Row count is large enough to amortize wider per-program bundles.

## Avoid When

- Row count is tiny and wider bundles cannot amortize setup.
- Increasing `BLOCK_M` introduces second full passes or unstable numeric behavior.
- Main bottleneck is elsewhere (layout/store shape, algorithm structure, unrelated scalar traps).
- Ping-pong/multibuffer variants are introduced without clear MTE-vector overlap evidence.

## Signals

### Code

- `program_id(0)` maps directly to one row.
- Repeated per-row pointer/control setup dominates loop body.
- Inner-dimension tiling exists (`BLOCK_N`), but row axis remains under-batched.

### Profile

- Scalar/control pressure stays high with one-row programs.
- Moderate row batching gives clear gains, but over-widening regresses.
- Useful cues include `aiv_scalar_ratio`, `aiv_mte2_ratio`, and `op_statistic` Avg/Count deltas; treat `BAR` cycles as diagnostic context, not a success metric by itself.
- Barrier/wait growth with many short programs is a common indicator that row granularity is too fine.

Profiler interpretation notes:

- `op_statistic` Avg should be compared on matched shapes/workload; Count changes can otherwise hide regressions.
- If `aiv_mte2_ratio` dominates while scalar ratio is low, row batching may be secondary to transfer/layout levers.
- If scalar ratio remains high after moderate `BLOCK_M` increases, combine with scalar-control cleanups rather than widening blindly.

## Optimization Strategy

1. **Prefer the 2D vectorized BLOCK_M variant.** Unlike the looped BLOCK_ROWS approach (which processes rows one-by-one in a Python-level for-loop), the 2D BLOCK_M variant uses `offs_m[:, None]` + `offs_n[None, :]` broadcasting to process multiple rows simultaneously with coalesced memory access. This is the strongly preferred form for row-structured elementwise/fused-epilogue kernels.
2. Introduce `BLOCK_M > 1` as `tl.constexpr` and remap row ownership to row blocks using `offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`.
3. Use 2D addressing: `row_offsets = offs_m[:, None] * row_stride` and `col_offsets = offs_n[None, :]` to create a full `[BLOCK_M, BLOCK_N]` tile.
4. Keep one-pass inner-dimension streaming when possible.
5. Tune `BLOCK_M` progressively with parent-vs-parent checks.
6. Add shape/dtype gates when one global `BLOCK_M` regresses some regimes.
7. Compose with inner-tile and launch-parameter tuning only after each row-batching step is validated.

### Tiered BLOCK_M example

```python
if total_rows >= 131072:
    BLOCK_M = 64
elif total_rows >= 32768:
    BLOCK_M = 32
else:
    BLOCK_M = 8
grid = (triton.cdiv(total_rows, BLOCK_M),)
```


## Implementation sketches (Triton)

### Variant A: 2D BLOCK_M (PREFERRED) — coalesced multi-row access with broadcasting

Use this when the kernel operates on a 2D [rows, cols] view of the data. The 2D broadcast pattern enables coalesced loads across both dimensions. This is the variant the structural optimization priority gate requires evaluating first.

```python
@triton.jit
def _row_structured_kernel(
    x_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    cols,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < rows
    mask_n = offs_n < cols
    mask = mask_m[:, None] & mask_n[None, :]

    row_offsets = offs_m[:, None] * in_row_stride
    vals = tl.load(x_ptr + row_offsets + offs_n[None, :], mask=mask, other=0.0)

    # row-wise compute on vals [BLOCK_M, BLOCK_N]...

    out_row_offsets = offs_m[:, None] * out_row_stride
    tl.store(out_ptr + out_row_offsets + offs_n[None, :], result, mask=mask)
```

Host launch:
```python
grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
_row_structured_kernel[grid](x_2d, out_2d, ...)
```

### Concrete example: row-wise fused operation with two independent input windows

```python
@triton.jit
def _swiglu_kernel(
    x_ptr,
    out_ptr,
    in_row_stride,
    out_row_stride,
    half_cols,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask_m = offs_m < rows
    mask_n = offs_n < half_cols
    mask = mask_m[:, None] & mask_n[None, :]

    row_offsets = offs_m[:, None] * in_row_stride
    gate_raw = tl.load(x_ptr + row_offsets + offs_n[None, :], mask=mask, other=0.0)
    value_raw = tl.load(x_ptr + row_offsets + half_cols + offs_n[None, :], mask=mask, other=0.0)

    gate = gate_raw.to(tl.float32)
    value = value_raw.to(tl.float32)
    out = gate * tl.sigmoid(gate) * value

    out_row_offsets = offs_m[:, None] * out_row_stride
    tl.store(out_ptr + out_row_offsets + offs_n[None, :], out.to(gate_raw.dtype), mask=mask)
```

### Variant B: Looped BLOCK_ROWS (FALLBACK) — sequential row processing

Use this only when the 2D BLOCK_M variant is not applicable (e.g., when rows have different lengths, or the per-row computation requires per-row loop-carried state that prevents merging into a 2D tile). This variant processes rows one-by-one in a for-loop, which incurs per-row scalar overhead.

```python
pid_row = tl.program_id(axis=0)
col_offsets = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = col_offsets < out_cols

for r_offset in range(BLOCK_ROWS):
    pid_row = row_pid * BLOCK_ROWS + r_offset
    if pid_row < rows:
        row_start = pid_row * in_row_stride
        gate_raw = tl.load(gate_ptr + row_start + col_offsets, mask=mask, other=0.0)
        value_raw = tl.load(value_ptr + row_start + half_cols_or_zero + col_offsets, mask=mask, other=0.0)
        # ... per-row compute ...
```

### Why Variant A outperforms Variant B

- **Coalesced memory access:** Variant A loads a full `[BLOCK_M, BLOCK_SIZE]` tile in one operation; Variant B makes `BLOCK_ROWS` separate 1D loads.
- **No Python-level loop:** Variant A avoids the per-row for-loop, eliminating per-row scalar dispatch overhead.
- **Better vector utilization:** The 2D broadcast pattern feeds the VECTOR unit wider tiles, reducing SCALAR:VECTOR instruction ratio.
- **Grid dimension trade-off:** Variant A uses 2D grid `(cdiv(rows, BLOCK_M), cdiv(cols, BLOCK_SIZE))`, which enables better core utilization than Variant B's `(cdiv(rows, BLOCK_ROWS), cdiv(cols, BLOCK_SIZE))`.

## Failure Modes And Anti-signals

- Assuming larger `BLOCK_M` is monotonic; it often is not.
- Applying one wide-row setting globally and regressing small/short regimes.
- Introducing a second full pass while widening rows.
- Treating PMR as universal even when another primary lever dominates.

## Risks

- Wider row bundles increase temporary footprint and scheduling pressure.
- Gated dispatch adds maintenance complexity.
- Multi-lever edits can hide regressions without strict parent comparisons.

## What To Verify After Applying

1. Correctness on boundary rows and tail masks.
2. Parent-vs-child benchmark improvement in each representative regime (prefer the project compare-perf authority when available).
3. Fewer launches/programs for intended shapes.
4. No unintended extra global passes.
5. Dispatch gates route correctly for tiny vs large regimes.

## Related Patterns

- `grid-flatten-and-ub-buffering`
- `tiling`
- `parallel`
- `layout-store-and-block-pointers`
- `software-pipeline` (overlap tuning after row granularity is already chosen)
