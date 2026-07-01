# Padded Last-Dim Row–Column Tiling

## Summary

Optimize constant-pad and similar bounded copy kernels by replacing a flat 1D traversal over `numel(out)` with an `out_rows x out_dim_last` structure:

- grid over logical rows,
- tile over last-dimension columns,
- hoist row-invariant input base work out of the column loop.

Use `BLOCK_ROWS > 1` for small-last-dim regimes, optional `NO_COL_PAD` fast paths when the last axis has no pad, and host-side `BLOCK_COLS` refinement.
Adopt only with correctness + benchmark gates on representative pad shapes.

## Use When

- The operator is constant pad or another regular per-axis bounds copy (not gather/scatter).
- Baseline kernel heavily uses `//` and `%` per element to recover coordinates.
- Profiling shows scalar/mask overhead concentrated on last-dim boundary handling.

## Signals

### Code

- Single linear `offsets` path with repeated high-stride coordinate decode.
- One global validity mask combines all dimensions for each lane.
- Multi-phase column loops with different store masks for left/interior/right regions.

### Profile

- Scalar/control cost out of proportion to copy bandwidth.
- Hot path dominated by masked-load/compare chains for pad bounds.

## Avoid When

- The hot path is index-driven gather/scatter (use `gather-load` / `discrete_memory_access`).
- Dynamic tile-kind branching inside the column loop cannot be validated on backend lowering.
- Tail stores omit `col_mask` without proof for `out_dim_last % BLOCK_COLS != 0`.

## Optimization Strategy

1. Define `out_rows` (all dims except last) and `out_dim_last` (last axis).
2. Use row-block grid `ceil_div(out_rows, BLOCK_ROWS)` with row masks.
3. Iterate columns with `ceil_div(out_dim_last, BLOCK_COLS)` and always keep `col_mask`.
4. Hoist row-invariant input base pointer before column loop.
5. Add `NO_COL_PAD` constexpr path when `pad_left_last == 0` and `in_dim_last == out_dim_last`.
6. Tune `BLOCK_COLS` on host with bounded refinement; avoid tail-unsafe shortcuts.
7. Use `NATIVE_MASKED_LOAD` split only with backend-validated dtype/shape regimes.
8. Prefer `tl.static_range(BLOCK_ROWS)` style row-lane unrolling where beneficial, while keeping row-stride mapping tied to `num_programs * BLOCK_ROWS`.

### Regime-specific guidance

- `BLOCK_ROWS`: start from `4` or `8` when `out_dim_last` is small and `out_rows` is large.
- `NATIVE_MASKED_LOAD`: behavior is backend-dependent; one useful pattern is:
  - narrower last-dim regime: masked load with `other=fill_value`,
  - wider last-dim regime: masked load with `other=0` followed by explicit `tl.where`.
- Keep the last-axis column loop uniform and retain `col_mask` on every tail store unless you have explicit proof for a safe no-tail-mask branch.

### Host refinement note

When refining `BLOCK_COLS`, a practical bounded policy is to halve toward a floor (often around 16) while capping additional column-iteration growth (for example avoid >4x loop inflation). This keeps refinement from overfitting tiny tails at large overhead cost.

## Reference Structure

```python
rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
row_mask = rows < out_rows
in_row_base = compute_in_row_base(rows)  # hoisted

for c in range(0, tl.cdiv(out_dim_last, BLOCK_COLS)):
    cols = c * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    col_mask = cols < out_dim_last
    store_mask = row_mask[:, None] & col_mask[None, :]
    # ... masked load / fill / store ...
```

## What To Verify After Applying

- Correctness for asymmetric pads and last-dim tails (`out_dim_last % BLOCK_COLS != 0`).
- Benchmark gains vs baseline and framework reference under same harness.
- Profile confirms reduced scalar/mask overhead before widening thresholds further.
- Zero-output edge behavior is explicitly handled.

## Related Patterns

- `tiling`
- `program-multiple-rows`
- `loop-invariant-hoisting`
- `autotune`
- `compile_hint`
- `vec-cmp`
