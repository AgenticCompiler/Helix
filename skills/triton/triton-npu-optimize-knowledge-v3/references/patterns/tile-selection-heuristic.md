# Tile Selection Heuristic (Grid Minimization)

## Summary

Replace or augment `@triton.autotune` with a host-side tile-selection heuristic (`_choose_tiling`) that sweeps candidate `BLOCK_M × BLOCK_SIZE` tile sizes and chooses the pair that minimizes total grid programs (`gm × gn`). This is most effective when the operator spans wide shape ranges where a fixed autotune config set cannot adapt well.

## Use When

- Autotune with a limited config set (≤10 configs) produces inconsistent winners across diverse shape regimes.
- The operator evaluates on diverse shapes where rows and/or columns vary by orders of magnitude (e.g., rows from single digits to thousands, cols from tens to tens of thousands).
- The kernel already uses 2D grid decomposition `(cdiv(rows, BLOCK_M), cdiv(cols, BLOCK_SIZE))`.
- Autotune overhead (compile time, first-run search) is problematic, or autotune key design is fragile.
- You need per-shape-adaptive tile sizing that autotune with static configs cannot provide.

## Avoid When

- The core algorithm/layout is still changing (stabilize structure first using `program-multiple-rows`).
- The kernel uses a 1D grid that cannot benefit from 2D tile decomposition.
- UB capacity constraints force strict upper bounds on tile size (use `tiling` pattern for hierarchical tiling).
- The search space is trivially small (e.g., only 1-2 tile size choices) — manual selection suffices.

## Signals

### Code

- Kernel launches with `grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_SIZE))`.
- Existing autotune configs only vary `BLOCK_SIZE` and `BLOCK_ROWS` (or `BLOCK_M`) across a handful of values.
- Per-shape benchmarks show that the "best" tile size shifts dramatically between small and large shapes.

### Profile

- Some shapes show excessive grid programs (high launch count) while others are compute-bound.
- Autotune search overhead is visible in first-run benchmarks.
- Cached autotune winners for one shape bucket are suboptimal for others due to overly coarse keys.

## Optimization Strategy

1. Write a `_choose_tiling(rows, cols, max_block_m, max_block_size, product_limit)` function on the host.
2. Sweep candidate `block_size` values (powers-of-2 descending from `max_block_size` down to 128).
3. For each `block_size`, compute `block_m = min(max_block_m, rows, product_limit // block_size)`.
4. Compute `gm = triton.cdiv(rows, block_m)` and `gn = triton.cdiv(cols, block_size)`.
5. Reject candidates where `gm >= 65536` or `gn >= 65536` (hardware grid limit).
6. Choose the pair that minimizes total grid programs `gm × gn`.
7. The product limit controls UB residency: increase it gradually (4096 → 8192 → 16384) while monitoring for UB overflow or regressions.
8. Combine with `exact-tile-no-boundary-fast-path` when the selected tile sizes perfectly divide shape dimensions.

## Implementation sketch

```python
def _choose_tiling(rows, cols, max_block_m=128, max_block_size=4096, product_limit=16384):
    best_bm = 1
    best_bs = 1
    best_grid = rows * cols  # worst-case: one program per element
    bs_candidates = [max_block_size]
    while bs_candidates[-1] >= 128:
        bs_candidates.append(bs_candidates[-1] // 2)
    for bs in bs_candidates:
        if bs > cols:
            continue
        bm = min(max_block_m, rows, product_limit // bs)
        gm = triton.cdiv(rows, bm)
        gn = triton.cdiv(cols, bs)
        if gm >= 65536 or gn >= 65536:
            continue
        grid = gm * gn
        if grid < best_grid:
            best_grid = grid
            best_bm = bm
            best_bs = bs
    if best_bm == 1 and best_bs == 1:
        bs = max(1, min(triton.next_power_of_2(cols), max_block_size))
        best_bm = min(max_block_m, rows, product_limit // bs)
        best_bs = bs
    return best_bm, best_bs
```

### Host launch with heuristic + exact-tile fast path

```python
block_m, block_size = _choose_tiling(rows, cols)

grid = lambda meta: (
    triton.cdiv(rows, meta["BLOCK_M"]),
    triton.cdiv(cols, meta["BLOCK_SIZE"]),
)

if rows % block_m == 0 and cols % block_size == 0:
    _kernel_nomask[grid](x, out, ..., BLOCK_M=block_m, BLOCK_SIZE=block_size)
else:
    _kernel[grid](x, out, ..., BLOCK_M=block_m, BLOCK_SIZE=block_size)
```

## Tuning knobs

| Parameter | Start value | When to increase | Max safe value |
|-----------|-------------|------------------|----------------|
| `max_block_m` | 32 | Small-col shapes dominate, rows ≥ 128 | 128 (monitor UB pressure) |
| `max_block_size` | 4096 | Large-col shapes dominate (cols ≥ 8192) | 4096 (NPU hardware limit) |
| `product_limit` | 4096 | Large shapes plateau, UB not overflowing | 16384 (raise in 2x steps) |

**Product limit progression:** Start at 4096. If benchmark shows improvement without UB overflow, increase to 8192. If improvement continues, increase to 16384. Stop when UB overflow occurs or gains plateau.

## What To Verify After Applying

- Correctness across all shape/dtype combinations (the tile choice must not affect output values).
- No UB overflow on the largest tile configuration (monitor for `product_limit` violations).
- Grid programs are minimized across the full shape set compared to the baseline.
- The heuristic does not degrade small-shape performance (compare parent-vs-child per shape).
- When combined with exact-tile-no-boundary-fast-path, the no-mask kernel is selected when divisibility holds.

## Related Patterns

- `autotune` (alternative; use heuristic when autotune config set is insufficient)
- `program-multiple-rows` (prerequisite; BLOCK_M must be in place first)
- `exact-tile-no-boundary-fast-path` (composes: no-mask kernel when tile divides shape perfectly)
- `tiling` (hierarchical tiling for UB overflow prevention)
