---
priority: high
---

# Triton-Ascend Autotune Decision Pattern

## Summary

Use Triton-Ascend autotune as the default way to search split sizes, tile sizes, and selected compile options when the kernel structure is already reasonable and the main open question is parameter choice.

Treat this pattern as a routing rule: try fully automatic autotune first, add `hints` when parser inference is incomplete, and hand-write `triton.Config` candidates only when the search space must be constrained manually.

## Use When

- The kernel structure already looks semantically correct, and the likely headroom is in `BLOCK_*` selection, split shape, or Ascend-specific compile options such as `multibuffer`.
- The current optimization loop is drifting toward repeated manual tiling edits without strong evidence that a structural rewrite is needed first.
- The hot path exposes one or more free `tl.constexpr` parameters that are not hard-coded at launch time.
- Bounds masks or loop structure still map cleanly back to runtime shape arguments, so a shape-keyed autotune cache is plausible.
- The operator is vector-like rather than a Cube-only kernel path that needs a different optimization route.
- You are not already in a launch-mode experiment that explicitly changes execution style; if you are applying a launch-mode pattern, recheck num_warps and grid decomposition after enabling the experiment's launch mode.
- **Two-pass kernel with intermediate store+reload:** when the kernel stores an intermediate tensor in a first loop pass and reloads it in a second loop pass, hand-written autotune configs must be accompanied by UB-aware pruning. Skipping pruning and using conservative configs to avoid UB overflow limits the search space and leaves performance on the table. Pruning is what enables a wide productive block size range without UB-spill cliffs.

## Avoid When

- The real problem is structural, such as a manual matmul or reduction that should first become a regular tiled `tl.dot` loop.
- All relevant `tl.constexpr` parameters are already fixed at launch time, so the kernel exposes no meaningful tuning space.
- A semantic constraint fixes one grid dimension or one tile shape so tightly that generated candidates would mostly be invalid or meaningless.
- One parameter simultaneously controls multiple unrelated axes or both launch count and inner tile semantics in a way that automatic parsing cannot represent cleanly.
- The kernel is correctness-fragile under repeated benchmarking and has not yet added the reset or restore hooks needed for safe autotune evaluation.
- **Route 1 (`configs=[]`) generated zero or one candidate on Ascend NPU.** The Ascend autotune parser often cannot infer useful candidates for row-wise vector kernels because the BLOCK_SIZE-to-loop relationship is indirect (range loop over n_cols instead of a simple pid-split). When `TRITON_PRINT_AUTOTUNING=1` shows no candidates or a single degenerate candidate, immediately escalate to Route 3 with explicit `triton.Config` lists.

## What To Verify After Applying

- Verify the chosen route is the least manual one that still fits the kernel:
  - `configs=[]` first when parser inference should succeed
  - `hints` when semantics are clear but inference is incomplete
  - explicit `triton.Config` lists only when the search space truly needs manual control
- Verify `key` tracks the runtime shape arguments that actually change the best configuration. On Ascend NPU, never use `key=[]` when autotuned parameters affect UB usage such as block sizes. A shared cache across incompatible shapes can apply a config that overflows UB on subsequent shapes, causing catastrophic regressions or compilation failures. Use shape-dependent keys like `key=["hidden_size", "num_rows"]` so each unique shape gets its own independently benchmarked optimal config.
- Verify update-style kernels use `reset_to_zero`, `restore_value`, hooks, or equivalent safeguards so repeated autotune trials do not corrupt outputs.
- Verify the searched parameters are Ascend-relevant for the config-space search, especially `BLOCK_*`, `multibuffer`, and `unit_flag`, rather than treating GPU-only defaults such as `num_warps` or `num_stages` as the default search surface.
- Verify the selected block sizes still satisfy semantic constraints such as `BLOCK_SIZE <= tiled logical extent` when padding would otherwise change results.
- Verify `TRITON_PRINT_AUTOTUNING=1` or equivalent logs show the inferred axes, candidate count, and chosen best configuration during debugging. **If `configs=[]` produced zero candidates on Ascend NPU, do not proceed — escalate to Route 3.**
- Verify grid-limit pruning is in place when the BLOCK_SIZE search range includes values small enough that `cdiv(n_elements, BLOCK_SIZE)` could exceed the hardware grid limit (65535 on Ascend NPU). This applies even to single-pass elementwise kernels where UB pruning is not needed.
- Verify the prune function returns all configs when bounds cannot be computed, and that it never returns an empty list.
- Verify the prune function's `LIVE_TENSORS` count matches the kernel's actual live tile buffer count. An overcount prunes valid configs; an undercount lets UB-overflow configs through.

## Route 1: Automatic Autotune First

Use `configs=[]` first when split and tiling structure can be inferred directly from the kernel DSL.

Typical signals:

- the free tuning parameters are `tl.constexpr` values not fixed at launch time
- split parameters come from `tl.program_id`
- tiling parameters come from `tl.arange` or loop step structure
- masks or bounds expressions map cleanly back to runtime shape axes

```python
@triton.autotune(
    configs=[],
    key=["n_rows"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)
```

**After setting `configs=[]`, verify the parser actually generated candidates.** Run with `TRITON_PRINT_AUTOTUNING=1` and check that the candidate count is ≥ 2. On Ascend NPU, the automatic parser often produces zero candidates for row-wise elementwise kernels — the BLOCK_SIZE-to-loop mapping (e.g., a `range(0, n_cols, BLOCK_SIZE_N)` loop) is not simple enough for the parser to recognize as a tiling parameter. If zero or one candidate: escalate to Route 3 immediately; do not leave `configs=[]` in place hoping it will work.

## Route 2: Add `hints` Before Hand-Writing Configs

Use `hints` when the kernel still fits auto-generated search, but parser inference is incomplete.

Typical signals:

- the split or tiling parameter is semantically clear to a human reviewer
- the path from `program_id` or `tl.arange` to masks is indirect
- low-dimensional or reduction axes need to be stated explicitly

When using `hints`, prefer axis-named `key` mappings so the cache aligns with the hinted axes.

```python
@triton.autotune(
    configs=[],
    key={"x": "n_rows", "y": "n_cols"},
    hints={
        "split_params": {"x": "BLOCK_M"},
        "tiling_params": {"y": "BLOCK_N"},
        "low_dim_axes": ["y"],
        "reduction_axes": [],
    },
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    for n0 in range(0, n_cols, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)[None, :]
        mask = (offs_m < n_rows) & (offs_n < n_cols)
        x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
        tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

## Route 3: Hand-Write `triton.Config` Candidates Last

Use explicit `triton.Config` lists only when the search space must be constrained manually.

**For two-pass kernels (store intermediate + reload): Route 4 pruning is mandatory.** Do not deploy hand-written configs on a store+reload kernel without `prune_configs_by`. The autotuner can select BLOCK combinations that overflow UB (192 KB), causing silent HBM-spill performance cliffs. Narrowing the config range to avoid overflow is not a substitute — it caps the search space and sacrifices the performance that autotune is meant to deliver. Use wide config ranges with pruning.

Typical signals:

- one grid axis is fixed by semantics and cannot be freely split
- one parameter couples launch count and inner tile shape
- the kernel exposes too little clean tuning structure for automatic generation
- candidate quality is still poor after adding `hints`

On Triton-Ascend, the main hand-written search dimensions should usually be:

- `BLOCK_*` sizes
- `multibuffer`
- `unit_flag` when relevant

For **Cube/matmul kernels** (those using `tl.dot` as the primary compute), `num_warps` is secondary — focus on `BLOCK_*` and `multibuffer` first. For **vector/elementwise kernels** (no `tl.dot`, row-wise loads and stores), `num_warps` × `BLOCK_SIZE` is a productive primary search space. Different warp counts change how the vector unit schedules loads and stores, and the optimal `num_warps` × `BLOCK_SIZE_M` combination shifts with problem shape. Test 2-3 warp counts (typically 4, 8, 16) across the full `BLOCK_SIZE_M` range.

```python
def get_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "multibuffer": mb})
        for bm in [256, 128, 64, 32]
        for bn in [128, 64, 32, 16]
        for mb in [True, False]
    ]


@triton.autotune(
    configs=get_configs(),
    key=["n_rows", "n_cols"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < n_rows) & (offs_n < n_cols)
    x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
    tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

## Route 4: Prune Configs by Capacity or Grid Limits

Use `prune_configs_by={"early_config_prune": prune_fn}` to filter autotune configs. Two kinds of pruning apply, often both on the same kernel:

- **Grid-limit pruning**: applies to **any** kernel (single-pass or multi-pass). When hand-written configs include small BLOCK_SIZE values, `cdiv(n_elements, BLOCK_SIZE)` can exceed the hardware grid limit (65535 on Ascend NPU). Add a grid-limit filter whenever the BLOCK_SIZE search range includes values that could overflow the grid maximum.
- **UB-capacity pruning**: applies to multi-pass kernels where multiple tile buffers are simultaneously live and could overflow the 192 KB Unified Buffer.

### Grid-limit pruning (applies to all kernels)

Signals:
- Hand-written configs include BLOCK_SIZE values small enough that `cdiv(n_elements, BLOCK_SIZE)` could exceed the hardware grid limit (65535 on Ascend NPU).
- The kernel is single-pass elementwise — UB pruning is not needed, but grid-limit pruning still applies.

```python
def prune_grid_limit(configs, named_args, **__):
    n_elements = named_args.get("n_elements", 0)
    if not isinstance(n_elements, int) or n_elements <= 0:
        return configs
    MAX_GRID = 65535
    pruned = [c for c in configs
              if triton.cdiv(n_elements, c.kwargs.get("BLOCK_SIZE", 1)) <= MAX_GRID]
    return pruned if pruned else configs

@triton.autotune(
    configs=get_configs(),
    key=["n_elements"],
    prune_configs_by={"early_config_prune": prune_grid_limit},
)
@triton.jit
def kernel(..., n_elements, BLOCK_SIZE: tl.constexpr, ...):
    ...
```

Return all configs when `n_elements` is unknown. Never return an empty list.

### UB-capacity pruning (multi-pass kernels)

**The primary trigger is kernel structure, not config range.** Any two-pass kernel (store intermediate in first loop + reload in second loop) must include pruning, regardless of how narrow or conservative the current config range is. Without pruning, the autotuner can select BLOCK combinations that overflow 192 KB UB. Using narrow configs to avoid overflow is not a valid workaround — it limits the search space and sacrifices the performance that autotune is meant to deliver. Pruning is what enables wide productive ranges (e.g., BLOCK_SIZE_M [4..128]) without risk.

Signals for UB pruning:

- **Two-pass kernel with store+reload** — this is the strongest and most common signal. The kernel has separate loops where the first loop stores an intermediate tensor (`tl.store`) and the second loop reloads that same tensor (`tl.load`). Both passes' tile buffers are simultaneously live. This structure mandates pruning; do not skip it.
- Configs span a wide `BLOCK_*` range and the kernel uses multi-tile live buffers (intermediate store+reload, multiple input tensors simultaneously live).
- Wide-column cases regress unexpectedly — the autotuner selected a BLOCK that overflows UB.
- The kernel loads multiple tile-row blocks into registers/UB per loop iteration.

Anti-signal — a prune function that only checks `BLOCK_SIZE_N <= n_cols` or similar dimension validity is NOT UB-aware pruning. It will not prevent HBM-spill performance cliffs. To be UB-aware, the prune must compute a capacity bound (`UB_BYTES / (n_cols * elem_size * LIVE_TENSORS)`) and filter configs against it.

Prune function pattern:

```python
def prune_ub(configs, named_args=None, **__):
    if named_args is None:
        return configs
    n_cols = named_args.get("n_cols", 0)
    # Get element size from any live tensor pointer
    ptr = named_args.get("X_ptr") or named_args.get("Y_ptr")
    if not isinstance(n_cols, int) or n_cols <= 0 or ptr is None:
        return configs
    try:
        elem_size = ptr.element_size()
    except Exception:
        return configs

    UB_BYTES = 192 * 1024
    # LIVE_TENSORS: count all simultaneously live tile buffers.
    # Two-pass kernels (store intermediate + reload): use 3 or more.
    # Single-pass kernels (no intermediate store): use a lower count.
    LIVE_TENSORS = 3
    max_rows = UB_BYTES // (n_cols * elem_size * LIVE_TENSORS)
    max_rows = max(4, min(128, max_rows))

    pruned = [c for c in configs
              if c.kwargs.get("BLOCK_SIZE_M", 64) <= max_rows]
    return pruned if pruned else configs

@triton.autotune(
    configs=get_configs(),
    key=["n_rows", "n_cols"],
    prune_configs_by={"early_config_prune": prune_ub},
)
@triton.jit
def kernel(..., n_rows, n_cols, BLOCK_SIZE_M: tl.constexpr, ...):
    ...
```

Key rules:

- **Pick LIVE_TENSORS conservatively.** It counts every tile buffer simultaneously live. Two-pass paths with store+reload need more than single-pass. Start at 3 for two-pass elementwise and reduce cautiously with profiling evidence.
- **Always fall back to all configs** when the prune function cannot compute UB bounds (unknown element size, missing named args). Returning an empty list causes autotune failure.
- **Floor and cap max_rows** so small n_cols don't force ridiculously large blocks and large n_cols don't force single-row blocks.
- **per-dtype awareness**: the formula automatically accounts for element size — fp32 (4 bytes) gets pruned more aggressively than fp16/bf16 (2 bytes).
- **Dimension-only pruning is NOT UB-aware.** A prune function that only checks `BLOCK_SIZE_N <= n_cols` or similar dimension validity is a correctness guard, not a UB-capacity filter. It will not prevent HBM-spill performance cliffs from oversized `BLOCK_SIZE_M` configs. The prune must compute a capacity bound from `UB_BYTES / (n_cols * elem_size * LIVE_TENSORS)` and filter `BLOCK_SIZE_M` against that bound.

## When Automatic Parsing Usually Fails

Prefer `hints` or custom configs when you see one or more of the following:

- the kernel has no meaningful free `tl.constexpr` parameters because they are fixed at launch or coupled too tightly to semantics
- no clear mask or bounds relation back to the runtime axis
- one parameter must cover an entire semantic dimension, such as `BLOCK_SIZE >= hidden_dim`
- a business or semantic rule fixes one grid dimension instead of allowing free tiling
- one parameter influences multiple axes at once

## Ascend-Specific Notes

- For **Cube/matmul kernels**, default config-space search should focus on `BLOCK_*`, `multibuffer`, and `unit_flag`. For **vector/elementwise kernels**, `num_warps` × `BLOCK_SIZE` is a productive search dimension — test 2-3 warp counts (4, 8, 16) alongside BLOCK_SIZE sweeps.
- **Never use `key=[]` on Ascend NPU when autotuned parameters affect UB usage.** The Ascend NPU UB capacity is limited to ~192 KB. A config selected for one shape may overflow UB on a different shape with wider columns or more rows per program. With `key=[]`, the cached config is shared across all shapes, causing catastrophic regressions when the cached BLOCK size is incompatible. Always use shape-dependent keys (e.g., `key=["hidden_size", "num_rows"]`) so each unique shape configuration gets its own independently benchmarked optimal config. More keys mean more autotune benchmarking overhead — choose keys that meaningfully differentiate shapes without creating excessive unique combinations.
- When launch hints interact, include a small bounded set of Ascend-relevant options such as `multibuffer`, `set_workspace_multibuffer`, or `enable_auto_bind_sub_block` instead of hand-picking one globally.
- If you are applying `a5-force-simt-only-discrete-access`, recheck `num_warps` and grid decomposition there after enabling `force_simt_only=True`.
- For any hand-written configs in a multi-pass kernel (one that stores an intermediate tensor in one loop and reloads it in another), add `prune_configs_by={"early_config_prune": prune_fn}` to filter configs that exceed UB capacity (192 KB). See Route 4 for the prune function pattern and LIVE_TENSORS calibration. Do not skip pruning by using conservative hand-tuned BLOCK bounds — that limits the search space and leaves performance on the table. Pruning is what enables a wide productive config range without UB overflow.
- For update-style kernels, repeated autotune evaluation can write outputs multiple times. Add `reset_to_zero`, `restore_value`, `pre_hook`, or `post_hook` before trusting benchmarks.
- Start debugging with `TRITON_PRINT_AUTOTUNING=1`.

## Related Patterns

- `tiling`: use it first when the kernel still needs a better tiled structure before any search space should be explored.
- `software-pipeline`: use it when the tile structure is already good and the next issue is overlap quality rather than parameter choice.
- `a5-force-simt-only-discrete-access`: use it when A5 is confirmed and the kernel is discrete-memory-access dominated; that launch-mode experiment intentionally rechecks `num_warps` and grid decomposition after changing the launch mode.
- `compiler-intrinsic-lowering-analysis` — intrinsic substitution can shift compute-vs-memory balance, potentially changing the optimal autotune configuration.
- `unrolled-dma-compute-overlap` — the sub-block structure must be paired with autotune configs that include `num_stages=2`.
