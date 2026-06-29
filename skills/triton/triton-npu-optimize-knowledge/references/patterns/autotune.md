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
- You are not already in a launch-mode experiment that explicitly changes execution style; if you are applying the A5 SIMT-only discrete-access pattern, `num_warps` and grid decomposition are rechecked there after `force_simt_only=True`.

## Avoid When

- The real problem is structural, such as a manual matmul or reduction that should first become a regular tiled `tl.dot` loop.
- All relevant `tl.constexpr` parameters are already fixed at launch time, so the kernel exposes no meaningful tuning space.
- A semantic constraint fixes one grid dimension or one tile shape so tightly that generated candidates would mostly be invalid or meaningless.
- One parameter simultaneously controls multiple unrelated axes or both launch count and inner tile semantics in a way that automatic parsing cannot represent cleanly.
- The kernel is correctness-fragile under repeated benchmarking and has not yet added the reset or restore hooks needed for safe autotune evaluation.

## What To Verify After Applying

- Verify the chosen route is the least manual one that still fits the kernel:
  - `configs=[]` first when parser inference should succeed
  - `hints` when semantics are clear but inference is incomplete
  - explicit `triton.Config` lists only when the search space truly needs manual control
- Verify `key` tracks the runtime shape arguments that actually change the best configuration.
- Verify update-style kernels use `reset_to_zero`, `restore_value`, hooks, or equivalent safeguards so repeated autotune trials do not corrupt outputs.
- Verify the searched parameters are Ascend-relevant for the config-space search, especially `BLOCK_*`, `multibuffer`, and `unit_flag`, rather than treating GPU-only defaults such as `num_warps` or `num_stages` as the default search surface.
- Verify the selected block sizes still satisfy semantic constraints such as `BLOCK_SIZE <= tiled logical extent` when padding would otherwise change results.
- Verify `TRITON_PRINT_AUTOTUNING=1` or equivalent logs show the inferred axes, candidate count, and chosen best configuration during debugging.
- Verify any custom `early_config_prune` / `prune_configs_by` callback reads runtime shape arguments from the positional `nargs` argument, not from the `**kwargs` / `**named_args` meta dict. The autotuner calls the prune callable as `early_config_prune(configs, nargs, **kwargs)`: `nargs` (built from `dict(zip(arg_names, args))`) holds runtime args like `n_cols`, while `**kwargs` holds only constexpr/meta such as `BLOCK_SIZE`. Reading a runtime shape via `named_args.get(...)` silently returns the default and disables pruning, so a too-small config can win and regress both perf and correctness.

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

Typical signals:

- one grid axis is fixed by semantics and cannot be freely split
- one parameter couples launch count and inner tile shape
- the kernel exposes too little clean tuning structure for automatic generation
- candidate quality is still poor after adding `hints`

On Triton-Ascend, the main hand-written search dimensions should usually be:

- `BLOCK_*` sizes
- `multibuffer`
- `unit_flag` when relevant

Do not import GPU-first search habits blindly. `num_warps` and `num_stages` are not the primary Ascend tuning knobs.

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

## When Automatic Parsing Usually Fails

Prefer `hints` or custom configs when you see one or more of the following:

- the kernel has no meaningful free `tl.constexpr` parameters because they are fixed at launch or coupled too tightly to semantics
- no clear mask or bounds relation back to the runtime axis
- one parameter must cover an entire semantic dimension, such as `BLOCK_SIZE >= hidden_dim`
- a business or semantic rule fixes one grid dimension instead of allowing free tiling
- one parameter influences multiple axes at once

## Ascend-Specific Notes

- Default config-space search should focus on `BLOCK_*`, `multibuffer`, and `unit_flag`, not treat `num_warps` or `num_stages` as the default Ascend autotune surface.
- When launch hints interact, include a small bounded set of Ascend-relevant options such as `multibuffer`, `set_workspace_multibuffer`, or `enable_auto_bind_sub_block` instead of hand-picking one globally.
- If you are applying `a5-force-simt-only-discrete-access`, recheck `num_warps` and grid decomposition there after enabling `force_simt_only=True`.
- For update-style kernels, repeated autotune evaluation can write outputs multiple times. Add `reset_to_zero`, `restore_value`, `pre_hook`, or `post_hook` before trusting benchmarks.
- Start debugging with `TRITON_PRINT_AUTOTUNING=1`.

## Failure mode: prune callback reads a runtime arg from the wrong dict

A custom `early_config_prune` / `prune_configs_by` callback can silently disable pruning by looking up a **runtime** shape argument in the **meta/constexpr** argument dict. When pruning never fires, illegal or too-small configs survive the search and can win, producing **both** a performance regression **and** a numeric error at the same time. This is a prune-contract bug, not an autotune search bug — and it is easy to misattribute to autotune itself, so confirm the prune callback before blaming configs or keys.

### How Triton splits arguments (the "why")

The autotuner keeps two argument channels, and a prune callback must read from the right one:

- **Runtime arguments** — tensor pointers, scalar sizes such as `n_cols`, strides: anything the kernel receives as a positional argument at launch. These are packed into `self.nargs`:

  ```python
  self.nargs = dict(zip(self.arg_names, args))   # built in run()/warmup() from positional *args
  ```

- **Meta / constexpr arguments** — the `Config` tunables such as `BLOCK_SIZE`, `num_warps`, `num_stages`. These arrive as keyword arguments to `run()` and are forwarded as the `**kwargs` tail.

The prune callable is invoked as:

```python
pruned_configs = self.early_config_prune(self.configs, self.nargs, **kwargs)
#                                                ^^^^^^^^^^^^   ^^^^^^^^
#                                                runtime args    constexpr/meta
```

So the **positional** second argument holds runtime shapes; the `**kwargs`/`**named_args` tail holds only constexpr/meta values. Runtime shapes are **never** in the `**kwargs`/`**named_args` tail.

This split has been stable across versions: runtime shapes have lived in the positional `nargs` argument from Triton 2.x through 3.2 (and triton-ascend 3.2.1). The only cross-version change is whether the constexpr `**kwargs` tail is forwarded at all — Triton 2.x calls `self.early_config_prune(self.configs, self.nargs)` with no `**kwargs`, while Triton 3.0+ forwards the constexpr `**kwargs` tail as well. In every version, runtime shapes are in `nargs`, never in `**kwargs`. Do not write a prune callback that reads a runtime shape from `**named_args`/`**kwargs` on the assumption that "shapes moved there" — they did not.

### Buggy shape (the "so what")

A common mistake is to read a runtime shape from the `**named_args`/`**kwargs` tail, expecting it to be there:

```python
def _prune_configs(configs, nargs, **named_args):
    n = named_args.get("n_cols", 0)   # BUG: n_cols is a runtime arg -> it lives in nargs, not named_args
    return [c for c in configs if c.kwargs.get("BLOCK_SIZE", 0) >= n]
```

`named_args` is the `**kwargs` tail = constexpr/meta only, so it never contains `n_cols`. `.get("n_cols", 0)` always returns the `0` default, the predicate `BLOCK_SIZE >= 0` is always true, and **no config is ever pruned**. A `BLOCK_SIZE` smaller than the runtime extent it must cover then survives the search and can be picked as best — which both regresses performance and breaks numerics when the kernel body assumes `BLOCK_SIZE` covers the whole logical dimension (for example a reduction or mask that is only correct when `BLOCK_SIZE >= n_cols`).

Corrected shape — read runtime shape args from the positional `nargs` argument:

```python
def _prune_configs(configs, nargs, **named_args):
    n = nargs.get("n_cols", 0)              # runtime arg -> read from the positional nargs
    return [c for c in configs if c.kwargs.get("BLOCK_SIZE", 0) >= n]
```

### Detection signals

- a prune predicate is keyed on a runtime shape (for example `n_cols`, `hidden_dim`) but reads it via `.get(...)` on the `**named_args` / `**kwargs` meta dict instead of the positional `nargs`.
- `TRITON_PRINT_AUTOTUNING=1` shows candidate configs that should have been pruned still being benchmarked, and a too-small `BLOCK_SIZE` selected as best.
- the selected `BLOCK_SIZE` is smaller than the runtime extent it is supposed to cover (for example `BLOCK_SIZE < n_cols`), and a numeric mismatch appears together with a performance regression at the same time.
- autotune regresses on both perf and correctness at once with no structural change to the kernel body — suspect the prune contract before suspecting the search.

## Related Patterns

- `tiling`: use it first when the kernel still needs a better tiled structure before any search space should be explored.
- `software-pipeline`: use it when the tile structure is already good and the next issue is overlap quality rather than parameter choice.
- `a5-force-simt-only-discrete-access`: use it when A5 is confirmed and the kernel is discrete-memory-access dominated; that launch-mode experiment intentionally rechecks `num_warps` and grid decomposition after `force_simt_only=True`.
