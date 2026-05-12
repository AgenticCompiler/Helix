# Triton Autotune Pattern

## Summary

**Autotune** here means Triton’s `@triton.autotune` decorator: runtime benchmarks a **small, bounded** set of launch/meta configurations and caches the fastest by key.

Use this after kernel structure is already sound. Autotune is a bounded search tool, not a replacement for fixing wrong algorithm/layout/dispatch contracts.

Practical rule: keep the total config combinations modest (roughly <=20) unless there is clear evidence that a larger grid is worth compile/search cost.

## When “autotune” means this card

This card applies when launch/meta choices are selected by `@triton.autotune` (or an equivalent bounded config search with the same behavior).

Manual tuple ladders without the decorator are related, but usually belong primarily to `tiling` and `grid-flatten-and-ub-buffering` unless converted into a true autotune config set.

## Use When

- Kernel logic is stable and correct, but several plausible launch/meta tuples remain.
- Hand-picked parameters are inconsistent across shape regimes.
- You can define reliable `key=` dimensions so unrelated workloads do not share one cached winner.
- You can compare against the immediate best parent, not only an old baseline.

## Avoid When

- Core algorithm/layout is still changing quickly.
- Search space is too large relative to compile/first-run budget.
- Launch contract ownership is unclear (same meta passed in both decorator and launch kwargs).
- Config trials write to shared accumulating outputs without per-trial reset/isolation.

## Signals

### Code

- Repeated manual edits of `BLOCK_*`, `num_warps`, `num_stages` with no robust winner.
- Duplicate launch metadata between decorator configs and call site.
- Key dimensions omit semantic/shape modes that change valid or optimal tuples.

### Profile

- Multiple nearby tuples perform similarly, and winner shifts by shape/mode bucket.
- Parent branch still shows underutilization after structural optimizations.
- Search overhead or compile churn becomes visible due to over-wide config grids.

## How autotune works (short)

1. `configs=[triton.Config(...), ...]` defines candidate tuples.
2. `key=[...]` defines cache partitioning dimensions.
3. First launch for a key benchmarks all configs and caches the winner.
4. Later launches for that key reuse the cached winner.

If keys are too coarse, wrong winners cross workload regimes. If keys are too fine, cache reuse collapses and search overhead rises.

## Recommended workflow

1. Stabilize structure first.
2. Start with a small config grid covering meaningful trade-offs only.
3. Fix launch contract errors (single ownership for each meta-parameter).
4. Validate against the immediate parent on the same harness.
5. Re-key after structural refactors.
6. Prune dead configs aggressively when evidence shows they never win.
7. Only replace autotune with manual dispatch when measurements prove a gain.

## Common repairs

### Bounded config lists

Use a short, intentional list of tuples; avoid blind Cartesian explosion.

### Key design

Include dimensions that materially alter valid/optimal tuples (mode flags, coarse shape/dtype buckets, layout class).

### Launch-site hygiene

Do not pass `num_warps`, `num_stages`, or block meta in both config and launch kwargs.

### Per-config isolation for accumulating outputs

Ensure each config trial compares on clean output state when kernels accumulate into global memory.

### Grid/hardware legality pruning

Remove configs that violate launch/hardware constraints instead of leaving them in the grid.

### Re-prune after memory-pressure changes

After major structural shifts, old winners may spill/regress; rerun bounded search and drop stale losers.

## Failure modes and anti-signals

- Autotune improves a historical baseline but regresses the current parent.
- Manual replacement of autotune without equivalent evidence causes regressions.
- Key mismatches create cached cross-talk across modes.
- Over-wide grids spend more time compiling/searching than improving runtime.
- Post-autotune rounds add helper op churn (casts/expands/zeroing) that erases kernel gains.

## Risks

- Stale keys after refactors.
- Hidden launch-contract bugs that surface only at runtime.
- Numeric/mode variants sharing one key despite different safe tuples.

## What to verify after applying

- Correctness across representative shape/mode boundaries.
- Performance vs immediate parent on same benchmark mix.
- Compile and first-run overhead remain acceptable.
- Key partitioning still matches structural assumptions after follow-up rounds.

## Related patterns

- `tiling`
- `grid-flatten-and-ub-buffering`
- `compile_hint`

## Reference examples

These preserved examples show canonical decorator usage patterns used in prior versions of this card.

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4, 5]
    ],
    key=["H", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    USE_TMA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    ...
```

```python
BS_LIST = [32, 64]

@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    ...
```
