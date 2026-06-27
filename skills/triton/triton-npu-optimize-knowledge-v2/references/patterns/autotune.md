---
priority: high
---

# Triton Autotune Pattern

## Summary

**Autotune** here means Triton’s `@triton.autotune` decorator: the runtime tries a **small, bounded** list of launch configurations (tile sizes, warp counts, pipeline stages, and other meta-parameters) and picks one that performs best on measured micro-benchmarks of the kernel.

Use this pattern when the kernel’s **shape is already reasonable** but you still have a handful of plausible launch choices and hand-tuning is unreliable. Autotune is **not** a substitute for fixing wrong structure, bad keys, or launch-site mistakes.

## When “autotune” means this card

This card applies when you are using **`@triton.autotune`** on a `@triton.jit` kernel (or an equivalent bounded config search wired the same way). Some workloads use **manual launch-size ladders** instead of the decorator; those are related but belong more under **`tiling`** and **`grid-flatten-and-ub-buffering`** unless you explicitly adopt decorator-based search.

## Use When

- The kernel body is **stable** (correctness and rough structure are settled).
- There are **several plausible** `(tile M, tile N, …)` or `(warp count, pipeline stages)` combinations, and no single choice wins on all benchmark shapes.
- You can keep the **total number of combinations small** (a practical upper bound is on the order of **20**; exceeding that often explodes compile time or search noise).
- You can define **`key=`** fields so unrelated shapes do not share a cached wrong winner.
- You can run a **parent comparison**: autotune must beat the **previous best hand-tuned** version on the same harness, not only beat an old baseline.

## Avoid When

- The bottleneck is still **wrong algorithm or layout** (autotune will only reshuffle a bad approach).
- Compile or search time dominates (large search spaces or very heavy kernels).
- Correctness depends on **accumulation order** or **shared output buffers** unless you add explicit reset or isolation for each config trial.
- Launch metadata is **duplicated** between the decorator `Config` and the launch site (this causes hard failures or silent wrong configs).

## How autotune works (short)

1. You list a small set of **`triton.Config(...)`** objects (each sets meta-parameters such as tile sizes, `num_warps`, `num_stages`).
2. You list **`key=`** arguments: values that should **separate** cached winners (for example logical mode flags, ranks, or coarse shape buckets).
3. At first launch for a given key, Triton **benchmarks** the configs and caches the fastest for that key.
4. Later launches reuse the cached choice.

If **`key=`** is too coarse, different workloads share one cached choice and performance becomes wrong. If **`key=`** is too fine, you lose cache reuse and pay more search cost.

## Recommended workflow

1. **Stabilize structure first** — fusion, layout, and correctness paths should be settled before a wide search.
2. **Add autotune with a tiny grid** — start with the smallest set that still expresses real trade-offs.
3. **Fix launch contract bugs** — every meta-parameter must be owned either by `Config` **or** by the launch site, not both (common errors: `BLOCK_M` passed twice, `num_warps` duplicated).
4. **Validate against the immediate parent** — compare to the last promoted branch on the **same** benchmark mix; beating an old baseline while regressing the parent is a failure.
5. **Re-key after refactors** — if strides, constexpr modes, or fusion boundaries change, invalidate or redesign `key=` so cached winners cannot apply to stale geometry.
6. **Prune dead configs** — if profiling or cache probes show only a few configs ever win, **removing** losers can improve measured time more than adding new guesses (search has real overhead).
7. **Try manual dispatch only with evidence** — replacing autotune with hand-written thresholds can win or lose; treat it like another experiment and compare to the autotuned parent.

## Common repairs

### Bounded config lists

Keep the Cartesian product of tunables small. Prefer a short list of meaningful pairs over enumerating every combination of many axes.

### Key design

Include anything that changes **legal tiles**, **numeric path**, or **dominant bottleneck class**:

- semantic mode flags (example: different padding or reduction modes),
- coarse dtype or width bucket,
- rank or pattern id when it changes valid tile sets.

Split keys when two modes must never share a cached winner (similar to splitting keys for unrelated pad modes).

### Launch-site hygiene

After adding `@triton.autotune`, audit the launch call:

- remove duplicate `num_warps`, `num_stages`, or block-size kwargs if `Config` already sets them,
- ensure `constexpr` and non-specialized arguments still match the kernel signature.

### Accumulating outputs

If configs benchmark against the **same output tensor**, each trial can read stale accumulation unless the autotune harness resets outputs per config. Use documented reset mechanisms when the kernel accumulates into global memory.

### Grid and hardware limits

Configs must respect hardware launch limits (maximum grid dimensions and similar). If some configs overshoot, **prune** them rather than hoping they are never selected.

### Prune after memory-pressure changes

If a prior round increased registers or temporary buffers, old warp or stage winners may **spill** or slow down. Re-run a reduced config list or prune using simple occupancy or spill heuristics before trusting old winners.

### Simplified code sketch

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
    ],
    key=["shape_bucket", "dtype_bucket"],
)
@triton.jit
def kernel(x_ptr, y_ptr, n_elements, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    ...
```

## Failure modes and anti-signals

- **Search after layout wins without re-keying** — cached choices become wrong for new stride patterns.
- **Wide search on compile-heavy kernels** — wall-clock or interactive time blows up; abandon search and shrink the grid.
- **Parent regression** — autotune improves an old baseline but loses to the immediately better hand-tuned branch; do not promote.
- **Replacing autotune with untested hand dispatch** — can regress if the manual policy does not reproduce the measured winners.
- **Accumulation without per-config reset** — silent wrong winners or flaky comparisons.
- **New side-effect ops after landing autotune** — if op statistics suddenly show extra host/device helper work (for example extra casts or zeros-like style setup), treat it as a regression signal and re-check dispatch boundaries, key design, and config validity.

## Risks

- Wrong or stale `key=` causes cross-talk between workloads.
- Over-large grids hurt developer iteration and CI time.
- Autotune hides launch mistakes until first run fails mysteriously.
- Numeric modes that share a key can pick a config that is only valid for one mode.

## What to verify after applying

- Correctness across all benchmark shapes, especially boundaries where tile validity changes.
- Performance vs **parent** branch on the same harness, not only vs an early baseline.
- Compile and first-run latency are acceptable for the chosen grid size.
- After structural follow-up rounds, re-check keys and prune configs if needed.
- Op statistics do not introduce new helper-operator churn that offsets kernel-level gains.

## Related patterns

- **`tiling`**: when the main lever is tile geometry itself rather than a small launch-meta sweep.
- **`grid-flatten-and-ub-buffering`**: when the issue is program count or staging rather than per-program meta-parameters.
- **`compile_hint`**: when lowering hints are the next small lever after autotune stabilizes.
