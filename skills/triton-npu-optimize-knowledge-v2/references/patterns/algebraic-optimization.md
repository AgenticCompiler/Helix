# Algebraic optimization (pure math reformulation)

## Summary

Look for **semantics-preserving** rewrites that reduce **memory passes**, **redundant full scans**, or **live ranges** before micro-tuning loads. The scope includes **floating-point identities** (for example single-pass mean/variance) and **operator-defined** equivalences (for example PyTorch **logical** ops with dtype-specific truthiness and broadcasting). Always validate against the reference; forms that are equivalent on paper can still **regress** after lowering to Ascend Triton (dependency chains, UB pressure, launch overhead).

## Use When

- The hot path performs **two or more full traversals** of the same data for statistics, normalization, or mergeable closed-form subexpressions.
- Profiler or IR suggests **duplicate MTE-heavy** phases that differ only by a scalar statistic of the same tensor.
- Elementwise **logical** ops (`logical_or`, `logical_and`, …) use **broadcasting**, and truth tests (`ne`, `!= 0`) run on **fully expanded** numeric tensors.
- You want fewer global passes or cheaper elementwise work **before** changing tile sizes, pipelines, or autotune grids.

## Avoid When

- The bottleneck is clearly **only** bad tile size or UB overflow with **no** redundant algorithmic passes (prefer `tiling` or footprint patterns first).
- Custom Triton fusion is attempted before a simpler **host/graph reorder** is proven correct and cheaper end-to-end.

## Signals

### Code

- Two loops or kernels with nearly identical `tl.load(x)` tiling along the same axis.
- `broadcast_tensors(x, y)` followed by elementwise truth tests on **wide dtypes** over the full broadcast shape.

### Profile

- `NotEqual` / `BroadcastTo` (or equivalent ops) scale with **broadcast-expanded** `numel`, not with `numel(x) + numel(y)`.
- Repeated transfer-dense stages that could be merged if math structure were reorganized.

### IR

- Repeated load or reduction structure around the same logical axis where a single pass could feed multiple accumulators (case-dependent).

## Related Patterns

- `tiling`
- `program-multiple-rows`
- `software-pipeline`
- `slice-intermediate`
- `autotune`

## What To Verify After Applying

- Correctness vs reference: **FP order**, **NaN**, **dtype promotion**; for logical ops also **bool**, **complex**, **empty tensors**, and broadcast corners.
- Profiler: fewer passes or lower time on the merged phase for the **same** outputs.
- UB and launch: the rewrite does not force unsafe per-program working sets on NPU.

---

## Feature scan checklist

Use this as a **grep-for-the-brain** over source and profiler hints. If **yes**, consider an algebraic rewrite *before* micro-tuning loads.

| # | Feature in code / profiler | Typical algebraic lever |
|---|----------------------------|-------------------------|
| F1 | **Repeated full scans** of the same tensor axis for statistics (e.g. `sum` then second pass with `mean`) | Fuse passes via **extra accumulators** per tile (moments, prefix sums, running aggregates) |
| F2 | **Redundant recomputation** of the same closed-form subexpression inside a loop | **Hoist** or replace with **incremental / merged** formula |
| F3 | **Explicit normalization** `x / sqrt(eps + sum(...))` with a reduction that could share partials | Combine **variance and norm** derivations so one reduction feeds both |
| F4 | **Block-wise** algorithms that **re-load** the same block to apply a global statistic | Merge blocks using **algebraic merge rules** (e.g. parallel prefix, Welford merge) so **one load** feeds merge |
| F5 | **Symmetric** or **idempotent** math that suggests halving work (e.g. `min(a,b)+max(a,b)`) | Exploit identities to **cut evaluations** |
| F6 | **Elementwise logical** after **broadcast** where truth tests run on **fully expanded** numeric tensors | **Truth map on original shapes**, then **broadcast bool masks**; **`bool` identity** short-circuit; **empty** `out_shape` early exit |

**Profiler corroboration (Ascend / msprof-style):** duplicate hot `tl.load` regions, high `MOV_*` / `WAIT_FLAG` in phases that differ only by a scalar derived from the same data—the workload may be **memory- or sync-bound** and a good candidate for **fewer passes**.

---

## Principles

1. **Prove equivalence** under the operator’s reference semantics (real-number model where relevant; then **NaN**, **dtype promotion**, **complex**, **logical** truthiness as needed).
2. **Measure**: some equivalent forms have **longer dependency chains** or **higher UB peak**; algebraic “fewer ops on paper” can **regress** after lowering.
3. **Document** new catalog entries with: *when*, *signals*, *rewrite*, *why it can help on NPU*, *risks*, *verification*.

---

## Technique catalog

Each subsection is an independent pattern. Add new `### Case N` entries over time.

### Case 1: Single-pass first and second moments (mean & variance)

**When to use**

- You need **mean** and **(population) variance** (or sum of squared deviations) along a reduction axis.
- The baseline uses **two full traversals** of that axis: e.g. `sum(x)` then `sum((x - mean)**2)` or separate kernels for mean and variance.

**Signals**

- Two loops (or kernels) with **nearly identical** `tl.load(x)` tiling along the same axis.
- OPPROF: statistics phase dominated by **MTE** / **wait-on-flag**, not by vector FLOPs alone.
- Common in **LayerNorm**, **InstanceNorm**, **batch variance** along a contiguous dimension.

**Rewrite (algebra)**

For length \(n\), \(\mu = \frac{1}{n}\sum_i x_i\), population variance \(\sigma^2 = \frac{1}{n}\sum_i (x_i-\mu)^2\).

Single traversal:

- \(S_1 = \sum_i x_i\), \(S_2 = \sum_i x_i^2\)
- \(\mu = S_1/n\), \(\sigma^2 = \max(0,\, S_2/n - \mu^2)\) (clamp fixes tiny negative FP noise before `rsqrt`)

**Why it can improve performance on NPU**

- **One fewer full memory pass** over the axis for statistics; MTE and loop synchronization are often amortized.
- One `tl.load(x)` can feed **two reductions** (`sum(x)`, `sum(x*x)`).

**Risks**

- **Numerical**: \(S_2/n - \mu^2\) can be less stable than two-pass centered sums for extreme magnitudes; align with product numerics requirements.
- **UB / registers**: holding both accumulators plus `x*x` may **raise peak UB** vs. one accumulator; co-tune with **`tiling`** / **`program-multiple-rows`**.
- **Alternatives**: **block-wise Welford merge** keeps a single pass but adds **merge arithmetic**; on some stacks it **does not** beat `S1+S2`—**benchmark**.

**Verification**

- Compare against reference (e.g. `torch.nn.functional.layer_norm`) on representative dtypes/shapes.
- Confirm profiler: **fewer** attributed loads or cycles in the statistics phase vs. two-pass baseline.

---

### Case 2: Truth map before broadcast for elementwise logical ops (`logical_or` / friends)

**Classification (is this “pure math”?)**

- **Yes, in the sense of this catalog**: it is a **semantics-preserving rewrite** under the **reference operator definition** (e.g. `torch.logical_or`), not a change to the intended output values.
- It is **not** a continuous real-number identity like Case 1; it is **logical / discrete** equivalence plus **broadcast algebra**: truthiness commutes with broadcasting so that per-output-index results match `logical_or(broadcast(x), broadcast(y))`.
- **Prove equivalence** under the reference (including `NaN`, `complex`, `bool`), then **measure**—fusion into custom Triton is optional and may still regress on a given NPU stack.

**When to use**

- Implementing **`torch.logical_or(x, y)`** (or similar) with **broadcasting**, where the naive pipeline does: `xb, yb = broadcast_tensors(x, y)` then **`ne` / truth tests on the full expanded numeric tensors**.
- You want fewer elementwise comparisons (and often less pressure from large temporary/broadcast paths) **before** custom Triton or mask kernels.

**Signals**

- Profiler / op statistics: **`NotEqual`** (or `Cast` around masks) scales with **broadcast-expanded** `numel`, not with `numel(x)+numel(y)`.
- Heavy **`BroadcastTo`** (or equivalent) appears **before** truth tests on wide dtypes (`fp16`/`bf16`/`fp32`).
- Shapes like `(N,1)` vs `(1,N)` or `(N,…,K)` vs `(N,1,…,1,K)` where expanded compares would touch **Θ(product of output dims)** on both operands.

**Rewrite (equivalence sketch)**

Let `T` be the reference **truth map** for one tensor (PyTorch: `logical_or` uses truthiness; for non-complex numerics, `T(a) = (a != 0)` matches `torch.ne` for the usual cases; `bool` is already boolean).

For broadcast output index `i` with preimage indices `i_x` into `x` and `i_y` into `y` (standard broadcast rules):

\[
\text{logical\_or}(x,y)[i] = T(x[i_x]) \lor T(y[i_y]) = \big(T(x) \text{ broadcast} \lor T(y) \text{ broadcast}\big)[i].
\]

Equivalent pipeline:

1. **`tx = T(x)`, `ty = T(y)`** on **original** shapes.
2. **`txb, tyb = broadcast_tensors(tx, ty)`** as **bool** (or fuse `∨` without materializing full numeric broadcast).
3. Micro-optimizations:
   - **`bool` short-circuit**: if `x.dtype is bool`, then `tx = x` (no `ne`).
   - **Empty output**: if `prod(out_shape)==0`, return `empty(out_shape, bool)` without broadcasting inputs (use `broadcast_shapes` for metadata).

**Why it can improve performance on NPU**

- **Fewer elementwise truth tests** on **expanded** floating-point buffers: compare each **stored** element of `x` and `y` once, then broadcast **bool** (narrower).
- Profiler: can shrink **`NotEqual`** / **`BroadcastTo`** when those were tied to naive ordering.

**Risks**

- **Semantics**: match reference for **`complex`** (`view_as_real` + combine real/imag), **`NaN`**, **`bool`**, and dtype rules—do not assume `!= 0` without checking each dtype.
- **Fusion is separate**: lowering to Triton may **regress** (indexing, UB, launch)—validate independently.

**Verification**

- Compare vs **`torch.logical_or(x, y)`** on dtypes/shapes (broadcast corners, empty, `NaN`, `complex` if supported).
- Profiler: reduced **`NotEqual`** / **`BroadcastTo`** before optional custom kernels.

---

### Case 3: (Reserved)

Add the next pure-math optimization here with the same subsection structure: *When*, *Signals*, *Rewrite*, *Why NPU*, *Risks*, *Verification*.

---

## Relation to other patterns

| Pattern | Interaction |
|---------|-------------|
| `tiling` | Algebraic rewrites often change **tile live ranges**; UB limits may force **smaller** `BLOCK_*` after fusion. |
| `program-multiple-rows` | More rows per program can **amortize** control overhead after you reduce passes—still subject to tile caps. |
| `software-pipeline` | Use **after** the math structure is stable; pipeline does not remove an **extra full scan** by itself. |
| `slice-intermediate` | If a fused algebraic path **spills UB**, stage with slices or revert a sub-expression. |
| `autotune` | Tune between **algebraic variants** with **safety bounds** on tile sizes. |

---

## Reading discipline

- Treat this file as a **living catalog**: use **Case 1** when F1/F3 match; use **Case 2** when F6 / logical-broadcast patterns match; do not blanket-apply one formula.
- Prefer **one** algebraic change per round, then re-profile.
- If evidence is weak, strengthen with **profiler attribution** before rewriting math.
