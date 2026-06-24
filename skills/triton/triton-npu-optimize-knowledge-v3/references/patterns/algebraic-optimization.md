# Algebraic optimization (pure math reformulation)

## Summary

Look for **semantics-preserving** rewrites that reduce **memory passes**, **redundant full scans**, or **dependency depth** before micro-tuning launch geometry.

The strongest wins usually come from changing math/dataflow shape (for example one-pass reductions, intrinsic substitution, reusable intermediates), not from launch knobs alone.

Always validate against the reference. Equivalent forms on paper can still regress after lowering to Ascend Triton (dependency chains, UB pressure, launch overhead).

## Use When

- The hot path performs **two or more full traversals** of the same data for statistics, normalization, or mergeable subexpressions.
- Profiler or IR suggests **duplicate transfer-heavy** phases differing only by scalar statistics from the same tensor.
- A manual transcendental expansion exists where a stable intrinsic can express the same semantics.
- Elementwise **logical** ops (`logical_or`, `logical_and`, etc.) with broadcasting evaluate truth on expanded numeric tensors.
- You need structural simplification **before** `tiling`, `software-pipeline`, or `autotune` retuning.

## Avoid When

- The bottleneck is clearly launch geometry / occupancy / UB footprint with no redundant algorithmic work (use `tiling` first).
- A rewrite only changes expression spelling, without reducing pass count, dependency depth, or effective work.
- Approximate formulas are introduced without strict correctness and parent-vs-child performance evidence.
- Custom Triton fusion is attempted before proving a simpler reorder or rewrite is sound.

## Signals

### Code

- Two loops (or kernels) with nearly identical `tl.load(x)` tiling over the same axis.
- Recomputed closed-form subexpressions in the hot epilogue.
- Manual transcendental forms (exp/divide compositions) in tight loops.
- `broadcast_tensors(x, y)` then truth tests on the expanded numeric tensors.

### Profile

- Repeated transfer-dense stages that could be merged if math structure were reorganized.
- `NotEqual` / `BroadcastTo` work scaling with broadcast-expanded size.
- Plateau after launch/pipeline tweaks, suggesting structural math/dataflow remains the limiter.

### IR

- Repeated load or reduction structure around the same logical axis where one traversal could feed multiple accumulators.
- Epilogue dependency chains that can be algebraically shortened.

## Failure Modes And Anti-signals

- "Cheaper-looking" approximations can regress badly against optimized intrinsic paths.
- Branchless alternatives (`max/min`-style) are not universally better than select/multiply forms.
- Algebraic edits that do not change pass count or dependency depth usually show noise-level movement.
- If an apparent gain disappears under minor launch changes, treat it as fragile secondary interaction, not robust algebraic improvement.

## Related Patterns

- `tiling`
- `program-multiple-rows`
- `software-pipeline`
- `slice-intermediate`
- `autotune`

## What To Verify After Applying

- Correctness vs reference: **FP order**, **NaN**, **dtype promotion**; for logical ops also **bool**, **complex**, **empty tensors**, and broadcast corners.
- Parent-vs-child performance under the same harness and launch regime.
- Profiler: fewer passes or lower time on the merged phase for the **same** outputs.
- UB and launch remain healthy; no new pressure that erases algebraic gains.

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

**Profiler corroboration (Ascend / msprof-style):** duplicate hot `tl.load` regions, high `MOV_*` / `WAIT_FLAG` in phases that differ only by a scalar derived from the same data - the workload may be **memory- or sync-bound** and a good candidate for **fewer passes**.

---

## Principles

1. **Prove equivalence** under the operator’s reference semantics (real-number model where relevant; then **NaN**, **dtype promotion**, **complex**, **logical** truthiness as needed).
2. **Measure**: some equivalent forms have **longer dependency chains** or **higher UB peak**; algebraic "fewer ops on paper" can **regress** after lowering.
3. **Apply one algebraic change at a time** so wins/failures remain attributable before launch/pipeline retuning.
4. **Document new catalog entries** with: *when*, *signals*, *rewrite*, *why it can help on NPU*, *risks*, and *verification*.

---

## Technique catalog

Each subsection is an independent pattern. Keep adding `### Case N` entries over time.

### Case 1: Single-pass first and second moments (mean and variance)

**When to use**

- You need **mean** and **(population) variance** (or sum of squared deviations) along a reduction axis.
- The baseline uses **two full traversals** of that axis: e.g. `sum(x)` then `sum((x - mean)**2)` or separate kernels for mean and variance.

**Signals**

- Two loops (or kernels) with **nearly identical** `tl.load(x)` tiling along the same axis.
- OPPROF: statistics phase dominated by **MTE** / **wait-on-flag**, not by vector FLOPs alone.
- Common in **LayerNorm**, **InstanceNorm**, and similar statistic-heavy paths.

**Rewrite (algebra)**

For length \(n\), \(\mu = \frac{1}{n}\sum_i x_i\), population variance \(\sigma^2 = \frac{1}{n}\sum_i (x_i-\mu)^2\).

Single traversal:

- \(S_1 = \sum_i x_i\), \(S_2 = \sum_i x_i^2\)
- \(\mu = S_1/n\), \(\sigma^2 = \max(0,\, S_2/n - \mu^2)\) (clamp fixes tiny negative FP noise before `rsqrt`)

**Why it can improve performance on NPU**

- **One fewer full memory pass** over the axis for statistics.
- One `tl.load(x)` can feed **two reductions** (`sum(x)`, `sum(x*x)`).

**Risks**

- **Numerical**: \(S_2/n - \mu^2\) can be less stable than two-pass centered sums for extreme magnitudes.
- **UB / registers**: holding both accumulators plus `x*x` may raise peak usage; co-tune with `tiling` / `program-multiple-rows`.
- **Alternatives**: block-wise Welford merge may or may not beat `S1+S2`; benchmark on target shapes.

**Verification**

- Compare against reference (for example `torch.nn.functional.layer_norm`) on representative dtypes/shapes.
- Confirm profiler shows reduced statistics-pass load pressure versus two-pass baseline.

---

### Case 2: Truth map before broadcast for elementwise logical ops (`logical_or` and friends)

**Classification (is this "pure math"?)**

- Yes: it is a **semantics-preserving rewrite** under the reference operator definition (for example `torch.logical_or`), not a behavior change.
- It is not a continuous real-number identity like Case 1; it is **logical/broadcast algebra**.

**When to use**

- Implementing logical ops with broadcasting where the naive path evaluates truth tests on full expanded numeric tensors.
- You want fewer elementwise comparisons before optional custom Triton fusion.

**Signals**

- `NotEqual`/`Cast` work scales with broadcast-expanded `numel`.
- Heavy `BroadcastTo` appears before truth tests on wide dtypes.
- Shapes like `(N,1)` vs `(1,N)` or `(N,...,K)` vs `(N,1,...,1,K)` where expanded compares touch the full output product on both operands.

**Rewrite (equivalence sketch)**

Let `T` be the reference truth map for one tensor. Under PyTorch logical semantics, truthiness is operator-defined; for non-complex numerics in common paths, `T(a) = (a != 0)` matches `torch.ne`, while `bool` is already boolean.

For broadcast output index `i` with preimage indices `i_x` into `x` and `i_y` into `y`:

\[
\text{logical\_or}(x,y)[i] = T(x[i_x]) \lor T(y[i_y]) = \big(T(x) \text{ broadcast} \lor T(y) \text{ broadcast}\big)[i].
\]

Equivalent pipeline:

1. Compute `tx = T(x)`, `ty = T(y)` on original shapes.
2. Broadcast bool tensors (or fuse OR without materializing expanded numeric inputs).
3. Apply micro-optimizations:
   - bool identity short-circuit (`bool` input does not need `ne`)
   - empty-output early return

**Why it can improve performance on NPU**

- Fewer truth tests on expanded floating-point buffers.
- Broadcast and downstream ops operate on narrower boolean dataflow.

**Risks**

- Must match reference semantics for `complex`, `NaN`, `bool`, and dtype behavior.
- Fusion/lowering remains a separate performance question; validate on target.

**Verification**

- Compare against `torch.logical_or(x, y)` across broadcast corners and dtype edge cases.
- Confirm reduction in `NotEqual` / `BroadcastTo` pressure before custom kernel retuning.

---

### Case 3: Epilogue simplification and intermediate reuse

**When to use**

- A fused epilogue computes the same subexpression multiple times.
- Equivalent epilogue forms exist (for example select/multiply variants, factored polynomial variants, domain-range simplifications).

**Signals**

- Repeated multiply/add chains around one temporary.
- Branchless alternatives or specialized constants suggested by known value ranges.

**Rewrite**

- Reuse already computed intermediates instead of recomputing.
- Factor polynomials to reduce instruction pressure while preserving semantics.
- Use range-informed simplifications only when domain guarantees are valid.

**Why it can improve performance on NPU**

- Shorter epilogue dependency chains.
- Lower scalar/vector instruction pressure in the hottest loop body.

**Risks**

- Not all algebraic forms lower equally well; "simpler" on paper can be slower.
- Some branchless variants regress versus select/multiply in practice.

**Verification**

- Compare epilogue forms under the same launch/tile/scheduler regime.
- Keep parent-vs-child checks to isolate algebraic effect from geometry changes.

---

## Relation to other patterns

| Pattern | Interaction |
|---------|-------------|
| `tiling` | Algebraic rewrites often change live ranges/footprint; retune block sizes after structural math changes. |
| `program-multiple-rows` | Row coarsening can amplify gains after pass reduction, but still must stay UB-safe. |
| `software-pipeline` | Apply after math structure is stable; pipeline does not remove extra scans by itself. |
| `slice-intermediate` | If fused algebraic paths spill UB, use staged slicing. |
| `autotune` | Use bounded search across algebraically valid variants once structure is fixed. |

---

## Reading discipline

- Treat this file as a living catalog: choose cases by symptom fit, not by one-size-fits-all preference.
- Prefer one algebraic change per round, then re-profile.
- If evidence is weak, improve profiler attribution before further math rewrites.
