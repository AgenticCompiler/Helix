# Loop-Invariant Hoisting Pattern

## Summary

Apply **Loop-Invariant Code Motion (LICM)** to Triton kernels: move computations that do **not** depend on the loop induction variable out of the loop, so each iteration performs only the minimal work that truly varies.

## Use When

- The kernel has a hot inner loop (often a K loop in GEMM-like kernels).
- Each loop iteration repeats substantial pointer math, mask construction, type casts, or shape bookkeeping.
- Profiling shows scalar/control work is disproportionately high relative to useful compute.
- You have a `report.txt` output from `extracted_bin_data` (or you have already extracted simulation data and are about to analyze it). Focus on its overall content section.
- `report.txt` overall `[Pipe Distribution]` shows high SCALAR instruction/cycle share while VECTOR or CUBE useful compute is not dominant.
- `report.txt` overall `[Key Ratios]` shows SCALAR much larger than VECTOR, suggesting bookkeeping is heavier than useful vector work.
- For matmul, reduction, or dot-like kernels, `report.txt` shows CUBE work is low or not sustained even though `tl.dot` should be the main work.
- `report.txt` overall `[Pipe Distribution]` shows MTE2/MTE3 are not the dominant buckets.
- `[SCALAR Instr Types]` or `[TRACE Events]` are dominated by `ADD`, `ADD_IMM`, `MUL`, `MADD`, `CMP`, `JUMPCMP`, `SIGNEXT`, or `ZEROEXT` around address generation, mask construction, casts, or bounds checks.

## Signals

### Code

- Inner loop recomputes expressions of the form:
  - `base(pid, offs) + delta(loop_var)`
  - e.g. `a_ptr + offs_m*stride_am + k*stride_ak`
- Masks are rebuilt each iteration even when parts are invariant:
  - e.g. `a_mask_m = offs_m < M` is invariant, but recomputed into `a_mask` each iter.
- A loop-body expression uses only `program_id`, shape/stride arguments, outer-axis offsets, or `tl.constexpr` values and does not depend on the current loop induction variable.
- In nested loops, an expression depends on an outer loop variable but not the current inner loop variable, so it can be hoisted one loop level instead of completely outside all loops.
- The loop repeatedly rebuilds the same base pointer, boundary mask, cast, broadcast, or `tl.arange`-derived tensor before combining it with a loop-varying delta.
- The candidate is pure address, mask, cast, or metadata computation; hoisting it does not move a side-effecting operation or a load whose value may change between iterations.
- Treat this pattern as a candidate only when at least one code signal above is confirmed; profile evidence alone is insufficient.

### IR

- Repeated arithmetic chains (`muli/addi/index_cast`) inside `scf.while` / `scf.for` bodies.
- Loop bodies contain repeated `subi/minsi/maxsi` patterns for bounds handling.

### Profile

- AIV scalar dominated by `LD_XD_XN_IMM`, `ST_XD_XN_IMM`, `ADD(_IMM)`, `CMP_IMM`.
- Timeline shows CUBE waiting on flags around the loop, while AIV performs control-heavy work.
- `report.txt` overall `[Pipe Distribution]` shows high SCALAR instruction/cycle share while VECTOR or CUBE useful compute is not dominant. This supports loop-invariant hoisting when the code has a hot loop with repeated pointer math, masks, casts, or bounds checks, because LICM removes scalar bookkeeping from every iteration without changing the math.
- `report.txt` overall `[Key Ratios]` shows SCALAR much larger than VECTOR, for example a large `SCALAR:VECTOR_instr` or `SCALAR:VECTOR_cycles` ratio. This matches LICM when the loop body spends more work preparing addresses, masks, or loop-local metadata than doing useful vector work, and some of that setup can be computed once outside the loop.
- For matmul, reduction, or dot-like kernels, `report.txt` shows CUBE work is low or not sustained even though `tl.dot` should dominate. This can indicate CUBE is being starved by scalar loop bookkeeping, especially repeated address generation or mask construction before each load/dot step.
- `report.txt` overall `[Pipe Distribution]` shows MTE2/MTE3 are not the dominant buckets. This matters because LICM targets scalar/control overhead inside the loop; if memory movement dominates, a memory-layout, block-pointer, or software-pipeline pattern is more likely to be the first lever.
- `[SCALAR Instr Types]` or `[TRACE Events]` are dominated by `ADD`, `ADD_IMM`, `MUL`, `MADD`, `CMP`, `CMP_IMM`, `JUMPCMP`, `SIGNEXT`, or `ZEROEXT` near address generation, mask construction, casts, or bounds checks. These are direct signatures of repeated scalar bookkeeping, and they strengthen the LICM diagnosis when the repeated expression can be split into an invariant base plus a loop-varying delta.
- Treat the `report.txt` evidence as a trigger only when it matches the code signal; the repeated work must be inside a hot loop and must have a loop-invariant component that can be hoisted safely.

## Optimization strategy

For any expression `E(loop_var)`:

1. Split it into **loop-invariant base** and **loop-varying delta**:
   - `E(loop_var) = BASE + DELTA(loop_var)`
2. Compute `BASE` once outside the loop.
3. Compute only `DELTA` inside the loop, and combine.

This pattern has several common specializations in Triton.

## Specialization A: Pointer address-generation hoisting (formerly “hoist-base-pointers”)

### Goal

Reduce per-iteration address-generation by hoisting invariant pointer bases.

### Before

```python
k = 0
while k < K:
    k_offs = k + offs_k
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak)
    b_ptrs = b_ptr + (k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    a = tl.load(a_ptrs, ...)
    b = tl.load(b_ptrs, ...)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

### After

```python
a_base = a_ptr + (offs_m[:, None] * stride_am)
b_base = b_ptr + (offs_n[None, :] * stride_bn)

k = 0
while k < K:
    k_offs = k + offs_k
    a_ptrs = a_base + (k_offs[None, :] * stride_ak)
    b_ptrs = b_base + (k_offs[:, None] * stride_bk)
    a = tl.load(a_ptrs, ...)
    b = tl.load(b_ptrs, ...)
    acc += tl.dot(a, b)
    k += BLOCK_K
```

## Specialization B: Mask / bounds hoisting (partial LICM)

### Goal

Hoist invariant parts of masks and bounds checks outside the loop.

### Example

- Invariant:
  - `a_mask_m = offs_m < M`
  - `b_mask_n = offs_n < N`
- Varying with `k_offs`:
  - `k_mask = k_offs < K`

Inside the loop build masks from precomputed invariants:

```python
a_mask_m = offs_m < M
b_mask_n = offs_n < N

k = 0
while k < K:
    k_offs = k + offs_k
    k_mask_row = k_offs[None, :] < K
    k_mask_col = k_offs[:, None] < K
    a_mask = a_mask_m[:, None] & k_mask_row
    b_mask = k_mask_col & b_mask_n[None, :]
    ...
```

## Performance impact expectations

- Lower AIV scalar/control overhead, especially on large-K loops.
- Cleaner loop bodies can improve backend scheduling and reduce flag/wait overhead.
- LICM is typically a **low-risk, incremental** optimization: does not change math, only where expressions are computed.

## Pitfalls / risks

- **Broadcast orientation mistakes**: `[:, None]` vs `[None, :]` must be preserved.
- **Over-hoisting**: do not hoist expressions that depend on `k_offs` or other loop-varying values.
- LICM does not eliminate transform costs (e.g. ND2NZ) by itself; treat layout issues as separate patterns.

## Avoid When

- The repeated work actually depends on the loop variable and cannot be split into an invariant base plus a varying delta.
- `report.txt` shows MTE2/MTE3 dominate the profile, or hotspot views point primarily to memory transfer, layout conversion, gather/scatter, or store shape rather than scalar loop bookkeeping.
- CUBE or VECTOR utilization is already high and scalar/control work is small; LICM is unlikely to be the primary lever.
- The main scalar cost comes from flattened coordinate decoding, modulo addressing, or scalar traps that need a structural rewrite first; prefer `scalar-latency-traps` or `block-pointer-dimensionality`.

## What To Verify After Applying

1. **Correctness**: compare against reference across boundary shapes (non-multiples of block sizes).
2. **Profiler**: reduced scalar instruction mix (`LD/ST/ADD/CMP`) and improved wall time.
3. **IR sanity**: fewer repeated arithmetic ops inside loop bodies (qualitative evidence).

## Related Patterns

- Complements **`compile-hint`**: after LICM, add alignment/contiguity hints.
- Complements **`software-pipeline`**: LICM simplifies loop bodies; pipeline overlaps remaining transfer/compute.
- Complements **`remove-implicit-transpose`**: layout fixes reduce transform work; LICM reduces residual loop control cost.
