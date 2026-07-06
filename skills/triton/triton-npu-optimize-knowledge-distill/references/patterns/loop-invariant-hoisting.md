# Loop-Invariant Hoisting Pattern

## Summary

Apply **Loop-Invariant Code Motion (LICM)** to Triton kernels: move computations that do **not** depend on the loop induction variable out of the loop, so each iteration performs only the minimal work that truly varies.

## Use When

- The kernel has a hot inner loop (often a K loop in GEMM-like kernels, or a T-chunk loop in sequence kernels).
- Each loop iteration repeats substantial pointer math, mask construction, type casts, `tl.arange` tensor creation, or shape bookkeeping.
- Loop iterations re-load small invariant data arrays (weights, parameters, lookup tables) whose values never change across iterations.
- Profiling shows scalar/control work is disproportionately high relative to useful compute.

## Signals

### Code

- Inner loop recomputes expressions of the form:
  - `base(pid, offs) + delta(loop_var)`
  - e.g. `a_ptr + offs_m*stride_am + k*stride_ak`
- Masks are rebuilt each iteration even when parts are invariant:
  - e.g. `a_mask_m = offs_m < M` is invariant, but recomputed into `a_mask` each iter.
- A weight or parameter array is loaded via `tl.load(...)` inside a row-task loop body, meaning it is reloaded on every row-task iteration even though it does not depend on the row-task induction variable. This happens in two forms: (a) inside inner column loops nested within an outer row-task loop (two-pass path), e.g. `W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask)` inside `for col_offset` nested in `for row_task_id`; (b) directly inside a row-task loop body with no nested column loop (single-tile fused path), e.g. `W_chunk = tl.load(W_ptr + cols_off, mask=cols_mask)` inside a `for row_task_id` loop where the whole row fits in one column tile. Both forms waste bandwidth — hoist the load before the row-task loop.
- Column offset arrays (`cols_off = col_offset + tl.arange(...)`) and column masks (`cols_mask = cols_off < n_cols`) are rebuilt inside the inner column loop on every iteration. When the column extent is the same across all row tasks, these are loop-invariant and can be computed once before the outer loop.

### IR

- Repeated arithmetic chains (`muli/addi/index_cast`) inside `scf.while` / `scf.for` bodies.
- Loop bodies contain repeated `subi/minsi/maxsi` patterns for bounds handling.

### Profile

- AIV scalar dominated by `LD_XD_XN_IMM`, `ST_XD_XN_IMM`, `ADD(_IMM)`, `CMP_IMM`.
- Timeline shows CUBE waiting on flags around the loop, while AIV performs control-heavy work.

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

## Specialization C: Closed-form window divisor (pooling)

### Goal

Replace **per-tap counting** inside a fixed-kernel window loop with an **O(1) algebraic divisor** computed from the same clip bounds used for loads.

### Before

```python
for kd in range(KERNEL_D):
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            if window_mask:
                acc += load(...)
                valid_count += 1
divisor = valid_count
```

### After

```python
tstart_d, ..., d_len, h_len, w_len = clip_window(...)
divisor = closed_form_volume(start_*, tend_*, COUNT_INCLUDE_PAD, ...)
for kd in range(KERNEL_D):
    ti = tstart_d + kd
    if kd < d_len:
        ...
        acc += load(x_base + ti * stride_d + ...)
```

This is not mere hoisting: the counting loop is **eliminated**, not moved. See `simt-clip-window-closed-reduction` for padding semantics and Triton/Ascend details.

## Specialization D: Pre-load small loop-invariant data arrays

### Goal

When an inner loop re-loads a small constant data array (weights, filter coefficients, lookup tables) on every iteration, load it once outside the loop and reuse across all iterations.

### When this applies

- The data is small enough to fit in registers (e.g. D_CHK × width floats, typically < 1KB).
- The data does not change across loop iterations.
- The loop trip count is large enough that the redundant loads accumulate.
- The kernel has a loop over row tasks (`for row_task_id in ...`). A parameter or weight array (e.g., `W_chunk`) is loaded inside the loop body but does not depend on `row_task_id` — the same load executes redundantly on every iteration. This applies both when the row loop has inner column-iteration loops (two-pass path) and when the row fits in one tile and the loop body has no nested column loop (single-tile fused path).

### Before

```python
for ti in range(NUM_T_CHK):
    for owi in range(width):
        # Same weight data loaded NUM_T_CHK × width times
        w_col = tl.load(weight_ptr + off_d * width + owi).to(tl.float32)[:, None]
        out_block += x_win * w_col
```

### After

```python
# Load all weight columns once per program
w_col_0 = tl.load(weight_ptr + off_d * width + 0).to(tl.float32)[:, None]
w_col_1 = tl.load(weight_ptr + off_d * width + 1).to(tl.float32)[:, None]

for ti in range(NUM_T_CHK):
    out_block += x_win_0 * w_col_0
    out_block += x_win_1 * w_col_1
```

The saving is proportional to `(NUM_T_CHK - 1) × width` redundant loads and their associated address computations. Combining pre-loaded columns with manual loop unrolling eliminates both the data reloads and the loop control overhead.

### Risk

- Pre-loading expands register pressure. Verify the loaded data plus live accumulators fit within register budget.
- For data larger than register capacity, consider UB staging instead.

## Specialization E: Precompute loop-invariant arithmetic expressions

### Goal

Hoist elementwise arithmetic that does not depend on the inner loop induction variable, so the loop body does only the work that varies per iteration.

### When this applies

- An inner loop iterates over a parameter (e.g., GQA groups) and each iteration recomputes the same elementwise expressions on tensors that are invariant across iterations.
- The invariant tensors are loaded once per tile before the loop, but arithmetic on them (diffs, scaling, type conversions) is repeated each iteration.
- The loop trip count is >1, so the wasted arithmetic scales with iteration count.

### Before

```python
for g in tl.static_range(G):
    q = tl.load(q_ptr + g_offs, ...)
    k_diff = k_ul_f32 - k_ur_f32   # recomputed G times per tile
    v_diff = v_ul_f32 - v_ur_f32   # recomputed G times per tile
    diff = tl.sum(q * k_diff, axis=1) * SCALE
    attn = 1.0 / (1.0 + tl.math.exp(-diff))
    o = v_ur_f32 + attn[:, None] * v_diff
    tl.store(o_ptr + g_offs, o, ...)
```

### After

```python
k_diff = k_ul_f32 - k_ur_f32   # computed once, outside loop
v_diff = v_ul_f32 - v_ur_f32   # computed once, outside loop
for g in tl.static_range(G):
    q = tl.load(q_ptr + g_offs, ...)
    diff = tl.sum(q * k_diff, axis=1) * SCALE
    attn = 1.0 / (1.0 + tl.math.exp(-diff))
    o = v_ur_f32 + attn[:, None] * v_diff
    tl.store(o_ptr + g_offs, o, ...)
```

The saving is proportional to `(G - 1) × cost_of_diff_ops` per tile. For larger G, the savings scale linearly. This applies to any elementwise arithmetic on loop-invariant tensors: diffs, pre-scaling, pre-conversion to fp32, mask construction from invariant bounds.

### Risk

- Hoisting increases live register count across loop iterations. Verify the hoisted values plus per-iteration live state fit within register budget.
- Broadcast orientation (`[:, None]` vs `[None, :]`) must be preserved when moving expressions outside the loop.
- Do not hoist if the invariant expression is conditionally used (e.g., behind a mask that varies per iteration).

## Performance impact expectations

- Lower AIV scalar/control overhead, especially on large-K loops.
- Cleaner loop bodies can improve backend scheduling and reduce flag/wait overhead.
- LICM is typically a **low-risk, incremental** optimization: does not change math, only where expressions are computed.

## Pitfalls / risks

- **Broadcast orientation mistakes**: `[:, None]` vs `[None, :]` must be preserved.
- **Over-hoisting**: do not hoist expressions that depend on `k_offs` or other loop-varying values.
- LICM does not eliminate transform costs (e.g. ND2NZ) by itself; treat layout issues as separate patterns.

## What To Verify After Applying

1. **Correctness**: compare against reference across boundary shapes (non-multiples of block sizes).
2. **Profiler**: reduced scalar instruction mix (`LD/ST/ADD/CMP`) and improved wall time.
3. **IR sanity**: fewer repeated arithmetic ops inside loop bodies (qualitative evidence).

## Related Patterns

- Complements **`compile-hint`**: after LICM, add alignment/contiguity hints.
- Complements **`software-pipeline`**: LICM simplifies loop bodies; pipeline overlaps remaining transfer/compute.
- Complements **`remove-implicit-transpose`**: layout fixes reduce transform work; LICM reduces residual loop control cost.
- `simt-clip-window-closed-reduction` — algebraic normalizer replacement plus clip-window loads for fixed-window reduces
- `block-ptr-advance-reuse` — advance reduces descriptor creation cost; LICM reduces other loop overhead
- `padded_row_col_copy` — uses LICM-style hoisting of per-row base addresses before the column loop
