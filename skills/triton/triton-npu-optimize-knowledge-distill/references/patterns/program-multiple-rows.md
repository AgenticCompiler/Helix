# Program Multiple Rows Pattern

## Summary

Amortize per-program fixed costs and improve vector-friendly batching for **row-reduction or row-wise fused kernels** by mapping **multiple rows** to one Triton `program_id` via `BLOCK_M > 1`, instead of one row per program.

## Use When

- The kernel is **naturally row-wise**: each output row depends mainly on one row of input (e.g. row-wise LogSumExp, row norms, row softmax statistics).
- Profiling or timeline views suggest **high scalar/control overhead**, **under-filled vector work per program**, or **many tiny programs** relative to problem size `B` (batch / number of rows).
- The row-wise math already uses **tile loops along `N`** (`BLOCK_N`); increasing **`BLOCK_M`** does not force an extra full pass over global memory if you keep a **single streaming pass** over `N` per program.
- The kernel processes rows one at a time with 1D per-row vector access and `program_id` indexes a single row. Use `ROWS_PER_PROGRAM` with a `tl.static_range` loop to batch multiple rows per program, amortizing launch overhead.
- The row count `B` is large enough that `cdiv(B, BLOCK_M)` still provides enough programs to keep all NPU cores busy. When `B` is small and `BLOCK_M` is aggressive, the reduced program count destroys parallelism.

## Signals

### Code

- `program_id(0)` indexes **rows 1:1** (`pid_m` is the row index), and the inner loop only tiles **`N`**.
- Scalar helpers (`program_id`, pointer arithmetic per row) run once **per row**; vector units see **narrow** tensors (e.g. `(1, BLOCK_N)` loads).
- A `TOKEN_BLOCK_SIZE_TABLE` or similar lookup table maps column sizes to BLOCK_M values, but the values are small (e.g., single-digit for columns ≤ 1024, ≤ 32 for columns ≤ 512). These are baseline/conservative defaults that leave substantial UB budget unused. Push every entry higher until either the UB budget formula says stop or a benchmark step regresses.
- `ROWS_PER_PROGRAM=1` with no scaling: each program processes exactly one row. When BLOCK_SIZE is small (e.g. ≤ 128), per-program launch overhead dominates useful work. Scaling ROWS_PER_PROGRAM inversely with BLOCK_SIZE amortizes this overhead.

### Profile

- **`aiv_scalar_ratio`** or scalar-related time is **disproportionately high** compared to useful vector math, for workloads where `B` is large enough that vector throughput should dominate.
- **`op_statistic`** (per-kernel): **Avg** latency improves when the same logical work uses **fewer launches** (compare with care: **Count** and input shapes must be comparable across runs).
- If **`aiv_mte2_ratio`** is **not** the sole dominant bucket, pure “double-buffer the loads” may be the wrong first lever; **program batching** can still help by making each program’s inner loop **wider** along rows.
- Frequent **barrier / wait** patterns tied to **many short programs** or **thin** vector blocks.
- When reducing BLOCK_N to allow larger BLOCK_M, the column iteration count increases. The net win depends on whether the row-task reduction is larger than the column-iteration increase. Profile both tile paths at the same shape and compare total execution time — higher MTE2 utilization alone does not guarantee faster total time.
- **Regression on small-B cases with large BLOCK_M**: when program count drops below what fills all cores, per-program latency increase outweighs the amortization gain. Check small-B benchmarks separately when increasing BLOCK_M.
- **Note:** High **`BAR`** cycle counts alone are **not** a success metric; correlate with **wall time**, **op_statistic Avg**, and correctness.

## Implementation sketch (Triton)

1. Add **`BLOCK_M: tl.constexpr`** and treat **`pid_m`** as a **block of rows**:
   - `rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`
   - `row_mask = rows < B`
2. Build **2D tiles**: `vals` shape **`(BLOCK_M, BLOCK_N)`**; row-wise reductions use **`axis=1`** (max, sum) so running state **`m`, `s`** are **`(BLOCK_M,)`**.
3. Adjust **grid**: `grid = (triton.cdiv(B, BLOCK_M),)`.
4. **Stores**: one scalar per row → `tl.store(y_ptr + rows * stride_ym, x, mask=row_mask)`.
5. **Tune `BLOCK_M`**: start with a small power of two (e.g. 4–16). Too large **`BLOCK_M * BLOCK_N`** may hurt UB/register pressure; validate with benchmark + profile.

6. **Gate aggressive BLOCK_M by row count**: When BLOCK_M ≥ 8, verify `cdiv(B, BLOCK_M)` still produces enough programs to keep all cores busy. If B is small (e.g. 128 rows with BLOCK_M=8 gives only 16 programs), fall back to a smaller BLOCK_M. The optimal BLOCK_M depends on both UB budget and row count — large B can tolerate larger BLOCK_M because there are enough programs to fill cores. A dispatch table like the following is typical:

   ```
   if B >= 1024 and BLOCK_N <= 4096:  BLOCK_M = 8   # abundant parallelism, UB-safe
   elif BLOCK_N <= 8192:              BLOCK_M = 4   # good parallelism
   else:                              BLOCK_M = 2   # UB-tight, conservative
   ```

   The 1024-row threshold ensures `cdiv(B, 8)` ≥ 128 programs — enough to saturate even large core counts.

   **When `B < BLOCK_M` (fewer rows than the block size):** reduce `BLOCK_M` to the smallest power of two that can cover all rows:

   ```
   if B < BLOCK_M:
       BLOCK_M = max(1, 2 ** ((B - 1).bit_length()))
   ```

   This avoids launching programs where most lanes are idle because the row count is smaller than the tile height. Without this reduction, a shape with 3 rows and BLOCK_M=64 launches one program where 61 of 64 lanes are wasted.

7. **Select `BLOCK_M` from UB budget**: compute the max safe M from a precise formula before falling back to trial-and-error.

   ```
   max_M = UB_BYTES // (element_size * BLOCK_SIZE_N * num_tile_buffers)
   ```

   Where `num_tile_buffers` counts all live tiles each program must hold simultaneously:

   - **Two-pass paths** (store intermediate + reload): use a conservative count (e.g., 14 for 2-byte dtypes, 10 for 4-byte). The intermediate store buffer and reload tile both consume UB.
   - **Fused single-pass paths** (no intermediate store): use a lower count (e.g., 10 for 2-byte dtypes). Fewer live tiles means more UB budget for BLOCK_M.
   - **Per-dtype gating**: apply the formula per dtype with the correct element size. Smaller dtypes allow larger M at the same BLOCK_N.
   - Round M down to a multiple of the launch granularity (e.g., 8) to avoid partial wavefront waste.
   - **Dtype-aware overall budget**: Scale the total tile element budget by element size in addition to the per-element factor in the formula. For a fixed UB capacity, a 4-byte dtype (fp32) fits half as many total elements as a 2-byte dtype (fp16/bf16). Halve the effective budget: `budget = base_budget // max(1, elem_size // 2)`. Without this scaling, fp32 tiles that are safe for fp16 can silently overflow UB.

   Prefer this formula-driven approach over blind trial-and-error because the UB budget is a hard constraint — exceeding it causes silent performance cliffs, not gradual degradation.

   - **Recompute inherited lookup tables**: If the code already has a `TOKEN_BLOCK_SIZE_TABLE` mapping column dimensions to BLOCK_M values, do not assume the values are already optimal. The baseline values are almost always conservative and leave unused UB capacity. Compute the UB-budget-maximizing BLOCK_M for each table entry; replace any value that is smaller than the formula allows. The table structure is retained — only the values are pushed upward. After updating, add a per-shape override (e.g., a dtype-gated special case for small columns like n_cols=256 with fp16/bf16) if the formula produces a BLOCK_M larger than any single table entry.

8. **Select BLOCK_N from shape aspect ratio (tall vs wide paths)**: When the row count greatly exceeds the column count, reducing BLOCK_N (e.g. from 2048 to 1024) frees UB budget for a proportionally larger BLOCK_M. The larger BLOCK_M reduces program count. Apply a strict threshold: only use the narrow-width path when `n_rows > 1.5 * n_cols`. Empirically, this is the boundary where row-task reduction outweighs doubled column iteration overhead.

   Additionally, apply a **tall-to-wide fallback**: only use the narrow-width path when it enables a strictly larger BLOCK_M than the wide path. When both paths produce the same BLOCK_M (common for fp32 where the dtype-aware budget constrains both equally), the wider BLOCK_N is strictly better — fewer column iterations for the same rows per tile.

   ```python
   wide_m = compute_block_size_m(wide_n, n_rows, grid_size, budget)
   if n_rows > TALL_THRESHOLD * n_cols:
       tall_m = compute_block_size_m(narrow_n, n_rows, grid_size, budget)
       if tall_m > wide_m:
           return narrow_n, tall_m
   return wide_n, wide_m
   ```

9. **Adaptive grid sizing for row-tiled kernels**: The baseline grid often uses a flat `grid = (num_cores,)`. Replace with a three-tier grid based on the number of row tasks to prevent under-subscription and over-subscription:

   ```python
   num_row_tasks = (n_rows + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
   if num_row_tasks > num_cores * 16:
       num_programs = num_cores * 3
   elif num_row_tasks > num_cores * 8:
       num_programs = num_cores * 2
   else:
       num_programs = min(num_cores, num_row_tasks) if num_row_tasks < num_cores else num_cores
   grid = (num_programs,)
   ```

   For row-tiled kernels, each program processes row tasks in a strided loop (`for row_task_id in range(pid, num_row_tasks, grid_size)`). When `num_row_tasks` greatly exceeds `num_cores`, launching extra programs (2x–3x the core count) gives the hardware scheduler more work items to distribute, improving tail latency. When `num_row_tasks` is small, cap the grid at `num_row_tasks` to avoid launching idle programs.

### Implementation sketch: 1D row-loop variant

When each row is processed as a 1D vector (no column tiling), batch rows with `tl.static_range`:

```python
@triton.jit
def kernel(x_ptr, out_ptr, num_rows, dim,
           ROWS_PER_PROGRAM: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * ROWS_PER_PROGRAM
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    row_base = row_start * dim
    for r in tl.static_range(ROWS_PER_PROGRAM):
        row = row_start + r
        if row < num_rows:
            vals = tl.load(x_ptr + row_base + offs, mask=mask, other=0.0)
            result = compute(vals)
            tl.store(out_ptr + row_base + offs, result, mask=mask)
            row_base += dim
```

Compute `ROWS_PER_PROGRAM` on the host to keep grid within hardware limits:

```python
_MAX_COREDIM = 65535  # Ascend NPU grid dimension limit
def _compute_rows_per_program(num_rows):
    rows = 1
    while (num_rows + rows - 1) // rows > _MAX_COREDIM:
        rows += 1
    return rows
```

This variant is appropriate when the kernel has no inner column-tiling loop — each row fits in one vector tile. The `tl.static_range` unrolls at compile time, so there is zero loop control overhead in the generated code.

### RPP scaling by block size

When `BLOCK_SIZE` is small (narrow rows, e.g. half_dim ≤ 128), each program does very little work per row. Per-program launch overhead dominates. Scale `ROWS_PER_PROGRAM` beyond the minimum needed for grid-limit compliance:

```
if BLOCK_SIZE <= 64:    RPP *= up to 4x (cap at 8)
elif BLOCK_SIZE <= 128: RPP *= up to 2x (cap at 8)
else:                   no scaling (BLOCK_SIZE already provides enough work)
```

Maintain a minimum grid (e.g. 128 programs) so parallelism is not destroyed. This scaling amortizes launch overhead for narrow rows where each program would otherwise process just 32-64 elements per row. The cap prevents UB overflow from holding too many rows' intermediate state.

### Example reference (row-wise LSE + fused activations)

See the operator workspace pattern: row pointers `row_ptrs = x_ptr + rows[:, None] * stride_xm`, masked load to `(BLOCK_M, BLOCK_N)`, streaming LSE with `tl.max` / `tl.sum` on `axis=1`, then fused elementwise ops and masked store.

## Avoid When

- **Second full pass** over `x` for the same row (e.g. two-pass LSE) usually **increases global reads**; msprof often shows **more MTE / wait** unless the algorithm truly requires it. Prefer **single-pass streaming LSE** when numerically stable.
- **Ping-pong / multibuffer** without evidence of **MTE–vector overlap** can add **sync and UB** cost; treat as a **separate hypothesis** to validate.
- Do not conclude from **one** metric (e.g. `BAR` cycles) without **end-to-end** timing and comparable workload.
- **Row count too small for BLOCK_M**: When `cdiv(B, BLOCK_M)` produces too few programs to fill all cores, parallelism loss outweighs amortization gains. Apply aggressive BLOCK_M (≥ 8) only when `B` is large enough to sustain occupancy. A program count below `2 × core_count` is a strong anti-signal.
- **Narrow BLOCK_N with same BLOCK_M as wide path**: When reducing BLOCK_N does not enable a strictly larger BLOCK_M (e.g., dtype-aware budget constrains both paths to the same M), the narrow path doubles column iterations with no row-task reduction. Fall back to the wider BLOCK_N.

## What To Verify After Applying

1. **Correctness**: same dtypes, masks for `rows >= B`, and numerically stable reductions (e.g. LSE max-shift) unchanged in meaning.
2. **Benchmark**: compare **mean / geomean** with the same harness; use project **`compare-perf`** flow when available—avoid hand-computed speedups from raw logs.
3. **Profiler**: compare **`op_statistic` Avg** for the same op; note **Count** and tensor shapes. Optionally re-check **`op_summary`** vector/scalar/MTE mix.
4. **Parallelism check**: verify `cdiv(B, BLOCK_M)` still yields enough programs to fill all NPU cores. If program count drops too low (e.g., below `2 × core_count`), aggressive BLOCK_M can regress despite correct UB sizing. Check both small-B and large-B cases.

## Related Patterns

- Complements **`parallel`**: `BLOCK_M` widens work **within** one program; `tl.parallel` splits **independent** subgraphs across vector cores—orthogonal when dependencies allow.
- Differs from **`software-pipeline`**: multibuffer targets **load/compute overlap** along a tile loop; **`program-multiple-rows`** targets **program granularity** and **row batching**.
- `fuse-element-wise-intermediates-into-read-once-kernel` — uses BC batching (a form of program-multiple-rows) to reduce grid size
- `intra-kernel-pass-fusion` — fused per-row path makes larger BLOCK_M safer
- `sequential-kernel-fusion` — fused kernel can also process multiple rows per program
- `padded_row_col_copy` — also uses multi-row tiling for copy kernels
- `algebraic-optimization` — apply after math structure is stable
- `ub-bounded-column-block-size-maximization` — optimizes the column dimension; apply program-multiple-rows first to widen BLOCK_M, then tune BLOCK_N
