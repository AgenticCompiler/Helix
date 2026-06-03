# Program Multiple Rows Pattern

## Summary

Amortize per-program fixed costs and improve vector-friendly batching for **row-reduction or row-wise fused kernels** by mapping **multiple rows** to one Triton `program_id` via `BLOCK_M > 1`, instead of one row per program.

## Use When

- The kernel is **naturally row-wise**: each output row depends mainly on one row of input (e.g. row-wise LogSumExp, row norms, row softmax statistics).
- Profiling or timeline views suggest **high scalar/control overhead**, **under-filled vector work per program**, or **many tiny programs** relative to problem size `B` (batch / number of rows).
- The row-wise math already uses **tile loops along `N`** (`BLOCK_N`); increasing **`BLOCK_M`** does not force an extra full pass over global memory if you keep a **single streaming pass** over `N` per program.
- `report.txt` overall `[Pipe Distribution]` shows high SCALAR/control share while VECTOR work is low or thin, suggesting fixed per-program setup is large compared with useful row work.
- `report.txt` overall `[Key Ratios]` shows a high SCALAR:VECTOR ratio, and code inspection shows `program_id(0)` maps one-to-one to rows.
- `report.txt` overall `[VECTOR Unit]` shows low utilization or a small amount of vector work per program, consistent with narrow row tiles such as `(1, BLOCK_N)`.
- `report.txt` `[WAIT_FLAG / BAR Sync]`, `[Pipeline Flows]`, or timeline views show frequent short-program wait/barrier/control patterns rather than long sustained vector work.
- `report.txt` `[Pipe Distribution Over Each Core]` shows similar SCALAR-heavy / VECTOR-thin behavior across cores, suggesting the issue is global program granularity rather than a single-core anomaly.

## Signals

### Code

- `program_id(0)` indexes **rows 1:1** (`pid_m` is the row index), and the inner loop only tiles **`N`**.
- Scalar helpers (`program_id`, pointer arithmetic per row) run once **per row**; vector units see **narrow** tensors (e.g. `(1, BLOCK_N)` loads).

### Profile

- **`aiv_scalar_ratio`** or scalar-related time is **disproportionately high** compared to useful vector math, for workloads where `B` is large enough that vector throughput should dominate.
- **`op_statistic`** (per-kernel): **Avg** latency improves when the same logical work uses **fewer launches** (compare with care: **Count** and input shapes must be comparable across runs).
- If **`aiv_mte2_ratio`** is **not** the sole dominant bucket, pure “double-buffer the loads” may be the wrong first lever; **program batching** can still help by making each program’s inner loop **wider** along rows.
- Frequent **barrier / wait** patterns tied to **many short programs** or **thin** vector blocks.
- **Note:** High **`BAR`** cycle counts alone are **not** a success metric; correlate with **wall time**, **op_statistic Avg**, and correctness.
- `report.txt` overall `[Pipe Distribution]` shows high SCALAR/control share while VECTOR work is low or thin. This supports `program-multiple-rows` when code inspection shows one program handles only one row, because each row pays fixed scalar setup cost (`program_id`, row pointer arithmetic, masks) before doing a small amount of vector work; batching rows with `BLOCK_M > 1` amortizes that setup.
- `report.txt` overall `[Key Ratios]` shows a high SCALAR:VECTOR ratio, and the code maps `program_id(0)` one-to-one to rows. This matches `program-multiple-rows` when the useful math is row-wise and vector-friendly but each program is too narrow, because widening from `(1, BLOCK_N)` to `(BLOCK_M, BLOCK_N)` increases useful work per program without changing row independence.
- `report.txt` overall `[VECTOR Unit]` shows low utilization or only a small amount of vector work per program. This suggests the vector pipe is not saturated by the current row tile, which is the failure mode `BLOCK_M > 1` tries to fix by giving each program multiple independent rows to process together.
- `report.txt` `[WAIT_FLAG / BAR Sync]`, `[Pipeline Flows]`, or timeline views show frequent short-program wait/barrier/control patterns rather than long sustained vector work. This strengthens the diagnosis when the kernel launches many row programs, because the waits are plausibly tied to program granularity and thin row blocks rather than a single long memory or compute phase.
- `report.txt` `[Pipe Distribution Over Each Core]` shows similar SCALAR-heavy / VECTOR-thin behavior across cores. This points to a systematic granularity issue in the kernel mapping, not a one-core anomaly; if all cores run many similarly thin row programs, batching multiple rows per program is a plausible structural fix.
- Treat the `report.txt` evidence as a trigger only when it matches the code signal: `program_id(0)` or an equivalent grid axis maps 1:1 to rows, rows are independent, and increasing `BLOCK_M` preserves a single streaming pass along `N`.

## Implementation sketch (Triton)

1. Add **`BLOCK_M: tl.constexpr`** and treat **`pid_m`** as a **block of rows**:
   - `rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)`
   - `row_mask = rows < B`
2. Build **2D tiles**: `vals` shape **`(BLOCK_M, BLOCK_N)`**; row-wise reductions use **`axis=1`** (max, sum) so running state **`m`, `s`** are **`(BLOCK_M,)`**.
3. Adjust **grid**: `grid = (triton.cdiv(B, BLOCK_M),)`.
4. **Stores**: one scalar per row → `tl.store(y_ptr + rows * stride_ym, x, mask=row_mask)`.
5. **Tune `BLOCK_M`**: start with a small power of two (e.g. 4–16). Too large **`BLOCK_M * BLOCK_N`** may hurt UB/register pressure; validate with benchmark + profile.

### Example reference (row-wise LSE + fused activations)

See the operator workspace pattern: row pointers `row_ptrs = x_ptr + rows[:, None] * stride_xm`, masked load to `(BLOCK_M, BLOCK_N)`, streaming LSE with `tl.max` / `tl.sum` on `axis=1`, then fused elementwise ops and masked store.

## Avoid When

- **Second full pass** over `x` for the same row (e.g. two-pass LSE) usually **increases global reads**; msprof often shows **more MTE / wait** unless the algorithm truly requires it. Prefer **single-pass streaming LSE** when numerically stable.
- **Ping-pong / multibuffer** without evidence of **MTE–vector overlap** can add **sync and UB** cost; treat as a **separate hypothesis** to validate.
- Do not conclude from **one** metric (e.g. `BAR` cycles) without **end-to-end** timing and comparable workload.

- Rows are not independent, or batching multiple rows would introduce cross-row dependencies, atomics, ordering constraints, or extra synchronization.
- The current program already processes multiple rows or a sufficiently wide 2D tile, so `BLOCK_M > 1` would mostly increase register/UB pressure instead of amortizing fixed setup.
- `report.txt` shows MTE2/MTE3, gather/scatter, layout conversion, or UB conflicts dominate the profile; in that case memory/layout/UB patterns are a better first lever than row batching.
- `report.txt` shows VECTOR/CUBE work is already well utilized and scalar/control work is small, so program granularity is unlikely to be the main bottleneck.
- The only evidence is frequent `BAR` or wait activity, but wall time, `op_statistic Avg`, and comparable shapes do not improve; do not treat synchronization counters alone as a success or selection metric.

## What To Verify After Applying

1. **Correctness**: same dtypes, masks for `rows >= B`, and numerically stable reductions (e.g. LSE max-shift) unchanged in meaning.
2. **Benchmark**: compare **mean / geomean** with the same harness; use project **`compare-perf`** flow when available—avoid hand-computed speedups from raw logs.
3. **Profiler**: compare **`op_statistic` Avg** for the same op; note **Count** and tensor shapes. Optionally re-check **`op_summary`** vector/scalar/MTE mix.
4. **Sanity**: if `B` is tiny, **`BLOCK_M > 1`** may help little; if `B` is huge, launching **`cdiv(B, BLOCK_M)`** programs should visibly reduce launch/program overhead.

## Related Patterns

- Complements **`parallel`**: `BLOCK_M` widens work **within** one program; `tl.parallel` splits **independent** subgraphs across vector cores—orthogonal when dependencies allow.
- Differs from **`software-pipeline`**: multibuffer targets **load/compute overlap** along a tile loop; **`program-multiple-rows`** targets **program granularity** and **row batching**.
