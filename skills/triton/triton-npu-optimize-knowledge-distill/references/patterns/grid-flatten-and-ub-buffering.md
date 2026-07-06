---
priority: high
---

# Grid Flattening And UB Buffering Pattern

## Summary

Flatten logical work items onto physical cores and batch small row-wise memory transfers into wider UB stores to reduce launch overhead and improve per-core work density.

## Use When

- The logical grid is much larger than the physical AICore or VectorCore count.
- Work is partitioned by batch or sequence buckets with visible load imbalance.
- Each program processes many tiny rows after grid-to-physical-core mapping.
- Gather-like code has continuous destination rows but still stores one row at a time.
- Scatter-weight-gradient-like code has repeated row loads that can be batched from continuous source rows.
- A kernel computes per-element coordinates (integer division/modulo) to map linear offsets to a multidimensional output — restructure the grid to eliminate these by computing base addresses per outer group once, then copying contiguous inner elements.
- Inner elements per outer group are smaller than the target block size, causing inner blocks to degenerate into masked single-access iterations. When you apply the per-group 2D grid repair, the packed 1D dispatch is a required companion — the 2D grid kernel alone cannot handle narrow inner dimensions efficiently.
- A grid axis dimension grows into the hundreds or thousands, making each program's per-launch data volume too small for efficient DMA.

## Signals

- Code still decodes per-element coordinates via integer division/modulo inside the vectorized loop body.
- A per-group 2D grid kernel exists but no packed 1D dispatch path handles cases where `inner_elements < BLOCK_SIZE`.
- Sub-block inner loops (`BLOCK_SUB`) or grid outer-dimension floors are used to work around narrow inner elements instead of switching to a packed 1D dispatch. These workarounds do not fix the root cause — when inner_elements is very small, each inner program still does a single masked access per iteration.
- Grid is always 2D regardless of shape; there is no `if inner_elements >= BLOCK_SIZE` dispatch at the host call site.
- Only a packed 1D dispatch kernel exists with no 2D grid + outer-cap path for the large-inner case. When `inner_elements >= BLOCK_SIZE` the packed kernel falls back to `groups_per_prog = 1`, launching one program per outer group. For large `n_outer_groups` (≥ 1024) this produces many thin programs with low per-launch DMA volume. The 2D grid with outer cap at 1024 is still required as the dispatch target for `inner_elements >= BLOCK_SIZE`.

## Avoid When

- Do not use only the 2D per-group kernel when inner elements can be smaller than BLOCK_SIZE for some shapes. The 2D kernel degrades to single-element masked accesses per inner program when inner_elements < BLOCK_SIZE. You must add a packed 1D dispatch fallback.
- Do not try to compensate for narrow inner dimensions with sub-block loops (`BLOCK_SUB`) or outer grid floors alone. These keep the 2D grid structure intact and do not fix the underlying work-per-program dilution.
- Do not use a packed 1D dispatch kernel as the sole kernel for all inner-element sizes. The packed approach handles narrow inner elements well, but when `inner_elements >= BLOCK_SIZE` and `n_outer_groups` is large (≥ 1024), `groups_per_prog = 1` launches one program per outer group with low per-launch DMA volume. Add a host-side dispatch: 2D grid with outer cap ≤1024 for `inner_elements >= BLOCK_SIZE`, packed 1D with autotune for `inner_elements < BLOCK_SIZE`.
- Do not skip autotune on the packed dispatch path. Hand-picking one BLOCK_SIZE inevitably under-serves or over-serves differently shaped cases. Search [512, 1024, 2048, 4096] with shape-gated autotune keys.

## Repairs

### Flatten logical tasks

Treat all independent logical work items as one continuous stream and split that stream evenly across physical cores:

```python
pid = tl.program_id(0)
task_start = pid * TASKS_PER_CORE
offs = task_start + tl.arange(0, TASKS_PER_CORE)
mask = offs < TOTAL_TASKS
```

Compute `TASKS_PER_CORE` on the host and pass it as `tl.constexpr` when it controls tensor shapes or loop bounds. Prefer uniform per-core loop structure and masks over early returns that create uneven control flow.

### Map logical grid to physical grid

When `logical_tasks > num_cores`, launch `num_cores` programs and loop over logical tasks inside the kernel. Keep `NUM_TASKS` and `NUM_CORES` as explicit constants so the compiler can simplify loop bounds.

This helps only when each physical program has enough work to amortize the internal loop. If the original grid is near the physical core count, the extra loop can be overhead.

### Collapse a lightweight batch/head grid axis into the program

For chunked attention or recurrence kernels, a grid like `(NT, B * HV)` can create many small programs when each `(batch, head)` lane does modest work. If the per-lane work is independent and writes disjoint output regions, consider launching only over chunks and looping over `B * HV` inside each program:

```python
pid_t = tl.program_id(0)

for i_bh in range(BH):
    i_b = i_bh // HV
    i_h = i_bh % HV
    # Process one batch/head lane for this chunk.
```

Wrapper launch:

```python
kernel[(NT,)](..., BH=B * HV)
```

instead of:

```python
kernel[(NT, B * HV)](...)
```

This is a work-density variant of grid flattening. It is not a universal replacement for parallel batch/head execution. Use it when the original per-head programs are too thin and `BH` is small enough that serializing the axis does not underutilize the device.

### Discover core counts and choose grid by task kind

Use a best-effort runtime query before hardcoding one grid size:

```python
import torch

print(torch.npu.device_count())
device = torch.npu.current_device()
props = torch.npu.get_device_properties(device)
print(props)
```

If this query fails, if `torch.npu` is unavailable in the current environment, or if `props` does not expose explicit per-engine counts, fall back to:

- cube cores: `24`
- vector cores: `48`

Do not turn one observed physical-core count into a universal launch rule. Pick the initial flattened grid from the operator task kind:

- `cube`-like operators: start from the discovered Cube-core count and verify that the kernel really stays Cube-dominant.
- `vector`-like operators: start from the discovered Vector-core count and retest after any major tiling or buffering rewrite.
- `mix` operators: choose the starting point from the dominant bottleneck side, or test both cube-first and vector-first launch counts when the profile is ambiguous.

Keep `NUM_CORES` aligned with the chosen task kind rather than hardcoding one global constant across unrelated operators.

### UB aggregate writes

After physical-core mapping, if each program produces several rows with continuous destination addresses, stage a small group of rows in UB and issue a wider 2D store. Choose `SUB_BLOCK_SIZE` so the staged rows plus live compute tensors fit UB.

Do not use aggregate writes for genuinely scattered destinations.

### UB bulk reads

For row-wise reductions or gradient accumulation where several consecutive source rows are loaded one at a time, load a small 2D slab into UB and select rows from the slab. Keep a fallback path if row indices are not continuous enough for a bulk read to be correct.

### Per-group contiguous copy with multi-dimensional grid

When work is partitioned into outer groups (leading dims) and inner elements (trailing dims), use a 2D grid: axis 0 = outer groups, axis 1 = inner blocks. Compute the output base address once per outer group, then copy inner elements contiguously. This eliminates ALL per-element coordinate computation (division/modulo), which is prohibitively expensive on Ascend NPU even in vectorized form.

```python
pid_outer = tl.program_id(axis=0)
pid_inner = tl.program_id(axis=1)
num_outer_pids = tl.num_programs(axis=0)
num_inner_pids = tl.num_programs(axis=1)

for outer_gid in range(pid_outer, n_outer_groups, num_outer_pids):
    in_base = outer_gid * inner_elements
    out_base = outer_gid * out_outer_stride + concat_offset

    for block_start in range(
        pid_inner * BLOCK_SIZE, inner_elements, num_inner_pids * BLOCK_SIZE
    ):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < inner_elements
        values = tl.load(x_ptr + in_base + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + out_base + offsets, values, mask=mask)
```

The host computes `n_outer_groups` and `inner_elements` from the tensor shape and concat dimension. Grid total must stay ≤ 65535 on Ascend. Cap and adjust grid dimensions when the product would exceed this limit.

This pattern applies whenever you have an outer/inner work partition and the inner elements are laid out contiguously in memory. It replaces any kernel that maps a linear offset through coordinate decoding into a multidimensional output pointer.

### Cap grid dimensions to increase per-program work density

When a grid axis dimension is very large relative to the per-element data volume, each program handles very little data (< 4 KB per launch), causing launch overhead and narrow DMA transfers to dominate. Cap that grid axis (e.g., at 1024) so each program strides over multiple outer groups, increasing per-program byte volume.

```python
grid_outer = min(n_outer_groups, 1024)
grid_inner = min(triton.cdiv(inner_elements, BLOCK_SIZE), 65535)
# Keep total grid ≤ 65535 for Ascend
if grid_outer * grid_inner > 65535:
    grid_inner = 65535 // grid_outer
    if grid_inner < 1:
        grid_inner = 1
        grid_outer = 65535
```

Cap the outer dimension first since it controls per-program granularity. The inner dimension preserves intra-group parallelism. Verify the cap doesn't introduce regressions on small cases — if `n_outer_groups` is already small, a cap that forces single-program execution would destroy parallelism.

### Packed dispatch for narrow inner dimensions

When `inner_elements < BLOCK_SIZE`, the 2D grid's inner blocks degrade into single masked accesses per iteration. **This is a required companion to the per-group contiguous copy repair** — the 2D kernel cannot handle narrow inner dimensions efficiently, but the reverse is also true: the packed 1D kernel cannot handle large inner dimensions with large `n_outer_groups` efficiently because `groups_per_prog = 1` launches too many thin programs.

The correct dispatch is:
- `inner_elements >= BLOCK_SIZE`: use the 2D grid kernel with outer cap ≤1024.
- `inner_elements < BLOCK_SIZE`: use the packed 1D kernel with autotune.

Both kernels are required. Neither alone covers all regimes.

The packed 1D kernel uses a 1D grid where each program handles multiple outer groups. Each program's inner loop processes the entire inner dimension in BLOCK_SIZE-sized bursts with a mask. The mask naturally handles partial tails.

```python
# Host computes groups_per_prog:
groups_per_prog = max(1, BLOCK_SIZE // inner_elements)
grid = (triton.cdiv(n_outer_groups, groups_per_prog),)
grid = (min(grid[0], 65535),)

# Kernel uses 1D grid, inner loop over BLOCK_SIZE bursts:
pid = tl.program_id(axis=0)
num_pids = tl.num_programs(axis=0)

for outer_gid in range(pid * groups_per_prog, n_outer_groups, num_pids * groups_per_prog):
    # Process groups_per_prog outer groups in this iteration
    for g in range(groups_per_prog):
        gid = outer_gid + g
        if gid >= n_outer_groups:
            break
        in_base = gid * inner_elements
        out_base = gid * out_outer_stride + concat_offset

        for block_start in range(0, inner_elements, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < inner_elements
            values = tl.load(x_ptr + in_base + offsets, mask=mask, other=0.0)
            tl.store(out_ptr + out_base + offsets, values, mask=mask)
```

Combine with autotune to select the optimal BLOCK_SIZE per shape. A search space of [512, 1024, 2048, 4096] lets the runtime balance program count against DMA burst efficiency per case.

Set a minimum grid floor (e.g., 32) so very small `n_outer_groups` don't collapse to a single program when parallelism is still available.

## Dependencies

- Apply `grid-flatten-and-ub-buffering` only after the kernel's basic access semantics are clear.
- UB aggregate write and UB bulk read usually depend on a prior grid-to-physical-core or flattening rewrite that gives each program multiple rows.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` separately enough to explain the result.

## Risks

- Flattening can erase useful multidimensional locality if offsets are decoded poorly.
- Physical-grid loops can hurt small workloads.
- Serializing a batch/head axis can under-parallelize large `B * HV` or Cube-heavy workloads.
- Large compile-time `BH` values can increase code size or make one program too long.
- UB staging can exceed UB capacity or reduce occupancy.
- Gather and scatter have different address-continuity requirements; do not reuse aggregate-write logic for scattered stores.
- Grid outer cap: if `n_outer_groups` is already small, the cap can collapse parallelism and regress small cases. Add a floor (e.g., `grid_outer >= 32`) when `n_outer_groups >= 32`.
- Packed dispatch: fixed BLOCK_SIZE can under-subscribe or over-subscribe differently shaped cases. Prefer autotune over a hand-picked size.
- Per-group 2D grid: verify inner elements are truly contiguous in memory before applying this pattern. If inner elements are strided, the contiguous-load assumption fails silently.

## What To Verify After Applying

- Record the physical core assumption and how it is discovered or passed.
- Validate edge cases where `TOTAL_TASKS` is not divisible by `NUM_CORES`.
- Benchmark small and large shapes if the operator supports both.
- If the win depends on row continuity, add that condition to the round summary.
- For batch/head-in-program loops, test several `B * HV` regimes and verify outputs are disjoint across the serial loop.
- For per-group 2D grid: verify the inner loop's `out_base` computation is correct for all outer groups, including the final tensor in a concat sequence.
- For grid outer cap: confirm per-program byte volume increased, and verify no regression on small cases where the cap might reduce parallelism below usable levels.
- For packed dispatch: verify that the mask `offsets < inner_elements` correctly handles the tail (inner_elements may not be a multiple of BLOCK_SIZE) and that `groups_per_prog` computation doesn't overflow when `inner_elements` is very small.

## Related Patterns

- Complements `program-multiple-rows`: that pattern widens row-wise work inside a program, while this pattern flattens logical work onto physical cores and batches memory movement inside each core.
- Combine with `autotune` only after the structural rewrite is correct; tune `TASKS_PER_CORE`, `BLOCK`, and `SUB_BLOCK_SIZE` with enough separation to explain the result.
- When the root cause is per-element integer division/modulo, check `scalar-latency-traps` first for scalar-broadcast repairs. If those don't apply (each element maps to a different output position), use the per-group contiguous copy repair in this card instead.
- `flat-index-decode-tiling` is the inverse: replace scalar-heavy 1D flat-index traversal with layout-aware multidimensional tiles. This card's per-group repair is a specific instance of that reversal for copy/scatter patterns.
- `merge-repeated-kernel-launches` — after merging launches, apply grid flattening for further overhead reduction
- `adaptive-launch-element-wise` — the chunked dispatch variant uses the same per-program inner loop concept
- `sequential-kernel-fusion` — after fusing kernels, apply grid flattening to the fused kernel
