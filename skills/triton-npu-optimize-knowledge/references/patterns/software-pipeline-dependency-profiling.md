---
priority: high
---

# Software Pipeline Dependency Profiling Pattern

## Summary

Use this pattern when `extracted_bin_data/report.txt` suggests transfer and compute are weakly overlapped, the kernel contains `tl.load`, and the load latency can plausibly be hidden behind vector/cube compute. Constructing a `for` loop or steady-state loop around regular `tl.load` work can enable compiler prefetch, improve pipeline parallelism, and then be tuned with `num_stages` or manual prefetch when needed.

## Use When

- `extracted_bin_data/report.txt` is available.
- The `[Pipe Overlap Ratio]` section shows a software-pipeline signal: very low `%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE))` and very low `%((VECTOR+CUBE)&MTE2/MTE2)`.
- In `[Pipe Distribution]`, compute `compute_cycles%` from both `VECTOR cycles%` and `CUBE cycles%` when both rows exist; if only one of the two rows exists, use the available row. Prefer this pattern when `compute_cycles%` and `MTE2 cycles%` are relatively balanced, such as each being at least about one-third of the other.
- `SCALAR cycles%` is not the dominant share in `[Pipe Distribution]`; if scalar cycles dominate, prefer scalar-related optimization first.
- The kernel contains `tl.load`.
- The `tl.load` path is regular enough that constructing a `for` loop or steady-state loop can enable compiler prefetch and improve performance through pipeline parallelism.

## Avoid When

- In `[Pipe Overlap Ratio]`, `%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE))` or `%((VECTOR+CUBE)&MTE2/MTE2)` is already high enough that load and compute overlap well.
- In `[Pipe Distribution]`, the manually computed `compute_cycles%` and `MTE2 cycles%` are clearly imbalanced, such as one being more than about 3x the other; prefer optimizing the dominant side first.
- In `[Pipe Distribution]`, `SCALAR cycles%` is the dominant share; prefer scalar-related optimization first.
- The kernel has no `tl.load` on the candidate path.
- The `tl.load` mainly loads scalar/index/control data; prefer `scalar-latency-traps` related optimization first.
- Prefetch or loop restructuring would move loads across true data dependencies.
- UB/register pressure cannot safely hold extra live tiles.

## Signals

### Profile

Use this profile gate when `extracted_bin_data/report.txt` exists. Locate the `[Pipe Overlap Ratio]` heading first, then read the overlap metric lines below it. Locate the `[Pipe Distribution]` heading separately, then read the cycle-share rows below it. Metric lines may be indented.

```text
[Pipe Overlap Ratio]

%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE)): 0.00%
%((VECTOR+CUBE)&MTE2/MTE2): 0.00%

[Pipe Distribution]  instr count / instr% / cycles / cycles% / dur%
Total instr: 260  |  Total cycles: 1000000
FLOWCTRL      instr=     15 (  5.8%)  cycles=     30000 (  3.0%)  dur= 2.50%
MTE2          instr=     32 ( 12.3%)  cycles=    360000 ( 36.0%)  dur=30.00%
MTE3          instr=     14 (  5.4%)  cycles=     60000 (  6.0%)  dur= 5.00%
SCALAR        instr=     40 ( 15.4%)  cycles=     90000 (  9.0%)  dur= 7.50%
VECTOR        instr=     90 ( 34.6%)  cycles=    320000 ( 32.0%)  dur=26.67%
CUBE          instr=     69 ( 26.5%)  cycles=    140000 ( 14.0%)  dur=11.67%
```

Interpret the fields as:

- `%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE))`: the share of vector/cube compute time overlapped with MTE2 load. This is the primary software-pipeline trigger.
- `%((VECTOR+CUBE)&MTE2/MTE2)`: the share of MTE2 load time overlapped with vector/cube compute. This should also be low before treating load/compute overlap as weak.
- `[Pipe Distribution]`: manually compute `compute_cycles%` from both `VECTOR cycles%` and `CUBE cycles%` when both rows exist; if only one of the two rows exists, use the available row. If only raw cycles are available, apply the same rule to raw cycles and divide by `Total cycles`. Compare `compute_cycles%` with `MTE2 cycles%`. Treat them as relatively balanced when each is at least about one-third of the other. In the example above, `compute_cycles% = 32.0% + 14.0% = 46.0%`, `MTE2 cycles% = 36.0%`, and `SCALAR cycles% = 9.0%`, which is a plausible software-pipeline scenario. If one side is more than about 3x the other, optimize the dominant side first. Also check `SCALAR cycles%`; if scalar cycles dominate, prefer scalar-related optimization before software-pipeline probing.

Use the full report section, not a single isolated line. Near-zero overlap, such as values below about 1%, is strong evidence that pipeline or prefetch may help.

### Code

- The kernel contains `tl.load`.
- The `tl.load` path is regular enough that a `for` loop or steady-state loop can expose repeated load/compute stages and enable compiler prefetch.
- `tl.load` is inside a loop, which already exposes repeated stages for `num_stages` tuning or manual prefetch.
- `tl.load` is outside a loop, but the kernel has a regular single-tile program shape that can be safely converted into a steady-state loop.

## Dependency Features

1. [profiling] Transfer/compute overlap is very low.

   Find the `[Pipe Overlap Ratio]` heading, then read `%((VECTOR+CUBE)&MTE2/(VECTOR+CUBE))` and `%((VECTOR+CUBE)&MTE2/MTE2)` below it. Very low values mean load and compute are mostly serialized.

2. [profiling] Vector/Cube compute and MTE2 load are relatively balanced.

   Find the `[Pipe Distribution]` heading, then compute `compute_cycles%` from both `VECTOR cycles%` and `CUBE cycles%` when both rows exist; if only one of the two rows exists, use the available row. If only raw cycles are available, compute the same share from `Total cycles`. Compare `compute_cycles%` with `MTE2 cycles%`. Treat them as relatively balanced when each is at least about one-third of the other; this keeps software pipelining plausible. If one side is more than about 3x the other, optimize the dominant side first. Also check `SCALAR cycles%`; if scalar cycles dominate, prefer scalar-related optimization first.

3. [code] Load existence and placement.

   Use this pattern when the kernel contains `tl.load` on a regular path where constructing a `for` loop or steady-state loop can enable compiler prefetch and improve performance. Check whether `tl.load` already appears in a loop. In-loop loads already expose repeated stages for overlap tuning. Loads outside a loop only match when the original program shape is regular enough to be converted into a steady-state loop.

4. [code] Manual prefetch state.

   Check whether the first loop load is already moved before the loop. If so, manual prefetch already exists and should not be duplicated.

## Optimization Strategy

1. Read `extracted_bin_data/report.txt` first when it exists.

   If the report exists, attempt this rule before choosing another optimization direction. Use the `[Pipe Overlap Ratio]` metrics and `[Pipe Distribution]` cycle shares as supporting evidence and record whether they support or weaken the software-pipeline probe.

2. Locate `tl.load` and classify placement.

   If there is no `tl.load` on the candidate path, choose another pattern. If `tl.load` exists, decide whether it is inside a loop or outside a loop.

3. Choose the pipeline transformation.

   - If `tl.load` is inside a loop, try bounded `num_stages` variants first.
   - If `num_stages` is flat and the first loop load is not already outside the loop, add manual prefetch.
   - If `tl.load` is outside a loop, try constructing a steady-state tile loop to make compiler prefetch possible when tiles are independent.
   - After constructing the steady-state loop, try a fixed grid sized to the target system's available core count.
   - If loop bounds are clear and the load is not protected by complex `if` logic, rely on compiler prefetch and tune `num_stages`.
   - Otherwise, use manual prefetch, then retune `num_stages`.

4. Keep only validated wins.

   The final change must pass correctness and improve representative benchmarks against the parent candidate.

## Optimization Example

### Steady-State Tile Loop

Use this when a regular single-tile program can safely process multiple independent tiles.

Before:

```python
@triton.jit
def elementwise_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = compute_vector_tile(x)
    tl.store(y_ptr + offs, y, mask=mask)
```

After:

```python
@triton.jit
def elementwise_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    block_start = pid * BLOCK_SIZE
    stride = num_programs * BLOCK_SIZE

    for i in range(block_start, n_elements, stride):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        y = compute_vector_tile(x)
        tl.store(y_ptr + offs, y, mask=mask)
```

This probe is useful only when the bounded grid leaves enough loop work per program and does not reduce parallelism more than it improves overlap.

### Fixed Grid Attempt

Use this after a steady-state loop rewrite when the original grid is very large. Size the grid from the target system's available core count; `48` is only a 48-core example.

```python
def grid(meta):
    return (48,)
```

### Manual Prefetch

Use this when a loop has serialized `load -> compute` and compiler prefetch is not formed.

```python
a_tile = tl.load(a_block_ptr)
b_tile = tl.load(b_block_ptr)

for _ in range(0, K, BLOCK_K):
    a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_K])
    b_block_ptr = tl.advance(b_block_ptr, [BLOCK_K, 0])

    acc = tl.dot(a_tile, b_tile, acc)

    a_tile = tl.load(a_block_ptr)
    b_tile = tl.load(b_block_ptr)
```

## What To Verify After Applying

- Correctness passes for full tiles, boundary tiles, dtypes, and representative shapes.
- Round evidence cites representative `report.txt` values from `[Pipe Overlap Ratio]` and the manually derived `compute_cycles%`, `MTE2 cycles%`, and `SCALAR cycles%` from `[Pipe Distribution]`.
- The code evidence records `tl.load` placement and manual-prefetch state when relevant.
- UB/register usage remains safe after prefetch or extra stages.
- Representative benchmarks improve against the parent candidate.

## Related Patterns

- `software-pipeline`
- `scalar-latency-traps`
- `classic-matmul`
- `layout-store-and-block-pointers`
- `autotune`
