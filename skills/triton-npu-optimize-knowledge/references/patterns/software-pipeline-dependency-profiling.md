---
priority: high
---

# Software Pipeline Dependency Profiling Pattern

## Summary

Use this pattern as a code-first software-pipeline probe whenever the kernel contains `tl.load`; if `extracted_bin_data/report.txt` exists, this probe must be attempted before choosing another optimization direction. If `tl.load` is outside a loop, try constructing a steady-state loop to make compiler prefetch possible; if `tl.load` is already inside a loop, try bounded `num_stages` tuning, with manual prefetch only when stage tuning is flat.

## Use When

- The kernel contains `tl.load`.
- If `tl.load` is not inside a loop, try constructing a steady-state loop to support compiler prefetch.
- If `tl.load` is already inside a loop, try bounded `num_stages` tuning; if stage tuning is flat, try manual prefetch.
- If `extracted_bin_data/report.txt` exists, this rule must be attempted. Use the report to judge whether most active `core*.veccore*` blocks show very low `OverlapRatio(VECTOR/CUBE & MTE2)`, very low `OverlapRatio(VECTOR/CUBE & MTE3)`, very low `OverlapRatio(MTE2 & MTE3)`, and low or moderate `Ratio(VECTOR/CUBE)`.
- Correctness and representative benchmark checks are available before keeping the change.

## Avoid When

- `OverlapRatio(VECTOR/CUBE & MTE2)` is already high enough that load and compute overlap well.
- `Ratio(VECTOR/CUBE)` is very high across most active cores; prefer compute-side optimization.
- The kernel has no `tl.load` on the candidate path.
- Prefetch or loop restructuring would move loads across true data dependencies.
- UB/register pressure cannot safely hold extra live tiles.

## Signals

### Profile

Use this profile gate when `extracted_bin_data/report.txt` exists. Parse it as text blocks headed by `core*.veccore*:`. Metric lines may be indented.

```text
core0.veccore0:
    OverlapRatio(VECTOR/CUBE & MTE2): 0.20%
    OverlapRatio(VECTOR/CUBE & MTE3): 0.10%
    OverlapRatio(MTE2 & MTE3): 0.00%
    Ratio(VECTOR/CUBE): 16.30%
```

Interpret the fields as:

- `OverlapRatio(VECTOR/CUBE & MTE2)`: compute/load overlap. This is the primary software-pipeline trigger.
- `OverlapRatio(VECTOR/CUBE & MTE3)`: compute/store overlap. Low values support weak overall overlap.
- `OverlapRatio(MTE2 & MTE3)`: load/store overlap. Low values support serialized transfer phases.
- `Ratio(VECTOR/CUBE)`: Vector/Cube compute share of total execution or total active time. Low or moderate values mean compute is not the only bottleneck.

Use the distribution across active cores, not a single core. Near-zero overlap, such as consistently below about 1%, is strong evidence that pipeline or prefetch may help.

### Code

- The kernel contains `tl.load`.
- `tl.load` is not inside a loop: construct a steady-state loop to support compiler prefetch.
- `tl.load` is inside a loop: tune `num_stages` first, then try manual prefetch if tuning is flat.

## Dependency Features

1. [profiling] Transfer/compute overlap is very low.

   Read `OverlapRatio(VECTOR/CUBE & MTE2)`, `OverlapRatio(VECTOR/CUBE & MTE3)`, and `OverlapRatio(MTE2 & MTE3)` from each active `core*.veccore*` block. Very low values across most active cores mean load, compute, and store are mostly serialized.

2. [profiling] Vector/Cube compute is not dominant.

   Read `Ratio(VECTOR/CUBE)` across active cores. Low or moderate values keep software pipelining plausible. Very high values indicate compute-side optimization is more likely to help.

3. [code] Load existence and placement.

   Use this pattern when the kernel contains `tl.load`. Check whether `tl.load` is inside a loop. Loads outside a loop favor steady-state loop construction to support compiler prefetch. In-loop loads favor `num_stages` tuning first, then manual prefetch if tuning is flat.

4. [code] Manual prefetch state.

   Check whether the first loop load is already moved before the loop. If so, manual prefetch already exists and should not be duplicated.

## Optimization Strategy

1. Read `extracted_bin_data/report.txt` first when it exists.

   If the report exists, attempt this rule before choosing another optimization direction. Use the four report metrics as supporting evidence and record whether they support or weaken the software-pipeline probe.

2. Locate `tl.load` and classify placement.

   If there is no `tl.load` on the candidate path, choose another pattern. If `tl.load` exists, decide whether it is inside a loop or outside a loop.

3. Choose the pipeline transformation.

   - If `tl.load` is not inside a loop but tiles are independent, construct a steady-state tile loop to support compiler prefetch.
   - If `tl.load` is inside a loop, try bounded `num_stages` variants first.
   - If `num_stages` is flat and the first loop load is not already outside the loop, add manual prefetch.
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
- Round evidence cites representative `report.txt` values for all four metrics.
- The code evidence records `tl.load` placement and manual-prefetch state when relevant.
- UB/register usage remains safe after prefetch or extra stages.
- Representative benchmarks improve against the parent candidate.

## Related Patterns

- `software-pipeline`
- `scalar-latency-traps`
- `classic-matmul`
- `layout-store-and-block-pointers`
- `autotune`
