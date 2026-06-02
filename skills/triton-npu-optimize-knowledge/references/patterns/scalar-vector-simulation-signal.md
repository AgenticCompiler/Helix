# Simulation Signal

## Summary

Match observed signals in overall content of `report.txt` against the categories below to identify the simulation bottleneck type, then follow the mapped pattern to the optimization.

------

## Use When

  - You have a `report.txt` output from `extracted_bin_data` (or you have already extracted simulation data and are about to analyze it). Focus on its overall content section.
  - Simulation data shows **abnormal Pipe distribution** in `report.txt` overall `[Pipe Distribution]` section (e.g., SCALAR instructions > 75%, VECTOR instructions < 15%, MTE2 cycles disproportionately high or zero).
  - `report.txt` overall `[Pipeline Flows]` section shows **no MTE2ToVECTOR flows** despite the kernel loading from global memory (Signal Category 5), or SCALARToVECTOR **avg > 50ns** (Signal Category 2).
  - `report.txt` overall `[TRACE Events]` section has **> 10,000 events** dominated by SIGNEXT/ADD/MUL/DIV/SUB/MADD (Signal Category 1).
  - `report.txt` overall `[VECTOR Unit]` UB Read or Write Conflict exceeds 100 (Signal Category 3).
  - `report.txt` overall `[VECTOR Unit]` Utilization avg < 30% (Signal Category 4).
  - You need to map an observed simulation signal to a related pattern and a concrete optimization direction.
  - You see `tl.load` with `mask`/`other` and want to determine whether the load is taking the slow SCALAR→VECTOR→MTE2 path (Path A) or the fast SCALAR→MTE2→VECTOR path (Path B).

  ## Avoid When

  - The optimization target is **pure tiling parameter tuning** (BLOCK_M/BLOCK_N/BLOCK_K, num_warps, grid config) — these are invisible in single-program simulation and must use hardware profiling.
  - The optimization target is **multi-program atomic contention** (e.g., `tl.atomic_add` under concurrent access) — simulation cannot reproduce this and signals will be misleading.
  - Simulation data shows **no signal hitting any threshold**.
  - The kernel is **inherently lightweight** (e.g., a trivial load+store touch kernel, a copy kernel, or a kernel with no compute loop) — an abnormal Pipe distribution is the natural characteristic of such operations, not an optimization target.

## Signal Matching Decision Guide

All metrics below are read from `report.txt` overall. Check signals in this order. The first match is the primary signal; secondary matches may co-occur.

1. **Cat 5 — Check overall `[Pipeline Flows]` section:** MTE2ToVECTOR count 0? If yes (and kernel has `tl.load`), → **Cat 5**.
2. **Cat 2 — Check overall `[Pipeline Flows]` section:** SCALARToVECTOR avg > 50ns? If yes, → **Cat 2**.
3. **Cat 1 — Check overall `[Pipe Distribution]` + overall `[TRACE Events]` sections:** SCALAR instruction % > 80%? If yes, → **Cat 1**.
4. **Cat 3 — Check overall `[VECTOR Unit]` section:** UB Read or Write Conflict > 100 AND utilization < 50% on conflicted instructions? If yes, → **Cat 3**.
5. **Cat 4 — Check overall `[VECTOR Unit]` section:** VECTOR utilization avg < 30%? If yes (and Cat 1/2 not matched), → **Cat 4**.

Multiple signals can co-occur. Common co-occurrences:

- Cat 1 + Cat 4: SCALAR explosion causes VECTOR starvation
- Cat 2 + Cat 1: Dispatch bottleneck with high SCALAR ratio (Cat 2 takes priority as the more actionable issue)

------

## Signal Category 1: Scalar Arithmetic Explosion

### Simulation Signature

| Metric               | Threshold                                  | report.txt section       | Source (original JSON)       |
| -------------------- | ------------------------------------------ | ------------------------------- | ---------------------------- |
| Trace event count    | > 10,000 (origin)                          | overall `[TRACE Events]` total_events  | `dataType_2_TRACE.json`     |
| Top event names      | SIGNEXT, ADD, MUL, DIV, SUB, MADD dominate | overall `[TRACE Events]` top_events    | `dataType_2_TRACE.json`     |
| SCALAR instruction % | > 80% of total instructions                | overall `[Pipe Distribution]` SCALAR   | `dataType_4_API_INSTR.json` |

> **Note on SIGNEXT:** SIGNEXT (sign extension) is a **type-conversion** instruction, not arithmetic. When SIGNEXT dominates over ADD/MUL/DIV in trace events, the root cause is likely **implicit dtype widening** (e.g., mixed-type arithmetic forcing int8→int32 / int16→int32 promotion, or i64→fp32 cast) rather than address computation. When ADD/MUL/DIV/SUB dominate, the root cause is classic scalar arithmetic. See the LayerNorm worked example in Cat 4 for a SIGNEXT-heavy case caused by i64→fp32 cast.

### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[Pipe Distribution]` SCALAR instr% > 80%
- **Precondition:** The kernel contains loops or compute logic that could be optimized (not a trivial load+store / touch kernel)
- **Confirmation:** overall `[TRACE Events]` total events > 10,000 AND top events dominated by SIGNEXT/ADD/MUL/DIV/SUB/MADD
- **Fire when:** Primary trigger AND precondition are both met. Confirmation helps identify root cause but is not required.

### What It Means

The kernel is spending most of its execution on scalar address computation, pointer arithmetic, coordinate decoding, or implicit type-conversion overhead. The VECTOR unit is under-fed because SCALAR is bottlenecked.

### Code Manifestations

Cat 1 can appear through several distinct code structures. Identify which one matches the current kernel, then apply the corresponding generic transform.

#### Manifestation A: Coordinate decode via `//` and `%`

Typical in: reduction kernels that flatten multi-dimensional indices into a single `pid`.

```python
# detect
while start < TOT:
    c_rel = idx // HW        # scalar division every iteration
    hw = idx - c_rel * HW    # scalar multiply + subtract
    ptrs = base_n + c_idx * HW + hw
    x = tl.load(x_ptr + ptrs, ...)
```

```python
# generic transform: explicit dimension unrolling with base+offset
while start < HW:
    x0 = tl.load(x_ptr + base + 0*HW + idx, ...)  # channel 0
    x1 = tl.load(x_ptr + base + 1*HW + idx, ...)  # channel 1
    ...
```

#### Manifestation B: Per-element scalar `tl.load` in a `tl.static_range` loop

Typical in: pooling, stencil, or sliding-window kernels where each loop iteration loads one scalar element.

```python
# detect
for kh in tl.static_range(0, KH):
    for kw in tl.static_range(0, KW):
        val = tl.load(base + (start + kw) * stride, mask=..., other=0.0)
        acc += val.to(tl.float32)
```

```python
# generic transform: load a contiguous slab once, then tl.gather per iteration
# SLAB_LEN = STRIDE * (BLOCK - 1) + KERNEL_SIZE
slab = tl.load(base + (w_start + tl.arange(0, SLAB_LEN)) * stride,
               mask=slab_mask, other=0.0)
lane_offsets = STRIDE * tl.arange(0, BLOCK)
for kw in tl.static_range(0, KW):
    val = tl.gather(slab, lane_offsets + kw, axis=0)
    acc += val.to(tl.float32)
```

#### Manifestation C: Loop-carried pointer recurrence

Typical in: kernels that update a pointer with `+=` inside a loop instead of computing each address from a stable base.

```python
# detect
ptr = base
for i in range(N):
    val = tl.load(ptr, ...)
    ptr = ptr + stride          # pointer chasing in loop
```

```python
# generic transform: base + offset addressing, no pointer mutation
for i in range(N):
    val = tl.load(base + i * stride, ...)
```

#### Manifestation D: Scalar condition feeding vector work

Typical in: kernels with per-element `if` branches that could be vector masks.

```python
# detect
if gamma >= 0:              # scalar branch for every element
    acc = tl.where(...)
```

```python
# generic transform: hoist scalar condition to vector mask
gamma_vec = tl.full([BLOCK], gamma, dtype=tl.float32)
mask = gamma_vec >= 0
acc = tl.where(mask, ..., 0.0)
```

### Optimization Direction

1. Replace `//` and `%` with compile-time-known strides (use `tl.constexpr` dimensions) — Manifestation A
2. Replace per-element scalar `tl.load` in loops with contiguous slab + `tl.gather` — Manifestation B
3. Replace loop-carried pointer recurrence with base+offset addressing — Manifestation C
4. Hoist scalar conditions to vector masks (`tl.full([BLOCK], val) >= threshold`) — Manifestation D

### Related Patterns

- `scalar-latency-traps`

### Worked Example

Origin simulation: 30,912 trace events, top events SIGNEXT(6,168), ADD(4,240), MUL(2,096), DIV(2,072), SUB(2,076), MADD(2,052).

Code anti-pattern (Manifestation A):

```python
while start < TOT:
    c_rel = idx // HW        # ← scalar division
    hw = idx - c_rel * HW    # ← scalar decode
    ptrs = base_n + c_idx * HW + hw
    x = tl.load(x_ptr + ptrs, ...)
```

Fix: explicit 4-channel unrolling with direct base+offset addressing.

```python
while start < HW:
    x0 = tl.load(x_ptr + base + 0*HW + idx, ...)  # channel 0
    x1 = tl.load(x_ptr + base + 1*HW + idx, ...)  # channel 1
    ...
```

Simulation after: 1,308 events (-95.8%), zero scalar arithmetic events. SCALARToVECTOR flows 12→6.

------

## Signal Category 2: Scalar-Vector Dispatch Bottleneck

### Simulation Signature

| Metric                   | Threshold                                            | report.txt section                           | Source (original JSON)                                        |
| ------------------------ | ---------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------- |
| SCALARToVECTOR avg_delta | > 50ns (significantly higher than other flow deltas) | overall `[Pipeline Flows]` SCALARToVECTOR avg              | `flows.json`                                                  |
| SCALAR instruction %     | > 75% of total instructions                          | overall `[Pipe Distribution]` SCALAR instr%                | `dataType_4_API_INSTR.json`                                   |
| VECTORToSCALAR flow count| > 300                                                | overall `[Pipeline Flows]` VECTORToSCALAR count            | `flows.json`                                                  |
| SCALAR cycles %          | > 40% of total cycles                                | overall `[Pipe Distribution]` SCALAR cycles%               | `dataType_4_API_INSTR.json`                                   |

### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[Pipeline Flows]` SCALARToVECTOR count > 0 AND avg > 50ns
- **Confirmation:** overall `[Pipe Distribution]` SCALAR instr% > 75% OR overall `[Pipeline Flows]` VECTORToSCALAR count > 300 OR overall `[Pipe Distribution]` SCALAR cycles% > 40%
- **Fire when:** Primary trigger is met AND at least 1 confirmation condition is met.
- **Differentiation from Cat 1:** Cat 1 has SCALAR% > 80% but SCALARToVECTOR avg is low; Cat 2's defining feature is high dispatch latency. If both SCALAR% > 80% and avg > 50ns, Cat 2 takes priority because the dispatch bottleneck is the more actionable issue.

### What It Means

The SCALAR unit dispatches work to VECTOR, but VECTOR takes a long time to become ready (high avg_delta). This creates pipeline bubbles where SCALAR waits for VECTOR. The kernel has many SCALAR→VECTOR and VECTOR→SCALAR synchronization points (e.g., many `tl.sum` operations each producing a scalar from a vector reduction).

### Code Manifestations

Cat 2 can appear through several distinct code structures. Identify which one matches the current kernel, then apply the corresponding generic transform.

#### Manifestation A: Per-iteration mask + predicated load + `tl.sum` in while-loop

Typical in: kernels with a while-loop over tiles where every iteration computes a mask and does a predicated load + scalar reduction.

```python
# detect
while m < M:
    idx = m + offs
    mask = idx < M                    # SCALAR mask compute every iteration
    vals = tl.load(base + idx, mask=mask, other=0.0, ...)  # predicated load
    s += tl.sum(vals)                 # VECTOR→SCALAR sync every iteration
    m += BLOCK_M
```

```python
# generic transform: split full-tile (no mask) + partial-tile tail + increase num_stages
while m + BLOCK_M <= M:
    idx = m + offs
    vals = tl.load(base + idx).to(tl.float32)  # no mask, no predication
    s += tl.sum(vals)
    m += BLOCK_M
if m < M:
    idx = m + offs
    mask = idx < M
    vals = tl.load(base + idx, mask=mask, other=0.0, ...)
    s += tl.sum(vals)
# In launch config: num_stages=4 (or higher) to overlap dispatch
```

#### Manifestation B: Many small `tl.sum` / `tl.max` reductions instead of fewer larger ones

Typical in: per-channel accumulation kernels where each channel produces a separate VECTOR→SCALAR sync.

```python
# detect
for c in range(C):
    vals_c = tl.load(base_c + offsets, ...)
    s += tl.sum(vals_c)               # one VECTOR→SCALAR per channel per iteration
```

```python
# generic transform: merge per-channel loads into wider operations, accumulate in vector
# Load multiple channels at once, accumulate before reducing to scalar
vals_all = tl.load(base + channel_offsets[:, None] * stride + n_offsets[None, :], ...)
s += tl.sum(vals_all)                # single VECTOR→SCALAR for all channels
```

### Optimization Direction

1. **Split while-loop into full-tile (no mask) + partial-tile tail**: Full tiles don't need `mask = idx < M` or predicated loads — saves SCALAR mask compute and removes VECTOR predication overhead — Manifestation A
2. **Increase `num_stages` in autotune configs**: Deeper software pipelining hides SCALAR→VECTOR dispatch latency by overlapping iterations — Manifestation A
3. **Batch more work per SCALAR→VECTOR dispatch**: Merge per-channel loads into wider operations where memory layout permits — Manifestation B
4. **Reduce VECTOR→SCALAR syncs**: Accumulate in vector registers where possible, reduce to scalar only at the end — Manifestation B

### Related Patterns

- `scalar-latency-traps`
- `software-pipeline`

### Worked Example

Opt simulation: SCALARToVECTOR avg_delta=85.4ns (MTE2ToVECTOR only 0.5ns, VECTORToSCALAR only 0.5ns). SCALAR instructions 78.5% (55,296/70,432). VECTORToSCALAR flows 384.

Code anti-pattern (Manifestation A):

```python
while m < M:
    idx = m + offs
    mask = idx < M                     # SCALAR: computed every iteration
    vals0 = tl.load(base0 + idx, mask=mask, other=0.0, ...)  # predicated
    vals1 = tl.load(base1 + idx, mask=mask, other=0.0, ...)
    vals2 = tl.load(base2 + idx, mask=mask, other=0.0, ...)
    s0 += tl.sum(vals0)                # VECTOR→SCALAR
    ...
```

Fix:

```python
# Full tiles: no mask, no predication
while m + BLOCK_M <= M:
    idx = m + offs
    vals0 = tl.load(base0 + idx, cache_modifier=".cg").to(tl.float32)  # no mask
    ...
    m += BLOCK_M
# Only tail tile uses mask
if m < M:
    idx = m + offs
    mask = idx < M
    vals0 = tl.load(base0 + idx, mask=mask, other=0.0, ...)
    ...

# Autotune: increased num_stages for deeper pipelining
configs=[
    triton.Config({"BLOCK_M": 8192}, num_warps=8,  num_stages=4),
    triton.Config({"BLOCK_M": 8192}, num_warps=16, num_stages=5),
    ...
]
```

------

## Signal Category 3: UB Conflict Bottleneck

### Simulation Signature

| Metric                                            | Threshold                                  | report.txt section                                        | Source (original JSON)                                       |
| ------------------------------------------------- | ------------------------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------ |
| UB Read Conflict on VECTOR                        | > 100 total across all VECTOR instructions | overall `[VECTOR Unit]` UB Read Conflict                               | `dataType_4_API_INSTR.json` — sum `UB Read Conflict` for Pipe=VECTOR |
| UB Write Conflict on VECTOR                       | > 100 total across all VECTOR instructions | overall `[VECTOR Unit]` UB Write Conflict                              | `dataType_4_API_INSTR.json` — sum `UB Write Conflict` for Pipe=VECTOR |
| Low VECTOR utilization on conflicted instructions | < 50%                                      | overall `[VECTOR Unit]` Top-conflict instrs U% column                  | `dataType_4_API_INSTR.json` — Vector Utilization Percentage  |

### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[VECTOR Unit]` UB Read Conflict > 100 OR UB Write Conflict > 100
- **Confirmation:** overall `[VECTOR Unit]` Top-conflict instrs show utilization < 50% (U% column)
- **Fire when:** Primary trigger AND confirmation are both met. Conflicts without low utilization may be benign (hardware handles them efficiently), so do not fire on conflicts alone.

### What It Means

VECTOR instructions (typically VADD reductions and VMUL squares) are experiencing Unified Buffer bank conflicts. Multiple vector operations are reading from the same UB bank simultaneously, causing stall cycles and reducing effective VECTOR throughput.

### Code Manifestations

Cat 3 can appear through several distinct code structures. Identify which one matches the current kernel, then apply the corresponding generic transform.

#### Manifestation A: Multiple `tl.load` into same UB region + immediate reduction

Typical in: kernels that load several channels and immediately reduce each one, causing reads and writes to collide in the same UB banks.

```python
# detect
vals0 = tl.load(base0 + idx, ...)    # writes to UB
vals1 = tl.load(base1 + idx, ...)    # may conflict with vals0's UB banks
s0 += tl.sum(vals0)                  # reads from UB — may conflict with vals1 write
s1 += tl.sum(vals1)
```

```python
# generic transform: separate load phase from reduction phase
vals0 = tl.load(base0 + idx, ...)    # load phase: all loads first
vals1 = tl.load(base1 + idx, ...)
# ... other scalar work here to separate load and reduce ...
s0 += tl.sum(vals0)                  # reduce phase: all reductions after
s1 += tl.sum(vals1)
```

#### Manifestation B: Predicated loads causing fragmented UB layout

Typical in: kernels where mask/predication on full-tile loads produces uneven UB write patterns that conflict with subsequent reductions.

```python
# detect
mask = idx < M
vals = tl.load(base + idx, mask=mask, other=0.0, ...)  # predicated → fragmented UB write
s += tl.sum(vals)                                       # reduction on fragmented data
```

```python
# generic transform: split full-tile (unmasked) + partial-tile (masked)
# Full tiles: unmasked load → clean UB layout, no bank conflicts
while m + BLOCK_M <= M:
    vals = tl.load(base + m + offs).to(tl.float32)  # no mask → clean UB write
    s += tl.sum(vals)
    m += BLOCK_M
# Only tail tile uses mask
if m < M:
    mask = m + offs < M
    vals = tl.load(base + m + offs, mask=mask, other=0.0, ...)
    s += tl.sum(vals)
```

#### Manifestation C: `tl.gather` (VGATHER) instructions causing UB conflicts

Typical in: kernels using `tl.gather` on loaded slabs where the gather operation itself creates bank conflicts on read and write.

```python
# detect (from sim_features: top-conflict instrs are VGATHER)
slab = tl.load(...)
for kw in tl.static_range(0, KW):
    val = tl.gather(slab, offsets + kw, axis=0)  # VGATHER → UB read/write conflict
    acc += val
```

```python
# generic transform: reduce gather iterations or reorganize slab layout
# Option 1: wider accumulation per gather call (fewer individual VGATHER ops)
# Option 2: load smaller slabs to reduce UB pressure per gather
# Option 3: pad slab length to avoid same-bank alignment in gather indices
```

### Optimization Direction

1. Separate loads from reductions — interleave loads from different channels with independent scalar work — Manifestation A
2. Remove mask/predication from full-tile loads (unmasked loads produce cleaner UB layout) — Manifestation B
3. Use `cache_modifier` tuning (`.cg` vs `.ca`) to control which cache level data lands in — Manifestation A / B
4. Pad or reorder channel access to avoid same-bank conflicts — Manifestation A / C

### Related Patterns

- `reorder-load`

### Worked Example

Opt simulation: UB Read Conflict 480 on VECTOR, all on VADD/VMUL reduction operations. Utilization 0-50% on conflicted instructions.

The opt3 optimization (Manifestation B: split full-tile + tail-tile) removes mask predication from full-tile loads, which changes the UB write pattern and reduces bank conflicts on the subsequent reductions.

------

## Signal Category 4: Vector Underutilization

### Simulation Signature

| Metric                          | Threshold      | report.txt section                              | Source (original JSON)                                        |
| ------------------------------- | -------------- | ------------------------------------------------------ | ------------------------------------------------------------- |
| VECTOR utilization (avg)        | < 30%          | overall `[VECTOR Unit]` Utilization avg                       | `dataType_4_API_INSTR.json` — average of Vector Utilization Percentage for Pipe=VECTOR |
| VECTOR instruction %            | < 15% of total | overall `[Pipe Distribution]` VECTOR instr%                   | `dataType_4_API_INSTR.json`                                   |
| SCALAR:VECTOR instruction ratio | > 4:1          | overall `[Key Ratios]` SCALAR:VECTOR_instr                    | `dataType_4_API_INSTR.json`                                   |

### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[VECTOR Unit]` Utilization avg < 30%
- **Precondition:** The kernel contains vector compute logic (e.g., arithmetic, reduction, `tl.where`, `tl.dot`) — not a trivial load+store / touch kernel where low VECTOR utilization is expected
- **Confirmation (optional, strengthen diagnosis):** overall `[Pipe Distribution]` VECTOR instr% < 15% AND overall `[Key Ratios]` SCALAR:VECTOR_instr ratio > 4:1
- **Fire when:** Primary trigger AND precondition are both met. Confirmation conditions are not required but strengthen confidence in the diagnosis.
- **Differentiation:** If Cat 1 or Cat 2 also matches, treat Cat 4 as a **consequence** (VECTOR is underutilized because SCALAR is overloaded or dispatch-bottlenecked). Only treat Cat 4 as the **primary signal** when Cat 1 and Cat 2 do NOT match — this means VECTOR starvation has a different root cause (e.g., `tl.where` with int64 operands forcing scalar fallback, preventing `vec_cmp` hardware path).

### What It Means

The VECTOR unit has low occupancy — there aren't enough vector instructions to keep it busy, or vector instructions are stalled waiting for data/scalar. This usually co-occurs with Signal 1 (Scalar Arithmetic Explosion) or Signal 3 (Dispatch Bottleneck).

### Code Manifestations

Cat 4 can appear through several distinct code structures. Identify which one matches the current kernel, then apply the corresponding generic transform.

#### Manifestation A: `tl.where` with int64 operands causing scalar fallback

Typical in: kernels where `tl.where` comparison uses index tensors (int64 by default), preventing hardware `vec_cmp` path.

```python
# detect
cols = tl.arange(0, BLOCK_SIZE)
xbar = tl.where(cols < N, x - mean, 0.0)   # cols is int64 → scalar fallback
```

```python
# generic transform: cast index to float32 before comparison
cols_cmp = cols.to(tl.float32)                         # int64 → float32 cast
xbar = tl.where(cols_cmp < N, x - mean, 0.0)          # now uses vec_cmp
```

#### Manifestation B: Scalar-heavy loop with tiny vector work per iteration

Typical in: kernels where each loop iteration has lots of scalar address computation but only one or two vector operations.

```python
# detect
for ...:
    # lots of scalar address computation
    vals = tl.load(...)          # only vector work
    s += tl.sum(vals)            # then back to scalar
```

```python
# generic transform: widen the vector work per iteration
# Load wider tiles, compute multiple outputs per iteration
vals = tl.load(base + offsets[:, None] * stride + lane_offsets[None, :], ...)
# More vector arithmetic per scalar dispatch
s += tl.sum(vals, axis=1)       # wider reduction
```

#### Manifestation C: No vector compute at all (pure scalar / load+store)

Typical in: trivial kernels that only move data without arithmetic, where low VECTOR utilization is inherent and not an optimization target.

```python
# detect: kernel has no arithmetic, only load + store
vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
tl.store(y_ptr + offsets, vals, mask=mask)
```

```python
# This is NOT an optimization target — Cat 4 should not fire for this case.
# Check the Avoid When section: inherently lightweight kernels are excluded.
```

### Optimization Direction

1. Move more work to VECTOR: use `tl.where` instead of scalar `if`, use vector masks instead of scalar branches — Manifestation A / B
2. Cast int64 indices to float32 before `tl.where` comparison to enable `vec_cmp` hardware path — Manifestation A
3. Increase the ratio of VECTOR work per scalar dispatch (wider loads, larger tiles) — Manifestation B
4. Fuse multiple kernels to amortize scalar overhead across more vector work — Manifestation B

### Related Patterns

- `scalar-latency-traps`
- `vec-cmp`
- `software-pipeline`
- `software-pipeline-dependency-profiling`

### Worked Example

**Case: LayerNorm vectorized compare fix (Manifestation A)**

Profiling evidence for `_layer_norm_fwd_fused` kernel: `aiv_vec_ratio < 10%`, `aiv_scalar_ratio ~60%`. The simulation pipeline shows SCALAR and FLOWCTRL saturated while MTE2/VECTOR are regularly interrupted.

Root cause: `tl.where` with `int64` comparison operands causes NPU scalar fallback instead of hardware vectorized compare (`vec_cmp`).

Code anti-pattern:

```python
@triton.jit
def layer_norm_fwd_fused(
    X, Y, W, B, RES, Mean, Rstd,
    stride, N, eps,
    BLOCK_SIZE: tl.constexpr
):
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xbar = tl.where(cols < N, x - mean, 0.0)   # cols is int64 → scalar fallback
    # ... rest of norm computation
```

Fix: explicitly cast index to `float32` before comparison, enabling `vec_cmp` hardware path:

```python
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    cols_cmp = cols.to(tl.float32)                         # int64 → float32 cast
    xbar = tl.where(cols_cmp < N, x - mean, 0.0)          # now uses vec_cmp
```

After fix: VECTOR utilization increases significantly, scalar ratio drops. The `tl.load`/`tl.store` mask parameter (`cols < N`) is auto-optimized by the compiler, but `tl.where` requires manual vectorization via dtype cast.

------

## Signal Category 5: Missing Memory Engine (No MTE2↔VECTOR)

### Simulation Signature

| Metric              | Threshold     | report.txt section              | Source (original JSON) |
| ------------------- | ------------- | -------------------------------------- | ---------------------- |
| MTE2ToVECTOR flows  | 0 (or near 0) | overall `[Pipeline Flows]` MTE2ToVECTOR count | `flows.json`           |
| SCALARToVECTOR flows| > 0 (present) | overall `[Pipeline Flows]` SCALARToVECTOR count | `flows.json`         |

### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[Pipeline Flows]` MTE2ToVECTOR count = 0 (missing rows also count as 0)
- **Confirmation:** overall `[Pipeline Flows]` SCALARToVECTOR count > 0
- **Fire when:** Primary trigger AND confirmation are both met, AND the kernel logically should load from global memory (i.e., it contains `tl.load` from non-constant pointers). If the kernel is pure register computation with no global memory access, MTE2 absence is expected — do not fire.

### What It Means

The kernel has SCALAR↔VECTOR activity but no MTE2↔VECTOR flows. This means the memory engine (MTE2) is not being used to feed the VECTOR unit — data is likely being routed through SCALAR first, or the kernel doesn't load from global memory at all (e.g., pure register computation).

If the kernel SHOULD be loading data (most cases): this is a severe issue — the memory path is completely bypassed.

### Code Manifestations

Cat 5 can appear through several distinct code structures. Identify which one matches the current kernel, then apply the corresponding generic transform.

#### Manifestation A: Scattered index-driven `tl.load` bypassing MTE2

Typical in: kernels using non-contiguous index vectors to drive `tl.load`, routing data through SCALAR→VECTOR instead of MTE2→VECTOR.

```python
# detect
idx = tl.load(idx_ptr + rn * stride_idx)      # load index vector
val = tl.load(x_ptr + idx * stride_x, mask=mask)  # non-contiguous gather → scalar path
```

```python
# generic transform: contiguous tile load + local gather
rm = tl.arange(0, M)
rn = tl.arange(0, N)
idx = tl.load(idx_ptr + rn * stride_idx)
x_shared = tl.load(x_ptr + rm * stride_x)    # contiguous MTE2ToVECTOR load, shape [M]
val = tl.gather(x_shared, idx, axis=0)         # local gather on loaded tile, shape [N]
```

#### Manifestation B: Per-element scalar `tl.load` producing zero MTE2→VECTOR flows

Typical in: kernels where each program loads a single scalar element (1 output = 1 program), resulting in zero MTE2ToVECTOR activity because each transfer is too narrow for the memory engine.

```python
# detect: each program processes 1 output element
pid = tl.program_id(0)
val = tl.load(x_ptr + pid, ...)   # single scalar element, no MTE2→VECTOR
tl.store(y_ptr + pid, val)
```

```python
# generic transform: each program processes BLOCK outputs, vectorized load
pid = tl.program_id(0)
offs = pid * BLOCK + tl.arange(0, BLOCK)
mask = offs < N
vals = tl.load(x_ptr + offs, mask=mask, other=0.0)  # vectorized MTE2→VECTOR
tl.store(y_ptr + offs, vals, mask=mask)
```

#### Manifestation C: Pure register computation with no `tl.load` at all

Typical in: kernels that compute results from register/constant data without loading from global memory — zero MTE2ToVECTOR is expected, not a bug.

```python
# detect: kernel has no tl.load from global memory
out = tl.full([BLOCK], val, dtype=tl.float32) * 2  # register-only computation
tl.store(y_ptr + offsets, out)
```

```python
# Cat 5 should NOT fire for this case — MTE2 absence is expected.
# Check the Matching Rule: kernel should logically load from global memory.
```

### Optimization Direction

1. Replace scattered `tl.load(x_ptr + idx * stride)` with contiguous tile load + `tl.gather` — Manifestation A
2. Ensure `tl.load` is used for all global memory accesses (not scalar loads) — Manifestation B
3. Increase per-program work: process multiple elements per program with vectorized loads — Manifestation B
4. Verify `cache_modifier` is set appropriately (`.cg` for streaming, `.ca` for cached) — Manifestation A

### Related Patterns

- `discrete_memory_access`
- `block-pointer-dimensionality`

### Worked Example

**Case: Scattered index-driven load bypassing MTE2 (Manifestation A)**

When a kernel uses non-contiguous index vectors to drive `tl.load`, the hardware may fall back to scalar-gather paths that bypass MTE2ToVECTOR entirely, showing zero MTE2ToVECTOR flows in simulation.

Code anti-pattern (GPU-style scatter):

```python
@triton.jit
def scatter_load_kernel(
    x_ptr, idx_ptr, out_ptr, M, N,
    stride_x, stride_idx,
    BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    rn = tl.arange(0, BLOCK_N)
    idx = tl.load(idx_ptr + rn * stride_idx)      # load index vector
    mask = idx < M
    val = tl.load(x_ptr + idx * stride_x, mask=mask)  # non-contiguous gather → scalar path
    tl.store(out_ptr + rn, val, mask=mask)
```

The `idx * stride_x` produces non-contiguous addresses. On NPU this can route through SCALARToVECTOR instead of the fast MTE2ToVECTOR path.

Fix: when the source has contiguous structure on at least one axis, load a contiguous tile first then index locally:

```python
@triton.jit
def gather_load_fixed(
        x_ptr, idx_ptr, out_ptr, M, N,
        stride_x, stride_idx,
):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)

    idx = tl.load(idx_ptr + rn * stride_idx)
    mask = idx < M

    x_shared = tl.load(x_ptr + rm * stride_x)    # contiguous MTE2ToVECTOR load, shape [M]
    val = tl.gather(x_shared, idx, axis=0)         # local gather on loaded tile, shape [N]
    tl.store(out_ptr + rn, val, mask=mask)
```

The contiguous `tl.load` now uses MTE2ToVECTOR, and `tl.gather` operates on already-loaded UB data.

------
