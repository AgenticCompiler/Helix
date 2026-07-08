---
priority: high
---

# Triton-Ascend Autotune: Signal-Driven Parameter Optimization

## Summary

Two-phase optimization from `report.txt` simulation data:

- **Phase 1 (Pre-Gate A-Cat-5):** Determine **whether** autotune is the right tool. A-Cat-5 firing is a hint that memory layout may be fragmented — verify it is not a simulator false positive before accepting the ruling. If verified as a true positive, route to `compile_hint` or `discrete_memory_access` instead of autotune; if verified as a false positive (e.g. pure data-movement kernel), proceed to Phase 2.
- **Phase 2 (Parameter Diagnostics):** Only if Pre-Gate passes — match `report.txt` statistical features against A-Cat-6 through A-Cat-4 to identify which parameters need tuning, then configure `@triton.autotune` using Route 1 (auto-infer `configs=[]`), Route 2 (`hints`), or Route 3 (hand-written `triton.Config` lists).

## Use When

- The kernel logic is mathematically correct, stable, and passes validation tests.
- You have a `report.txt` output from `extracted_bin_data`. Focus on its overall sections.
- The kernel structure already looks semantically correct, and the likely headroom is in `BLOCK_*` selection, split shape, `num_stages`, or autotune `key` configuration.
- The hot path exposes one or more free `tl.constexpr` parameters that are not hard-coded at launch time.
- Bounds masks or loop structure still map cleanly back to runtime shape arguments, so a shape-keyed autotune cache is plausible.
- The operator is vector-like rather than a Cube-only kernel path that needs a different optimization route.
- `report.txt` shows one or more of: near-0% MTE2&VECTOR overlap, SCALAR dominance, small ProcessBytes per MTE fetch, high WAIT_FLAG without compute overlap, register pressure symptoms.
- For A-Cat-3 and A-Cat-4, additional data sources beyond `report.txt` are required (see category notes).

## Avoid When

- The real problem is structural, such as a manual matmul or reduction that should first become a regular tiled `tl.dot` loop.
- The core algorithmic logic or global memory layouts are undergoing major design refactoring — stabilize the kernel structure first.
- The pipeline is entirely bound by global memory bandwidth limits (compute cycles fully hidden behind continuous MTE transfers) — autotune has hit its physical ceiling; operator fusion or lower precision (FP32 → FP16/BF16) is required instead.
- All relevant `tl.constexpr` parameters are already fixed at launch time with no meaningful tuning space.
- A semantic constraint fixes one grid dimension or one tile shape so tightly that generated candidates would mostly be invalid or meaningless.
- One parameter simultaneously controls multiple unrelated axes or both launch count and inner tile semantics in a way that automatic parsing cannot represent cleanly.
- The kernel is correctness-fragile under repeated benchmarking and has not yet added the reset or restore hooks needed for safe autotune evaluation.

## Global Constraints

Before outputting any code modifications or autotune configurations, the following rules must be satisfied:

1. **Hardware Alignment Enforcement:** All tiling sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) must be powers of 2. Inspect the kernel for underlying hardware alignment constraints or static assertions (e.g., `tl.static_assert(BLOCK_K % 16 == 0)`). Generated configurations must never violate these physical boundaries.
2. **SRAM (UB) Resource Trade-Off:** Tiling sizes and software pipelining stages compete for the same limited on-chip memory (SRAM/Unified Buffer). Total shared memory utilization scales proportionally with O(BLOCK_M × BLOCK_K × num_stages). If increasing `num_stages` to hide latency triggers an out-of-memory (OOM) error or causes register spilling, scale down `BLOCK_K` or other tile dimensions to compensate.
3. **Data-Driven Logic:** Base all performance reasoning purely on concrete metrics from `report.txt`: instruction cycle ratios, `WAIT_FLAG` execution frequencies, `ProcessBytes` averages, pipe overlap ratios, and pipeline flow deltas. Abandon qualitative visual descriptions.
4. **Autotune Timeout and Recovery:** If `run-test` times out, **reduce config count first (start with 3-4 configs for operators with many test cases) — NOT abandon autotune**. Host-side heuristics cannot pick per-shape optimal `BLOCK_SIZE`. Large `BLOCK_*` values (>8192) on atomic-add or heavily-masked kernels trigger disproportionate JIT compilation time. When timeout occurs with large `BLOCK_*` in configs, **remove the extreme values first** — reducing config count alone is insufficient if extreme `BLOCK_*` remains, because a single extreme config's JIT + benchmark cost can dominate the entire timeout budget.

## Phase 1: Pre-Gate — Should You Use Autotune? (A-Cat-5)

**Run this check BEFORE entering parameter diagnostics.** A-Cat-5 firing is a strong hint that autotune may be the wrong tool, but it is a hint, not a final ruling — on Ascend NPU, MSProf simulator signals on pure data-movement kernels can false-positive (see "Note on ProcessBytes computation" below). Verify the signal is not a false positive before accepting the ruling.

### Simulation Signature

| Metric | Threshold | report.txt section |
|---|---|---|
| ProcessBytes per MTE load | avg < 128 bytes (per data mover) | overall `[MTE2 Data Transport]` ProcessBytes / data mover avg |
| MTE2 data mover count | < 2 data movers despite kernel loading multi-element tensors | overall `[MTE2 Data Transport]` Data movers |
| SCALAR load instructions | SIGNEXT/MOV_XD_SPR/LDP_XI_XJ_XN dominate top SCALAR instr types | overall `[SCALAR Instr Types]` top instr names |
| MTE2&VECTOR overlap | near 0% despite kernel doing both load and compute | overall `[Pipe Overlap Ratio]` %(MTE2&VECTOR/VECTOR) |

> **Note on data movers:** `Data movers: 0` in `[MTE2 Data Transport]` with `Flow control: N` means all MTE2 instructions are flow-control (SET_FLAG/END_LABEL), not actual data movement. This is a strong signal that memory access is routing through SCALAR instead of MTE2.

> **Note on ProcessBytes computation:** `ProcessBytes avg` in `[MTE2 Data Transport]` is computed as `bytes moved / MTE2 instruction count`. When most MTE2 instructions are classified as flow-control (SET_FLAG/END_LABEL) — e.g. `Data movers: 1, Flow control: 7` — both the numerator and denominator are dominated by flow-control overhead, not real data movement. The resulting `ProcessBytes avg = 0B` (or any value < 128B) is a **simulator signal artifact**, not a real indication that the memory layout is fragmented.

> **When ProcessBytes is not a valid A-Cat-5 signal:** Do not treat `ProcessBytes avg < 128B` as a valid A-Cat-5 trigger when ALL of the following hold:
> - Kernel body is pure data-movement (only `tl.load` → optional mask/transpose → `tl.store`, no compute ops like `*`, `+`, `exp`, `sum`).
> - `Data movers ≥ 1` (at least one real MTE2 data mover, distinguishing from the all-flow-control case covered by the note above).
> - Host-side input is already `.contiguous()` and strides match physical layout.
>
> In this case, the kernel's actual bottleneck is launch overhead / grid decomposition, not memory fragmentation. Proceed to Phase 2 parameter diagnostics with `BLOCK_SIZE` and `BLOCKS_PER_PROGRAM` as candidate autotune parameters.

### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[MTE2 Data Transport]` ProcessBytes / data mover avg < 128 bytes, OR Data movers = 0 while kernel has `tl.load`
- **Confirmation:** overall `[Pipe Overlap Ratio]` %(MTE2&VECTOR/VECTOR) < 5%
- **Fire when:** Primary trigger AND confirmation are both met, AND kernel contains `tl.load` from non-constant pointers. If the kernel is pure register computation with no global memory access, small ProcessBytes is expected — do not fire. Firing triggers the verify-before-ruling flow in the Decision section below; it does not by itself route away from autotune.

### Decision When A-Cat-5 Fires

Before accepting A-Cat-5 as final, verify the signal is not a false positive:

1. **Check the kernel body.** If it is pure data-movement (only `tl.load` → optional mask/transpose → `tl.store`, no compute ops like `*`, `+`, `exp`, `sum`), see "Note on ProcessBytes computation" — A-Cat-5 may be a simulator artifact. Proceed to Phase 2 parameter diagnostics with `BLOCK_SIZE` and `BLOCKS_PER_PROGRAM` as candidates.
2. **Check `[SCALAR Instr Types]`.** If SCALAR is dominated by address computation (MOV_XD_SPR, LDP_XI_XJ_XN) AND the kernel has compute ops, the fragmentation is real — apply **`compile_hint`** if the access pattern is logically contiguous but the compiler can't prove it, or **`discrete_memory_access`** if the access is inherently index-driven.
3. **Check host-side strides.** If input strides don't match physical layout after transpose/view operations, apply `.contiguous()` rearrangement, re-run simulation, and re-check the Pre-Gate.

Only after ruling out the false-positive cases above, treat A-Cat-5 as final: stop Phase 2 parameter diagnostics — autotune parameter search (`BLOCK_*`, `num_stages`, `num_warps`) cannot resolve a fundamentally broken physical layout.

### What It Means

This section describes the case where A-Cat-5 is verified as a true positive (not a false positive). The compiler cannot statically verify that the memory access pattern is perfectly contiguous along the vectorization axis. It generates conservative, fragmented, narrow scalar loads. Autotune configurations cannot resolve a fundamentally broken physical layout.

### Code Manifestations

#### Manifestation A: Transposed / strided input without host-side contiguous rearrangement

Typical in: matmul kernels where input B is transposed but passed with original strides.

```python
# detect: stride arguments don't match physical layout after transpose
B = B.transpose(0, 1)            # logical transpose, not physically contiguous
_matmul_kernel[grid](A, B, C, stride_b_n=old_stride_n, stride_b_k=old_stride_k)
```

```python
# generic transform: force physical contiguity before kernel launch
B_kn = B.transpose(0, 1).contiguous()
_matmul_kernel[grid](A, B_kn, C, stride_b_n=B_kn.stride(1), stride_b_k=B_kn.stride(0))
```

#### Manifestation B: Un-coalesced dimension ordering in loop nest

Typical in: element-wise kernels where the fastest-varying loop index does not match the contiguous memory dimension.

```python
# detect: loop order doesn't match memory layout
for n in range(N):          # outer loop over non-contiguous dim
    for m in range(M):      # inner loop over contiguous dim — but should be fused
        ...
```

```python
# generic transform: fuse dimensions into single contiguous range
offs = pid * BLOCK + tl.arange(0, BLOCK)
```

### Optimization Direction

Ensure global memory layout aligns with the fastest-moving index in the loop.

1. **Host-Side Preparation:** If incoming matrices have irregular strides due to preceding operators (such as transpositions), apply explicit physical rearrangement on the host before kernel launch. Synchronize `stride` arguments with the modified layouts.
2. **In-Kernel Constraints:** Inject strong compiler hints to guarantee memory contiguity and baseline vector alignment.

```python
# Host-Side: Force physical contiguity and adjust strides
B_kn = B.transpose(0, 1).contiguous()

# Kernel Launch Synchronization
_matmul_kernel[grid](
    A, B_kn, C,
    stride_b_n=B_kn.stride(1),
    stride_b_k=B_kn.stride(0),
    ...
)
```

```python
# In-Kernel: Inject compiler hints for vectorization
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
```

### Related Patterns

- `compile_hint`
- `discrete_memory_access`

### Worked Example

Simulation: `[MTE2 Data Transport]` Data movers=0, Flow control=2, ProcessBytes avg=0B. `[Pipe Overlap Ratio]` MTE2&VECTOR/VECTOR=0.00%. `[SCALAR Instr Types]` dominated by MOV_XD_SPR, LDP_XI_XJ_XN (address computation).

The kernel loads from global memory but all data movement routes through SCALAR instead of MTE2 because the memory access pattern is non-contiguous. The compiler cannot generate wide vectorized loads.

Fix: apply host-side `.contiguous()` before kernel launch and align strides with physical layout. After fix: MTE2 data movers > 0, ProcessBytes avg > 128, MTE2&VECTOR overlap > 0%. After the Pre-Gate is cleared, proceed to Phase 2 parameter diagnostics if further tuning is needed.

------

## Phase 2: Parameter Diagnostics — Signal Matching Decision Guide

**Precondition: A-Cat-5 Pre-Gate must have been checked AND either did NOT fire, OR fired but was verified as a false positive (see Phase 1 Decision).** If A-Cat-5 fired and was verified as a true positive, stop here — fix the memory layout first (route to `compile_hint` / `discrete_memory_access` / `.contiguous()`), then re-run simulation and re-check the Pre-Gate before returning to Phase 2.

All metrics below are read from `report.txt` overall sections unless otherwise noted. Check signals in this order. The first match is the primary signal; secondary matches may co-occur.

1. **A-Cat-6 — Check `[Pipe Distribution]` + `[WAIT_FLAG / BAR Sync]`:** SCALAR cycles% > 50% AND WAIT_FLAG total < 100, but kernel has compute stalls? → **A-Cat-6 (Register Spilling)**
2. **A-Cat-1 — Check `[Pipe Overlap Ratio]` + `[WAIT_FLAG / BAR Sync]`:** MTE2&VECTOR overlap near 0%? WAIT_FLAG total high? → **A-Cat-1 (Pipeline Overlap Deficit)**
3. **A-Cat-2 — Check `[Pipe Distribution]` + `[Key Ratios]`:** SCALAR instr% > 80%? SCALAR:VECTOR_instr ratio > 4:1? → **A-Cat-2 (Scalar Overhead Dominance)**
4. **A-Cat-3 — Check hardware profiling (NOT in `report.txt`):** Grid blocks ≪ physical core count? Neither A-Cat-1 nor A-Cat-2 matched? → **A-Cat-3 (Parallelism Starvation)** — **requires msprof or benchmark data**
5. **A-Cat-4 — Check benchmark (NOT in `report.txt`):** Performance degrades on specific shapes but works optimally elsewhere? → **A-Cat-4 (Autotune Key Mismatch)** — **requires multi-shape benchmark data**

Multiple signals can co-occur. Common co-occurrences:

- A-Cat-2 + A-Cat-6: SCALAR dominance combined with register pressure from oversized blocks
- A-Cat-1 + A-Cat-2: Pipeline serialization together with scalar overhead — A-Cat-1 takes priority (fix pipeline first, then scale tiles)

### Common Non-Matches

- The core algorithmic logic or global memory layouts are undergoing major design refactoring — stabilize the kernel structure first.
- The pipeline is entirely bound by global memory bandwidth limits (compute cycles fully hidden behind continuous MTE transfers) — autotune has hit its physical ceiling; operator fusion or lower precision (FP32 → FP16/BF16) is required instead.
- All relevant `tl.constexpr` parameters are already fixed at launch time with no meaningful tuning space.

------

### A-Cat-6: Register Spilling

#### Simulation Signature

| Metric | Threshold | report.txt section |
|---|---|---|
| SCALAR cycles% | > 50% of total cycles | overall `[Pipe Distribution]` SCALAR cycles% |
| WAIT_FLAG total | < 100 (rules out A-Cat-1 pipeline stalls) | overall `[WAIT_FLAG / BAR Sync]` WAIT_FLAG total |
| VECTOR cycles% | > 20% but no corresponding WAIT_FLAG bottleneck | overall `[Pipe Distribution]` VECTOR cycles% |
| SCALAR:VECTOR_cycles ratio | > 3:1 | overall `[Key Ratios]` SCALAR:VECTOR_cycles |

> **Note on report.txt limitations:** `report.txt` does not have a direct local memory read/write counter. Register spilling must be inferred indirectly: high SCALAR cycles (register save/restore sequences expand into scalar instructions) combined with low WAIT_FLAG (ruling out pipeline serialization) and normal MTE2 overlap (ruling out memory stalls). When in doubt, check the compiled kernel assembly for stack-allocated local memory spills.

#### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[Pipe Distribution]` SCALAR cycles% > 50% AND overall `[WAIT_FLAG / BAR Sync]` WAIT_FLAG total < 100
- **Confirmation:** overall `[Pipe Distribution]` VECTOR cycles% > 20% AND overall `[Key Ratios]` SCALAR:VECTOR_cycles > 3:1
- **Fire when:** Primary trigger AND confirmation are both met, AND kernel is known to use large BLOCK sizes or high num_warps. This is a weaker signal than others in `report.txt` — false positives are possible when the kernel has genuine scalar-heavy logic (not spilling). Cross-check with kernel's `BLOCK_*` and `num_warps` values: spilling is more likely when `BLOCK_M * BLOCK_N * BLOCK_K > 2^20` or `num_warps >= 8`.
- **Differentiation from A-Cat-2:** A-Cat-2 has SCALAR instr% > 80% with high SCALAR:VECTOR_instr ratio — the scalar overhead comes from small tile sizes. A-Cat-6 has high SCALAR cycles% but instr% may be moderate — the extra cycles come from register save/restore (spill code), not from inherently scalar-heavy logic.

#### What It Means

The architectural resource limit of vector or scalar registers per thread block has been exceeded. When `BLOCK_SIZE` or `num_warps` are over-provisioned, the compiler is forced to spill excess variables out of high-speed registers into slow off-chip local memory structures. The spill/fill sequences manifest as additional SCALAR cycles without corresponding WAIT_FLAG stalls.

#### Code Manifestations

##### Manifestation A: Over-provisioned BLOCK with many live variables

```python
# detect: many acc variables live at once + large BLOCK
acc0 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
acc2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
# ... with BLOCK_M=256, BLOCK_N=256 → each acc is 256KB → register pressure
```

```python
# generic transform: reduce BLOCK dimensions or split accumulators across sequential passes
acc = tl.zeros([BLOCK_M // 2, BLOCK_N], dtype=tl.float32)  # halve register footprint
# process in two passes if needed
```

##### Manifestation B: High num_warps causing collective register exhaustion

```python
# detect: num_warps=8 or 16 in config
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2)
```

```python
# generic transform: reduce num_warps, compensate with larger BLOCK or num_stages
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3)
```

#### Optimization Direction

De-escalate register pressure by tightening block boundaries or optimizing block-level concurrency.

1. **Reduce Tiling Geometry:** Scale down `BLOCK_M`, `BLOCK_N`, or `BLOCK_K` to shrink the overall tensor block size residing in registers simultaneously.
2. **Re-evaluate Warp Allocation:** Adjust `num_warps`. Increasing warps can sometimes split register pressure across more threads, but excessive warps can also trigger collective spillages depending on compiler allocation bounds.
3. **Reduce Live Variables:** Fuse sequential operations, or compute partial results in separate kernel passes to reduce simultaneous register demand.

```python
# Option 1: Reduce tiling geometry
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)

# Option 2: Reduce num_warps, compensate with num_stages
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
```

#### Related Patterns

- `software-pipeline`

------

### A-Cat-1: Pipeline Overlap Deficit

#### Simulation Signature

| Metric | Threshold | report.txt section |
|---|---|---|
| MTE2&VECTOR overlap (MTE2 side) | < 5% (near 0%) | overall `[Pipe Overlap Ratio]` %(MTE2&VECTOR/MTE2) |
| MTE2&VECTOR overlap (VECTOR side) | < 5% (near 0%) | overall `[Pipe Overlap Ratio]` %(MTE2&VECTOR/VECTOR) |
| (VECTOR+CUBE)&MTE2 overlap | < 5% | overall `[Pipe Overlap Ratio]` %((VECTOR+CUBE)&MTE2/(VECTOR+CUBE)) |
| WAIT_FLAG total | > 50 (blocking stalls dominate) | overall `[WAIT_FLAG / BAR Sync]` WAIT_FLAG total |
| Pipeline flow SCALARToVECTOR avg_delta | > 50ns | overall `[Pipeline Flows]` SCALARToVECTOR avg |

#### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[Pipe Overlap Ratio]` %(MTE2&VECTOR/MTE2) < 5% OR %((VECTOR+CUBE)&MTE2/(VECTOR+CUBE)) < 5%
- **Confirmation:** overall `[WAIT_FLAG / BAR Sync]` WAIT_FLAG total > 50
- **Secondary confirmation (optional):** overall `[Pipeline Flows]` SCALARToVECTOR avg > 50ns
- **Fire when:** Primary trigger AND confirmation are both met. If MTE2 data movers = 0 (kernel is not actually loading data through MTE2), this signal may be a false positive — check Pre-Gate A-Cat-5 first.
- **Precondition:** The kernel contains both `tl.load` (MTE activity) and compute (VECTOR activity). If the kernel is pure load+store or pure compute, zero overlap is expected — do not fire.

#### What It Means

The execution pipeline suffers from shallow multi-buffering depth. The system executes in strict serial cadence ("fetch → wait → compute → fetch next"), failing to asynchronously prefetch memory blocks for future loop iterations while computing the current block.

#### Code Manifestations

##### Manifestation A: Single-buffered loop with no software pipelining

```python
# detect: no num_stages in config or num_stages=1
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4)
```

```python
# generic transform: add num_stages sweep to autotune space
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
```

##### Manifestation B: num_stages too low for the ratio of compute to load

```python
# detect: num_stages=2 but WAIT_FLAG still dominates
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2)
```

```python
# generic transform: push num_stages higher, reduce BLOCK_K if SRAM overflows
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=5),
```

#### Optimization Direction

Expand the software pipelining depth via the `num_stages` search space in `@triton.autotune`. Pushing stages to higher bounds (e.g., 3, 4, or more) instructs the compiler to pipeline multi-buffered data blocks ahead of time. If SRAM capacity limits are breached, combine with a proportionate decrease in `BLOCK_K` (refer to Global Constraint 2).

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
```

#### Related Patterns

- `software-pipeline`
- `software-pipeline-dependency-profiling`

#### Worked Example

Simulation: `[Pipe Overlap Ratio]` %(MTE2&VECTOR/MTE2)=0.00%, %(MTE2&VECTOR/VECTOR)=0.00%. `[WAIT_FLAG / BAR Sync]` WAIT_FLAG total=961. `[Pipeline Flows]` only SCALARToMTE3 flow present — no MTE2ToVECTOR or VECTORToMTE2 flows.

The kernel loads and computes in strict serial order with zero overlap. The 961 WAIT_FLAG events confirm the pipeline is stalling on every fetch.

Fix: add `num_stages` sweep [2, 3, 4] to autotune configs. After fix: MTE2&VECTOR overlap > 10%, WAIT_FLAG total reduced significantly.

------

### A-Cat-2: Scalar Overhead Dominance

#### Simulation Signature

| Metric | Threshold | report.txt section |
|---|---|---|
| SCALAR instruction % | > 80% of total instructions | overall `[Pipe Distribution]` SCALAR instr% |
| SCALAR:VECTOR instruction ratio | > 4:1 | overall `[Key Ratios]` SCALAR:VECTOR_instr |
| SCALAR cycles% | > 70% of total cycles | overall `[Pipe Distribution]` SCALAR cycles% |
| TRACE total events | > 10,000 | overall `[TRACE Events]` Total events |
| SCALAR&VECTOR overlap (VECTOR side) | > 80% | overall `[Pipe Overlap Ratio]` %(SCALAR&VECTOR/VECTOR) |

#### Matching Rule

Read from `report.txt` overall:
- **Primary trigger:** overall `[Pipe Distribution]` SCALAR instr% > 80%
- **Confirmation:** overall `[Key Ratios]` SCALAR:VECTOR_instr > 4:1 OR overall `[Pipe Distribution]` SCALAR cycles% > 70%
- **Fire when:** Primary trigger AND at least 1 confirmation condition are met.
- **Differentiation from A-Cat-6:** A-Cat-6 has SCALAR cycles% high but instr% may be moderate — the cycles come from spill code, not from inherently scalar-dominant instruction mix. A-Cat-2 has both instr% AND cycles% high. If SCALAR instr% > 80% and cycles% > 70%, A-Cat-2 is the primary signal. If SCALAR cycles% > 50% but instr% < 60%, check A-Cat-6.
- **Differentiation from scalar-vector-simulation-signal A-Cat-1 (Scalar Arithmetic Explosion):** That signal focuses on scalar arithmetic code structures. This signal focuses on the autotune remedy: scale up tile sizes to amortize scalar overhead.
- **Differentiation from atomic-add kernels:** When the kernel contains `tl.atomic_add` (or any atomic update path), A-Cat-2's SCALAR > 80% can be a false positive. The scalar overhead comes from per-program address computation (`pid * BLOCK_SIZE + tl.arange`, mask computation), not from per-element scalar work that would be amortized by a larger block. Larger `BLOCK_SIZE` on the atomic-update dimension *increases* atomic contention (more threads compete for the same output cell), making performance worse. If the kernel contains `tl.atomic_add`:
  - Do NOT scale up the atomic-update dimension's `BLOCK_*` blindly. Include small values `[128, 256, 512, 1024]` in configs.
  - Avoid `BLOCK_SIZE > 4096` on the atomic-update dimension — large blocks amplify atomic contention and trigger disproportionate JIT compilation cost (see Global Constraint 4).
  - Include the atomic-branch variable (e.g. `ACCUMULATE`) in `key` so atomic vs non-atomic paths get separate cache lines (see A-Cat-4 Manifestation C).

#### What It Means

Fundamental tiling parameters are too small. The non-reducible hardware scalar overhead required to maintain loop steps, calculate indices, and execute pointer jumps eclipses the execution time of actual matrix calculations. The high SCALAR&VECTOR overlap (VECTOR side > 80%) confirms that VECTOR work is almost entirely shadowed by SCALAR — the VECTOR unit is starved because SCALAR can't feed it fast enough.

#### Code Manifestations

##### Manifestation A: Tiny BLOCK sizes causing loop-dominated execution

```python
# detect: BLOCK_M=32, BLOCK_N=32 — many loop iterations, little compute per iteration
triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4)
```

```python
# generic transform: scale up tile sizes
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4)
```

##### Manifestation B: Fine-grained loop with single-element operations

```python
# detect: tl.static_range loop with single-element work
for i in tl.static_range(0, N):
    val = tl.load(ptr + i, ...)     # one scalar load per iteration
    acc += val
```

```python
# generic transform: load contiguous tile, use vector operations
offs = tl.arange(0, BLOCK)
vals = tl.load(ptr + offs, ...)     # vectorized load
acc = tl.sum(vals)                   # vector reduction
```

#### Optimization Direction

Aggressively scale up the macro-tiling sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) via autotune configurations to maximize mathematical density per block loop. Follow the Global Constraints: keep powers of 2, watch SRAM budget.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
```

#### Related Patterns

- `scalar-latency-traps`
- `tiling`

#### Worked Example

Simulation: `[Pipe Distribution]` SCALAR instr=125 (85.0%), VECTOR instr=2 (1.4%). `[Key Ratios]` SCALAR:VECTOR_instr = 62.5:1. `[Pipe Overlap Ratio]` %(SCALAR&VECTOR/VECTOR)=83.38% — VECTOR is almost entirely serialized behind SCALAR.

The kernel (Average_Pooling_2D) has BLOCK_SIZE too small relative to the scalar address computation per element. The SCALAR unit spends 85% of instructions computing addresses, leaving only 1.4% for actual VECTOR work.

Fix: scale up BLOCK_SIZE from 32→128 in autotune configs to amortize scalar overhead across more vector work per iteration.

------

### A-Cat-3: Parallelism Starvation

#### Profiling Signature

| Metric | Threshold | Data Source |
|---|---|---|
| Grid / Core Ratio | Total launched Grid blocks ≪ physical core count (e.g., Grid < core_count / 2) | **NOT in `report.txt`** — requires msprof `op_summary_*.csv` or hardware-level profiling |
| Hardware State | Performance low, but `report.txt` shows neither A-Cat-1 (MTE2&VECTOR overlap > 5%) nor A-Cat-2 (SCALAR instr% < 80%) | `report.txt` can rule out A-Cat-1/2 but cannot directly confirm A-Cat-3 |

> **Critical note on data source:** `report.txt` is a single-program simulation. It has no visibility into multi-core Grid occupancy, total launched blocks, or physical core count. A-Cat-3 requires hardware profiling data (msprof) or cross-validation against benchmark results. **Do NOT fire A-Cat-3 from `report.txt` alone.** If `report.txt` rules out A-Cat-1 and A-Cat-2 but performance remains low, flag A-Cat-3 as a hypothesis to validate with hardware profiling.

#### Matching Rule

- **Precondition:** A-Cat-1 and A-Cat-2 have been ruled out from `report.txt` data
- **Hardware confirmation (REQUIRED):** msprof shows Grid blocks launched < physical core count / 2
- **Fire when:** Precondition AND hardware confirmation are both met. This signal CANNOT be confirmed from `report.txt` alone.

#### What It Means

Macro-concurrency starvation. When macro-tiling dimensions (`BLOCK_M`, `BLOCK_N`) are oversized relative to the matrix dimensions, too few Grid blocks are created. Increasing `num_warps` will NOT resolve this — it only scales intra-block concurrency within an individual core, leaving other physical compute cores entirely idle.

Grid Size = ceil(M / BLOCK_M) × ceil(N / BLOCK_N)

#### Code Manifestations

##### Manifestation A: Oversized blocks for small problem dimensions

```python
# detect: BLOCK_M and BLOCK_N are close to problem dimensions
# M=512, N=256, BLOCK_M=256, BLOCK_N=256 → Grid = ceil(512/256)*ceil(256/256) = 2
```

```python
# generic transform: reduce BLOCK_M/BLOCK_N for small shapes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
```

##### Manifestation B: Large K dimension with small M,N — Split-K candidate

```python
# detect: M and N are small but K is large, Grid = 1 regardless of tiling
```

```python
# generic transform: implement split-K work decomposition
# Partition K across grid, accumulate partial sums in a second reduction kernel
```

#### Optimization Direction

1. **Sub-divide Macro Tiling:** Scale down `BLOCK_M` and `BLOCK_N` to multiply the total number of launched Grid blocks, ensuring every physical hardware core receives workload slices.
2. **Implement Split-K Work Decomposition:** For shapes where M and N are small but the reduction dimension K is massive, partition K across the execution grid (Split-K), and accumulate partial blocks in a secondary reduction step.

```python
# Strategy 1: Decrease block sizes to increase macro grid occupancy across cores
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
```

#### Related Patterns

- `split-k`
- `tiling`

------

### A-Cat-4: Autotune Key Mismatch

#### Profiling Signature

| Metric | Threshold | Data Source |
|---|---|---|
| Performance Decay | Operator achieves extreme peak efficiency on Shape A, but encounters severe degradation or execution faults on Shape B | **NOT in `report.txt`** — requires multi-shape benchmark data |
| Trace Analysis | Autotune cache queries register a false "hit", applying a config optimized for large matrices onto narrow boundary shapes | **NOT in `report.txt`** — requires autotune cache trace or benchmark logs |

> **Critical note on data source:** `report.txt` profiles a single kernel invocation with a single shape. It cannot detect cross-shape performance degradation. A-Cat-4 requires benchmark data across multiple input shapes. If benchmark data is not available, extend the `key` parameter as a defensive measure for any kernel that handles diverse shapes.

#### Matching Rule

- **Primary trigger (benchmark):** Performance on Shape B / Performance on Shape A < 0.5 (2x+ degradation) despite both shapes being within the valid range
- **Confirmation:** Autotune cache shows same config applied to both shapes
- **Fire when:** Primary trigger AND confirmation are both met. This signal CANNOT be confirmed from `report.txt` alone.

#### What It Means

The `key` parameter list declared in `@triton.autotune` fails to encapsulate the full geometric or structural variables of the incoming tensors. Shapes with completely different memory strides or aspect ratios end up sharing the same autotune cache line.

#### Code Manifestations

##### Manifestation A: Key only includes geometric dimensions, not strides

```python
# detect: key lacks stride dimensions
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K']     # missing stride info → cache collision
)
```

```python
# generic transform: add stride keys for layout-sensitive dimensions
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K', 'stride_a_m', 'stride_b_k']
)
```

##### Manifestation B: Boundary shapes sharing cache with bulk shapes

```python
# detect: single-dimension key causes cache collision between tiny and huge shapes
@triton.autotune(
    configs=[...],
    key=['N']     # N=32 and N=32768 get same config
)
```

```python
# generic transform: add bucket dimensions or use more granular keys
@triton.autotune(
    configs=[...],
    key=['N', 'BLOCK_SIZE']     # BLOCK_SIZE in key isolates different configs per shape
)
```

##### Manifestation C: Path-branch variable not in key

When a kernel has an `if`-branch on a `tl.constexpr` variable (e.g. `ACCUMULATE`, `mode`, `quant_mode`) leading to different optimal `BLOCK_SIZE` per branch, the branch variable must be in `key`. Otherwise, atomic vs non-atomic paths (or different mode paths) share the same cache line, and one of them gets a suboptimal `BLOCK_SIZE`.

```python
# detect: kernel has if-branch on a constexpr leading to different optimal BLOCK_SIZE
if ACCUMULATE:
    tl.atomic_add(...)   # optimal BLOCK_SIZE=128 (contention-limited)
else:
    tl.store(...)        # optimal BLOCK_SIZE=1024 (throughput-limited)

# fix: include the branch variable in autotune key
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["n_elements", "ACCUMULATE", "ELEM_BYTES"],  # ACCUMULATE separates atomic vs non-atomic cache
)
```

This pattern is especially common in:
- `index_put` / `scatter_add` with `accumulate` flag (atomic vs non-atomic path)
- `quantize` with `mode` flag (per-row vs per-tensor path)
- `interpolate` with `mode` flag (bilinear vs bicubic vs nearest path)

For all of these, the branch variable must be in `key`, otherwise one path will get the wrong `BLOCK_SIZE`.

#### Optimization Direction

Expand the autotune isolation cache keys to ensure structural variations (shapes, memory layout strides) are explicitly isolated.

```python
# Incorporate geometric dimensions alongside leading memory layout strides
@triton.autotune(
    configs=[...],
    key=['M', 'N', 'K', 'stride_a_m', 'stride_b_k']
)
def matrix_multiply_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_a_m, ...):
    ...
```

#### Related Patterns

- `software-pipeline`

------

## Phase 3: Config Routes (After Signal Diagnosis)

Once a Category has been diagnosed and the parameter tuning direction is clear, choose the least manual autotune configuration route that fits the kernel.

### Route 1: Automatic Autotune (`configs=[]`)

Use `configs=[]` first when the parser can infer split and tiling structure directly from the kernel DSL. Let the diagnosed Category guide the search direction, but let the compiler generate the search space automatically.

Typical signals:

- the free tuning parameters are `tl.constexpr` values not fixed at launch time
- split parameters come from `tl.program_id`
- tiling parameters come from `tl.arange` or loop step structure
- masks or bounds expressions map cleanly back to runtime shape axes

```python
@triton.autotune(
    configs=[],
    key=["n_rows"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < n_rows
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(y_ptr + offs, x, mask=mask)
```

### Route 2: Add `hints` Before Hand-Writing Configs

Use `hints` when the kernel still fits auto-generated search, but parser inference is incomplete. The diagnosed Category tells you what direction to tune — `hints` tells the parser how to map kernel parameters to axes.

Typical signals:

- the split or tiling parameter is semantically clear to a human reviewer
- the path from `program_id` or `tl.arange` to masks is indirect
- low-dimensional or reduction axes need to be stated explicitly

When using `hints`, prefer axis-named `key` mappings so the cache aligns with the hinted axes.

```python
@triton.autotune(
    configs=[],
    key={"x": "n_rows", "y": "n_cols"},
    hints={
        "split_params": {"x": "BLOCK_M"},
        "tiling_params": {"y": "BLOCK_N"},
        "low_dim_axes": ["y"],
        "reduction_axes": [],
    },
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]

    for n0 in range(0, n_cols, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)[None, :]
        mask = (offs_m < n_rows) & (offs_n < n_cols)
        x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
        tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

### Route 3: Hand-Write `triton.Config` Candidates

Use explicit `triton.Config` lists when the search space must be constrained manually. **Each Category above already provides concrete Config examples for this route.** Use those as starting points.

Typical signals for Route 3:

- one grid axis is fixed by semantics and cannot be freely split
- one parameter couples launch count and inner tile shape
- the kernel exposes too little clean tuning structure for automatic generation
- candidate quality is still poor after adding `hints`

On Triton-Ascend, the main hand-written search dimensions should usually be:

- `BLOCK_*` sizes
- `multibuffer`
- `unit_flag` when relevant

Do not import GPU-first search habits blindly. `num_warps` and `num_stages` are not the primary Ascend tuning knobs — only include them when the signal diagnosis (A-Cat-1 or A-Cat-6) specifically points to pipeline or register issues.

```python
def get_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "multibuffer": mb})
        for bm in [256, 128, 64, 32]
        for bn in [128, 64, 32, 16]
        for mb in [True, False]
    ]


@triton.autotune(
    configs=get_configs(),
    key=["n_rows", "n_cols"],
)
@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = tl.arange(0, BLOCK_N)[None, :]
    mask = (offs_m < n_rows) & (offs_n < n_cols)
    x = tl.load(x_ptr + offs_m * n_cols + offs_n, mask=mask, other=0)
    tl.store(y_ptr + offs_m * n_cols + offs_n, x, mask=mask)
```

### When Automatic Parsing Usually Fails

Prefer `hints` or custom configs when you see one or more of the following:

- the kernel has no meaningful free `tl.constexpr` parameters because they are fixed at launch or coupled too tightly to semantics
- no clear mask or bounds relation back to the runtime axis
- one parameter must cover an entire semantic dimension, such as `BLOCK_SIZE >= hidden_dim`
- a business or semantic rule fixes one grid dimension instead of allowing free tiling
- one parameter influences multiple axes at once

## What To Verify After Applying

- Verify the chosen route is the least manual one that still fits the kernel:
  - `configs=[]` first when parser inference should succeed
  - `hints` when semantics are clear but inference is incomplete
  - explicit `triton.Config` lists only when the search space truly needs manual control
- Verify `key` tracks the runtime shape arguments that actually change the best configuration.
- Verify update-style kernels use `reset_to_zero`, `restore_value`, hooks, or equivalent safeguards so repeated autotune trials do not corrupt outputs.
- Verify the searched parameters are Ascend-relevant for the config-space search, especially `BLOCK_*`, `multibuffer`, and `unit_flag`, rather than treating GPU-only defaults such as `num_warps` or `num_stages` as the default search surface.
- Verify the selected block sizes still satisfy semantic constraints such as `BLOCK_SIZE <= tiled logical extent` when padding would otherwise change results.
- Verify `TRITON_PRINT_AUTOTUNING=1` or equivalent logs show the inferred axes, candidate count, and chosen best configuration during debugging.

## Ascend-Specific Notes

- Default config-space search should focus on `BLOCK_*`, `multibuffer`, and `unit_flag`, not treat `num_warps` or `num_stages` as the default Ascend autotune surface.
- When launch hints interact, include a small bounded set of Ascend-relevant options such as `multibuffer`, `set_workspace_multibuffer`, or `enable_auto_bind_sub_block` instead of hand-picking one globally.
- If you are applying `a5-force-simt-only-discrete-access`, recheck `num_warps` and grid decomposition there after enabling `force_simt_only=True`.
- For update-style kernels, repeated autotune evaluation can write outputs multiple times. Add `reset_to_zero`, `restore_value`, `pre_hook`, or `post_hook` before trusting benchmarks. Avoid `num_warps=N` per-config on atomic-add kernels — it triggers the `_pre_hook` path on Ascend, forcing keyword argument passing that cascades into `BLOCK_SIZE: tl.constexpr` signature requirements.
- Start debugging with `TRITON_PRINT_AUTOTUNING=1`.

## Related Patterns

- `tiling`: use it first when the kernel still needs a better tiled structure before any search space should be explored.
- `software-pipeline`: use it when the tile structure is already good and the next issue is overlap quality rather than parameter choice.
- `software-pipeline-dependency-profiling`: use it when `report.txt` [Pipe Overlap] suggests weak overlap but structural loop issues may block pipelining.
- `compile_hint`: use it when Pre-Gate A-Cat-5 fires due to compiler information deficit rather than truly broken layout.
- `discrete_memory_access`: use it when Pre-Gate A-Cat-5 fires and memory access is inherently index-driven.
- `scalar-latency-traps`: use it when A-Cat-2 fires and the scalar overhead has identifiable code-pattern root causes.
- `a5-force-simt-only-discrete-access`: use it when A5 is confirmed and the kernel is discrete-memory-access dominated; that launch-mode experiment intentionally rechecks `num_warps` and grid decomposition after `force_simt_only=True`.
