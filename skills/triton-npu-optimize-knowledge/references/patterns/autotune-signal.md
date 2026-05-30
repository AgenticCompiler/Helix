# Autotune Signal: Tuning by Hardware Profiling and Simulation Analysis

## Summary

Match observed statistical features in profiling reports (`sim_features.txt` or execution trace logs) against structured categories to identify resource imbalances, then use `@triton.autotune` to dynamically search for the optimal tiling sizes, software pipelining stages, and warp counts that resolve those specific hardware bottlenecks.

## Use When

- The kernel logic is mathematically correct, stable, and passes validation tests.
- Structural profiling reports (msprof `op_summary_*.csv`, `sim_features.txt`, or execution trace logs) indicate resource under-utilization or latency-hiding failures.
- You need to fine-tune the ratio of memory fetching (MTE) to execution units (CUBE/VECTOR) across varying input tensor dimensions.
- The kernel structure already looks semantically correct, and the likely headroom is in `BLOCK_*` selection, `num_warps`, `num_stages`, or autotune `key` configuration.
- Profiling shows one or more of: near-0% MTE/CUBE overlap, SCALAR dominance, Grid/Core ratio mismatch, shape-dependent performance degradation, register spilling, or fragmented memory loads.

## Avoid When

- The core algorithmic logic or global memory layouts are undergoing major design refactoring — stabilize the kernel structure first.
- The pipeline is entirely bound by global memory bandwidth limits (compute cycles fully hidden behind continuous MTE transfers) — autotune has hit its physical ceiling; operator fusion or lower precision (FP32 → FP16/BF16) is required instead.
- All relevant `tl.constexpr` parameters are already fixed at launch time with no meaningful tuning space.

## Evidence To Confirm

- Profiling overlap ratio between memory transfer cycles (MTE) and execution core cycles (CUBE/VECTOR) is near 0%, or `WAIT_FLAG` execution stalls dominate the timeline.
- SCALAR control instructions (loop branching, pointer arithmetic) heavily outnumber vector/matrix math instructions in the hot loop.
- The total number of launched Grid blocks is significantly lower than the physical core count of the target hardware, while neither memory stalls nor scalar overhead are the primary issue.
- Performance degrades significantly on specific matrix shapes or boundary conditions while working optimally elsewhere, suggesting autotune cache key mismatch.
- Average `ProcessBytes` per MTE load is abnormally small (< 128 bytes), or IR reveals fragmented scalar memory loads instead of wide vectorized blocks.
- Massive spike in local memory read/write instructions appears without significant `WAIT_FLAG` stalls, suggesting register spilling rather than pipeline overlap issues.

## Related Patterns

- `autotune`
- `compile_hint`
- `software-pipeline`
- `tiling`

## Common Non-Matches

- The core algorithmic logic or global memory layouts are undergoing major design refactoring — stabilize the kernel structure first.
- The pipeline is entirely bound by global memory bandwidth limits (compute cycles fully hidden behind continuous MTE transfers) — autotune has hit its physical ceiling; operator fusion or lower precision (FP32 → FP16/BF16) is required instead.
- All relevant `tl.constexpr` parameters are already fixed at launch time with no meaningful tuning space.

## Global Constraints

Before outputting any code modifications or autotune configurations, the following rules must be satisfied:

1. **Hardware Alignment Enforcement:** All tiling sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) must be powers of 2. Inspect the kernel for underlying hardware alignment constraints or static assertions (e.g., `tl.static_assert(BLOCK_K % 16 == 0)`). Generated configurations must never violate these physical boundaries.
2. **SRAM (UB) Resource Trade-Off:** Tiling sizes and software pipelining stages compete for the same limited on-chip memory (SRAM/Unified Buffer). Total shared memory utilization scales proportionally with O(BLOCK_M × BLOCK_K × num_stages). If increasing `num_stages` to hide latency triggers an out-of-memory (OOM) error or causes register spilling, scale down `BLOCK_K` or other tile dimensions to compensate.
3. **Data-Driven Logic:** Base all performance reasoning purely on concrete metrics: instruction cycle ratios, `WAIT_FLAG` execution frequencies, `ProcessBytes` averages, and memory tracking metrics. Abandon qualitative visual descriptions.

## Signal Matching Decision Guide

Analyze profiling metrics in the following strict hierarchy. Start from the top and stop at the first match:

1. **Cat 5 — Check Memory Layout & Contiguity:** Fragmented loads? Low average `ProcessBytes` per MTE fetch despite massive total MTE counts? → **Cat 5 (Memory Fragmentation & Hint Deficit)**
2. **Cat 6 — Check Register Boundaries:** Sudden performance drops with high compute cycles but minimal memory blockages? Presence of stack-allocated local memory movements? → **Cat 6 (Register Spilling)**
3. **Cat 1 — Check Pipeline Overlap:** Near 0% cycle overlap between memory units (MTE) and execution units (CUBE/VECTOR)? Massive `WAIT_FLAG` instruction stalls? → **Cat 1 (Pipeline Overlap Deficit)**
4. **Cat 2 — Check Scalar Overhead:** SCALAR instructions heavily dominate execution cycles? Inner loop executing an excessive number of short iterations? → **Cat 2 (Scalar Overhead Dominance)**
5. **Cat 3 — Check Hardware Parallelism:** Latency hiding failing with low overall throughput? Number of launched Grid blocks vastly smaller than physical hardware Core count? → **Cat 3 (Parallelism Starvation)**
6. **Cat 4 — Check Cross-Shape Generalization:** Performance degrades significantly on specific matrix shapes or boundary conditions while working optimally elsewhere? → **Cat 4 (Autotune Key Mismatch)**

---

## Signal Category 5: Memory Fragmentation & Hint Deficit

### Profiling Signature

| Metric | Threshold / Feature |
|---|---|
| **ProcessBytes Stat** | Average `ProcessBytes` per MTE load is abnormally small (e.g., < 128 bytes) |
| **Instruction Gen** | Compiled assembly or IR reveals fragmented scalar memory loads instead of wide vectorized memory blocks |
| **Host-Side Layout** | Input tensors are passed into the kernel with heavy, non-contiguous strides (un-coalesced dimensions or unaligned transpositions) |

### Physical Cause

The compiler cannot statically verify that the memory access pattern is perfectly contiguous along the vectorization axis. It generates conservative, fragmented, narrow scalar loads. Autotune configurations cannot resolve a fundamentally broken physical layout.

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

---

## Signal Category 6: Register Spilling

### Profiling Signature

| Metric | Threshold / Feature |
|---|---|
| **Local Memory Activity** | Massive spike in local memory read/write instructions (excessive internal text labels indicating scratchpad spillages or stack data movement) |
| **Compute Cycle Penalty** | CUBE or VECTOR engine execution cycles skyrocket unexpectedly while total compute operations remain mathematically identical |
| **Pipeline State** | No significant `WAIT_FLAG` stalls are observed (ruling out Cat 1), yet execution time stalls |

### Physical Cause

The architectural resource limit of vector or scalar registers per thread block has been exceeded. When `BLOCK_SIZE` or `num_warps` are over-provisioned, the compiler is forced to spill excess variables out of high-speed registers into slow off-chip local memory structures.

### Optimization Direction

De-escalate register pressure by tightening block boundaries or optimizing block-level concurrency.

1. **Reduce Tiling Geometry:** Scale down `BLOCK_M`, `BLOCK_N`, or `BLOCK_K` to shrink the overall tensor block size residing in registers simultaneously.
2. **Re-evaluate Warp Allocation:** Adjust `num_warps`. Increasing warps can sometimes split register pressure across more threads, but excessive warps can also trigger collective spillages depending on compiler allocation bounds.

---

## Signal Category 1: Pipeline Overlap Deficit

### Profiling Signature

| Metric | Threshold / Feature |
|---|---|
| **Overlap Ratio** | Temporal execution overlap between memory transfer cycles (MTE) and execution core cycles (CUBE/VECTOR) is near 0% |
| **FLOWCTRL Stat** | Abnormally high concentration of blocking `[WAIT_FLAG]` markers in the execution logs (e.g., > 50-100 execution stalls per loop chunk) |

### Physical Cause

The execution pipeline suffers from shallow multi-buffering depth. The system executes in strict serial cadence ("fetch → wait → compute → fetch next"), failing to asynchronously prefetch memory blocks for future loop iterations while computing the current block.

### Optimization Direction

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

---

## Signal Category 2: Scalar Overhead Dominance

### Profiling Signature

| Metric | Threshold / Feature |
|---|---|
| **Instruction Ratio** | SCALAR control instructions (loop branching, pointer arithmetic, base address offsets) heavily outnumber vector/matrix math instructions |
| **Loop Counter** | Execution trace shows an excessive number of loop iterations, indicating very small tiling steps relative to the global data shape |

### Physical Cause

Fundamental tiling parameters are too small. The non-reducible hardware scalar overhead required to maintain loop steps, calculate indices, and execute pointer jumps eclipses the execution time of actual matrix calculations.

### Optimization Direction

Aggressively scale up the macro-tiling sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) via autotune configurations to maximize mathematical density per block loop.

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

---

## Signal Category 3: Parallelism Starvation

### Profiling Signature

| Metric | Threshold / Feature |
|---|---|
| **Grid / Core Ratio** | Total number of launched Grid blocks is significantly lower than the physical core count of the target hardware device |
| **Hardware State** | Execution performance is low, but profiling shows neither significant memory access stalls (rules out Cat 1) nor scalar overhead dominance (rules out Cat 2) |

### Physical Cause

Macro-concurrency starvation. When macro-tiling dimensions (`BLOCK_M`, `BLOCK_N`) are oversized relative to the matrix dimensions, too few Grid blocks are created. Increasing `num_warps` will NOT resolve this — it only scales intra-block concurrency within an individual core, leaving other physical compute cores entirely idle.

Grid Size = ceil(M / BLOCK_M) × ceil(N / BLOCK_N)

### Optimization Direction

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

---

## Signal Category 4: Autotune Key Mismatch

### Profiling Signature

| Metric | Threshold / Feature |
|---|---|
| **Performance Decay** | Operator achieves extreme peak efficiency on Shape A, but encounters severe degradation or execution faults on Shape B |
| **Trace Analysis** | Autotune cache queries register a false "hit", applying a tuning configuration optimized for massive matrices onto narrow or highly specific boundary shapes |

### Physical Cause

The `key` parameter list declared in `@triton.autotune` fails to encapsulate the full geometric or structural variables of the incoming tensors. Shapes with completely different memory strides or aspect ratios end up sharing the same autotune cache line.

### Optimization Direction

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
