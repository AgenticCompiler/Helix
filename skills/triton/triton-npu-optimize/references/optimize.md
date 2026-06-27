# Ascend Triton Kernel Performance Optimization Methodology

> This methodology document applies to **any** Ascend Triton kernel operator, not a specific one. Follow the steps in order during optimization.

---

## Step 1: Read Performance Data

The following files describe the performance characteristics of the current kernel. **All must be read first**:
1. `report.txt` — Pipeline distribution, source hotspots, synchronization statistics, TRACE events
2. `dataType_4_API_INSTR.json` — Per-instruction pipe, cycles, GPR count, UB conflict
3. `dataType_3_API_FILE.json` — Mapping between source line numbers and instruction addresses

---

## Step 2: Identify Bottlenecks (extract from report.txt)

Check each metric below. **Every item must have an analysis conclusion**:

| # | Metric | Data Source | Criteria |
|---|--------|-------------|----------|
| 1 | Pipeline bottleneck | `[Pipe Distribution]` | A pipe's cycles% is **significantly higher** than its instr% (e.g., SCALAR instr=76% but cycles=89%) → that pipe is the bottleneck |
| 2 | Hottest source lines | `[Source Code Info]` | Sort by Cycles descending; identify Top-3. Also record each line's Pipe distribution |
| 3 | High-latency ops | `[TRACE Events]` | Sort by count descending; watch for REM, DIV, MUL, SIGNEXT and other high-latency scalar ops |
| 4 | Synchronization overhead | `[WAIT_FLAG / BAR Sync]` | Total WAIT_FLAG > 5000 or total BAR > 10000 → sync bottleneck. Also check inter-pipe flow direction and count |
| 5 | Vector unit efficiency | `[VECTOR Unit]` | UB Read/Write Conflict > 0 → bank conflict exists. Utilization < 1% → vector pipe underutilized |
| 6 | Register pressure | GPR Count at LineNo:0 in `[Source Code Info]` | GPR Count > 28 → high register pressure, compiler generated spill code |

---

## Step 3: Check Each Optimization Opportunity (⚠️ must evaluate every item, none may be skipped)

For the 7 categories below, **evaluate each one individually**. Every item requires a conclusion: "Applied" or "Not applicable, reason: ___".

---

### Optimization 1: Reduce Kernel Launch Count

**Detection method**:
- Check whether the operator entry function (host side) calls `kernel[grid](...)` multiple times
- Check for Python `for` loops that contain kernel launches
- Check for multiple independent kernel launches on the same input data

**Action**: If launches can be merged into one → merge.

---

### Optimization 2: Reduce GM (Global Memory) Transfers

**Detection method**:
- Does the host create intermediate tensors solely for passing data between kernels (`torch.empty` → kernel write → kernel read)?
- Are there intermediate GM reads/writes that could be eliminated through kernel fusion?
- Does the kernel perform repeated loads from the same GM address (consider prefetching into registers or shared memory)?

**Action**: If intermediate tensors can be eliminated or GM access count reduced → apply.

---

### Optimization 3: Fast Path / Slow Path Separation (⚠️ highest priority, must not skip)

**Core idea**: Promote runtime conditions repeatedly evaluated inside the kernel's inner loops to the host side for one-time static determination, dispatching to a simplified kernel.

**Detection method** (scan kernel source in this order):

1. **Bounds-check conditions**:
   - Does the kernel contain bounds-clamping logic like `idx >= 0`, `idx < size`?
   - Does it use safe indexing patterns like `tl.where(bound_cond, real_idx, safe_idx)`?
   - Can these conditions be statically determined as always-true on the host side based on input parameters (shapes, padding, stride, etc.)?
   - → If yes: formulate the condition, implement `_can_skip_bounds_check()` on the host side

2. **Parameter specialization conditions**:
   - Does the kernel have parameters such as stride, dilation, scale, etc.?
   - When these parameters take special values (e.g., 1), can multiplications/divisions in the kernel be eliminated?
   - → If yes: create a fast kernel for the special case, omitting the corresponding address calculations

3. **Loop-internal conditional branches**:
   - Does the loop body contain `if condition` or `tl.where(condition, ...)`?
   - Does the `condition` depend on runtime variables but is actually constant for the majority of inputs?
   - → If yes: move the condition to a kernel parameter (`tl.constexpr`) or the host dispatch layer

4. **Configurable feature flags**:
   - Does the kernel have patterns like `if RETURN_INDICES`, `if TRAINING`, etc.?
   - Even if already using `tl.constexpr`, are unnecessary variables still allocated (e.g., initialization for non-required paths)?
   - → If yes: move variable declarations inside the conditional branch as well

**Action template**:
```
1. Implement _can_use_fast_path(params...) -> bool on the host side
2. Clone the original kernel to create _kernel_fast
3. In _kernel_fast, remove all bounds checks, safe indexing, conditional branches,
   and unnecessary variable initialization that are redundant under the fast path
4. Host dispatches to _kernel_fast or _kernel_slow based on _can_use_fast_path()
```

---

### Optimization 4: Eliminate Redundant Instructions

**Detection method** (check each instruction in the kernel loop body):

1. **Masked load + tl.where with the same mask**:
   ```python
   val = tl.load(ptr + offsets, mask=M, other=X)
   val = tl.where(M, val, Y)          # ← Redundant if subsequent uses are already guarded by M
   ```
   Check: Are all subsequent uses of `val` protected by the same `M` (or a subset of it)?

2. **Consecutive write overwrites**:
   ```python
   result = tl.where(cond_a, val_a, default)
   result = tl.where(cond_b, val_b, result)   # ← If cond_b covers all True cases of cond_a, the first is redundant
   ```
   Check: Relationship between the two conditions (mutually exclusive / subset / overlap)?

3. **Unused intermediate values**:
   - Is a value computed inside the loop guaranteed to be reassigned before being read?
   - → Delete the computation of that value

---

### Optimization 5: Reduce Instruction Count

**Detection method**:

1. **Combine mergeable operations**:
   - `(a > b) & (a > c)` → `a > tl.maximum(b, c)` or `a > b` if `b >= c` is already guaranteed
   - Consecutive `x + a + b + c` → `x + (a + b + c)` (constant folding)
   - Multiple `tl.where` on the same target without overwrite relationship → can they be merged into a single comparison?

2. **Repeated computation inside loops**:
   - Are there sub-expressions inside the loop body that are invariant in the immediately enclosing outer loop?
   - Note: The Ascend compiler already performs basic hoisting. Only manually promote complex expressions the compiler cannot infer

3. **Unnecessary init + update patterns**:
   - `best = tl.full([N], -inf)` + `best = tl.where(update, new_val, best)`
   - Can `tl.maximum` replace the manual compare+where pattern?

---

### Optimization 6: Avoid Unnecessary Type Promotion

**Detection method**:

1. **Redundant precision conversion**:
   - `tl.load(...).to(tl.float32)` — when the operator entry knows the input dtype is float32, `.to(tl.float32)` is redundant
   - Solution: control conversion via `INPUT_DTYPE_IS_FLOAT32: tl.constexpr` parameter

2. **Accumulator precision**:
   - Accumulator uses float32 but input is float16/bf16 — this is necessary (prevents overflow)
   - If input range is limited and operator semantics allow → consider maintaining input precision for accumulation

---

### Optimization 7: Optimize Instruction Composition (replace high-latency with low-latency)

**Detection method**:

1. **Division/modulo → bitwise**:
   - If divisor is a power of two → `a // 2^k` → `a >> k`; `a % 2^k` → `a & (2^k - 1)`
   - Note: The compiler typically already does this; only manually handle **non-constant** divisors

2. **Multiplication chains → precompute**:
   - If the loop contains `a * s0 + b * s1 + c * s2` patterns
   - Note: The compiler already does hoisting; only consider if the report shows high MADD counts

3. **Constraint**:
   - The replacement must not significantly increase total instruction count (1→1 or 1→2 is acceptable; 1→3+ is not)
   - Even if individual latency is lower, instruction bloat increases register pressure and may degrade overall performance

---

## Step 4: Prohibited Optimizations (cause degradation on Ascend Triton)

| # | Prohibited Item | Reason | Affected Scenario |
|---|-----------------|--------|-------------------|
| 1 | Manual hoisting of loop invariants | The Ascend compiler already performs equivalent optimization | Extracting invariant computation to outer loops |
| 2 | Large-scale `tl.static_range` unrolling | Instruction bloat multiplier = product of nested loop body sizes | Unrolling small loops |
| 3 | Splitting grid layout to eliminate division | Added sync/data-transfer overhead > cycles saved by eliminating division | Splitting 1D grid into 2D/3D to avoid `//` and `%` |
| 4 | Looking only at total instruction count | Instruction composition (per-instruction cost) is equally critical; 2 low-latency instructions may be worse than 1 high-latency | All scenarios |
| 5 | Adding too many intermediate variables | Increases register pressure, causing the compiler to generate more spill code | Splitting complex expressions into multiple intermediate variables |

---

## Step 5: Self-Check Checklist (must fill out every item after optimization)

Before submitting optimized code, confirm that every item below has been evaluated. **No item may be left blank**:

| # | Optimization Item | Assessment Conclusion |
|---|-------------------|-----------------------|
| 1 | Reduce kernel launch count | ☐ Applied / ☐ Not applicable, reason: ___ |
| 2 | Reduce GM transfers | ☐ Applied / ☐ Not applicable, reason: ___ |
| 3 | Fast path / slow path separation | ☐ Applied / ☐ Not applicable, reason: ___ |
| 4 | Eliminate redundant instructions | ☐ Applied / ☐ Not applicable, reason: ___ |
| 5 | Reduce instruction count | ☐ Applied / ☐ Not applicable, reason: ___ |
| 6 | Avoid unnecessary type promotion | ☐ Applied / ☐ Not applicable, reason: ___ |
| 7 | Optimize instruction composition | ☐ Applied / ☐ Not applicable, reason: ___ |
| — | No prohibited optimization techniques used | ☐ Confirmed |
| — | All modifications have correctness justification | ☐ Confirmed |

---

## Step 6: Key Metrics Comparison (Before vs. After Optimization)

| Metric | Expected Change | Warning Signal |
|--------|----------------|----------------|
| Total instruction count | ↓ or flat | Significant increase → code bloat |
| SCALAR cycles | ↓ | Primary bottleneck metric, should drop noticeably |
| VECTOR cycles | ↓ or flat | Auxiliary metric |
| Total WAIT_FLAG count | ↓ | Sync overhead; increase → extra sync introduced |
| Total BAR count | ↓ | Sync overhead |
| ProcessBytes | ↓ or flat | Increase → worse scheduling |
| Internal instruction count (LineNo:0) | ↓ or flat | Increase → higher register pressure |
| Inner loop cycles (hottest line in Source Code Info) | ↓ | Core optimization target |
