# Constexpr tile vs effective extent (indexed / masked loops)

## Summary

On **Ascend NPU**, Triton often lowers **masked “vector”** work into **scalar `scf.for`** loops whose **upper bound equals `tl.constexpr` tile sizes** (`BLOCK_SIZE`, `BLOCK_M`, `BLOCK_N`, …). When the **logical valid span** along that axis—**number of outputs**, **selected index length** (e.g. **index_select** / advanced indexing), **inner slice length**, etc.—is **smaller than the tile**, the lowered loop may still run the **full** trip count; correctness is restored with **masks** and **`arith.select`** against zeros, which is **wasted execution**. Choosing **smaller constexpr tiles on the host** and **recomputing grid / program counts** aligns loop bounds with the real workload. For **2D tiles**, **`min(hardware cap, live row/column extent)`** often removes masked padding along a narrow axis; **rounding up to the next power of two** is **not** universally faster—it can add **extra columns/rows** of masked work, so treat PO2 as a **JIT stability** option and **validate on device**. For **indexed writes** with **multiple launch shapes**, a common pattern is: choose an **inner** tile from **`cdiv(extent, b) * b`**, then if **`cdiv(inner, b) × (outer bundle)`** exceeds a **per-launch program limit**, **increase `b`** (fewer inner blocks) until under the cap or hit a max tile, else fall back to a **flat** kernel whose tile is chosen from the **flattened element count**—**flat** paths dominated by **`atomic_*`** may still see **little** gain from tile alone. MLIR **`{DiscreteMemAccess}`** and **`{ExtractedLoadOrStore}`** mark **per-lane** indexed **`memref.load`/`store`** (e.g. **`sizes [1]`**), not one coalesced DMA vector—**tile reduction cuts iterations**, it does not remove indexed semantics. **Profile the path that actually runs**.

## Use When

- Any kernel where **`tl.arange` / 2D tile** size is **`tl.constexpr`** but a **mask** or **valid count** shows the **active outputs or indices** are **much smaller** than that tile (including **index_select**, **gather-like** `tl.load(ptr + f(index))`, **scatter-like** `tl.atomic_add` / indexed stores, and similar **output-sized < loop-sized** patterns).
- Simulator or profiler shows **hot loops** or **LD/ST `call_count`** scaling with **full tile area** or **2048-style** constants, not with the **logical `numel`** of the case.
- MLIR shows **`scf.for`** upper bound tied to tile constants while **`DiscreteMemAccess`** / **`ExtractedLoadOrStore`** appear on extracted load/store paths.

## Signals

### Code

- **`BLOCK_SIZE`** or **`BLOCK_M`/`BLOCK_N`** are **large fixed constexprs** while **`mask = offsets < n_valid`** (or equivalent) has **`n_valid` ≪ tile** along that axis.
- **Multi-axis tiling**: product **`BLOCK_M * BLOCK_N`** dominates even when **one axis extent** (e.g. columns of the output slice) is smaller.
- **Host chooses different constexpr tiles** for **structured** (e.g. row × inner-block) vs **flat** one-dimensional launches when dispatch branches on **program count**.
- **Indexed globals**: **`tl.load` / `tl.store` / `tl.atomic_*`** with **dynamic byte offset** per lane, often paired with **per-lane masks**.

### Profile

- **`code_exe`**: time in **scalar loop bodies** or **indexed load/store** lines, not only in unrelated ops.
- **`instr_exe`**: **`LD_*` / `ST_*`** counts **≈ tile size per program** rather than **≈ valid element count**.

### IR

- **`scf.for %i = %c0 to %cN`** with **`N`** equal to a **tile constant** that does **not** shrink when the **logical extent** is smaller.
- **`tensor.extract`** / **`memref.load`** with **`DiscreteMemAccess`**, **`reinterpret_cast … sizes [1]`**.
- **`arith.select`** between computed values and **zero-filled** tensors along the tile (masked semantics).

## Problem Description

**Root cause**

- Lowering may keep **loop trip count = constexpr tile** even when **many iterations are masked out**; the backend does not always fold the mask into the bound.
- **Indexed** access patterns are lowered to **discrete** scalar memory ops—expected on NPU for those idioms.

**Symptoms**

- **Small shapes** pay **large per-kernel** time.
- After shrinking constexpr tile in IR, **dominant `scf.for` bound** and **memory-op counts** drop accordingly.

## Optimization Strategy

**Goal**: Align **`tl.constexpr` tile dimensions** and **launch geometry** with **effective extents** and **hardware limits**, minimizing **`ceil(extent / tile) * tile`**-style waste without breaking correctness or JIT.

### Key principles

1. **Identify the live path**: If multiple **`@triton.jit`** variants or host branches exist, **tile and grid heuristics apply to the variant that runs** for the measured shape—not to unused entry points.
2. **Measure one effective extent per tiled axis**: e.g. **output element count**, **index tensor length**, **slice width**—whatever the mask or semantic size ties to.
3. **Pick constexpr tile** from a **small discrete set** (often powers of two in a bounded range) using a simple cost proxy such as **`triton.cdiv(extent, b) * b`**; on ties, balance **wasted padding** vs **grid / program growth** (shape-dependent).
4. **Program / grid budget**: When launch uses **`cdiv(inner, BLOCK) * (outer product)`**, enforce **≤ device limit**; if over budget, **increase `BLOCK`** (fewer inner blocks) or **change dispatch**—do not tune tile in isolation from this product. A **two-step** host policy works well: first pick **`BLOCK`** from the **inner extent** using the same **`cdiv * b`** proxy as 1D tiles, then **double `BLOCK` toward a cap** while **`inner_blocks * outer_bundle` > limit`**; if still infeasible at max tile, use a **flat** launch whose **`BLOCK`** is chosen from **total flattened `numel`**, not from the inner extent alone.
5. **Non-power-of-two tiles** can remove tail waste but may **fail JIT** on some targets—validate.

### Generic modification patterns (not operator-specific)

| Concern | What to change |
|--------|----------------|
| **Oversized 1D tile** | Drive **`BLOCK_SIZE`** (or equivalent) from **`n_valid`** / inner extent so the lowered bound is **O(extent)**, not a legacy constant. |
| **Oversized 2D tile** | Set **`BLOCK_M` / `BLOCK_N`** from **live row/column (or last-axis) extents**, e.g. **`min(cap, out_cols)`** / **`min(cap, out_rows)`** so **`tl.arange`** span matches masked bounds—**prefer exact extent** when JIT allows; **next power-of-two** can **add** masked padding (benchmark both). |
| **Program limit + inner tile** | If **`programs = outer_bundle * cdiv(inner, BLOCK)`** exceeds **HW limit**, **raise `BLOCK`** until **`programs ≤ limit`** or **`BLOCK == max_tile`**, then pick **flat** vs structured dispatch; do not optimize padding only and ignore this product. |
| **Flat vs structured tile** | **Structured** launches: tile from **`inner_extent`**, then bump for limit. **Flat** launch over **`numel`**: use a **separate** **`effective_tile(numel)`** for **`grid`** and **`BLOCK_SIZE`**—do not reuse one constant when the two extents differ. |
| **Grid after tile change** | Recompute **`grid`** / **`num_programs`** / any **`cdiv(extent, BLOCK)`** so occupancy stays sane and limits are respected. |
| **Multi-kernel dispatch** | Encode **extent- and limit-based** branch conditions on the host so each kernel’s constexprs match the shapes that route to it. |
| **Atomics / reductions** | If **`atomic_*`** or **contention** dominates, **fewer masked iterations** may still **not** move end-to-end time—confirm in profile before relying on tile alone. |

### Implementation pitfall (host launch vs module default)

- A **module-level “max block”** (e.g. **2048**) is fine as an **upper cap**, but **no IR or trip-count win** appears until the **host launch** passes a **`tl.constexpr` tile derived from the live extent** (e.g. **`n_elements`**) and **`grid` uses the same tile**: e.g. **`tile = f(n_elements)`**, **`grid = (cdiv(n_elements, tile),)`**, **`kernel[grid](…, BLOCK_SIZE=tile)`**. If the code still does **`cdiv(n, MAX_BLOCK)`** with **`BLOCK_SIZE=MAX_BLOCK`**, small **`n`** keeps **full-tile** masked loops—this is a **launch-site omission**, not something the pattern doc replaces by editing only the JIT body.
- A practical default is a **short power-of-two sweep** between **`min_block`** and **`max_block`**, minimizing a cheap proxy such as **`cdiv(n, b) * b`** (total padded lane-work) and **tie-breaking toward larger `b`** when costs tie, so **grid** does not explode without intent.

## Detection Pattern

**Problematic (conceptual)**

```python
# Tile fixed at compile time; many lanes masked out
offs = pid * BLOCK + tl.arange(0, BLOCK)  # BLOCK large
mask = offs < n_valid  # n_valid much smaller than BLOCK
val = tl.load(src + g(offs), mask=mask, other=0)
```

**Directional fix (conceptual)**

```python
# Host: choose constexpr BLOCK so bounds match workload (e.g. next PO2 ≥ n_valid, or minimal cdiv*n proxy)
# Recompute grid = cdiv(n_valid, BLOCK) (and any fused axes).
```

**Program budget (conceptual)**

```python
# Inner tile from extent, then satisfy program cap (e.g. row-structured indexed writes).
BLOCK = effective_tile(inner_extent)  # e.g. PO2 sweep minimizing cdiv(inner, b) * b
bundle = outer_product  # dimensions folded into program axis
while cdiv(inner_extent, BLOCK) * bundle > HW_PROGRAM_LIMIT and BLOCK < MAX_BLOCK:
    BLOCK *= 2
# If still over limit at MAX_BLOCK, fall back to flat launch:
#   grid = (cdiv(numel, effective_tile(numel)),), BLOCK_SIZE = effective_tile(numel)
```

## DiscreteMemAccess and ExtractedLoadOrStore (IR)

**Why attributes appear**

- **Per-lane** **`tensor.extract`** + **`memref.load`/`store`** with **small `sizes`** reflect **indexed** global access, not a single wide vector transfer.
- **`ExtractedLoadOrStore`** on **`scf.for`**: body lowered from **masked tensor** code to **scalar** chains.

**After tile optimization**

- **`DiscreteMemAccess` often remains** for the **core indexed op**; gains come from **fewer iterations**, not from turning the op into contiguous DMA.
- For **staging / UB** alternatives (fewer discrete globals), see **[discrete_memory_access.md](discrete_memory_access.md)** when **workspace** allows.

## Avoid When

- Dominant cost is **outside** the tiled indexed region (e.g. another op, host overhead).
- Shrinking tiles **multiplies programs** past a **launch or sync knee**—re-measure total time.
- Bottleneck is **atomic saturation** or **algorithmic random write** patterns where **tile width** does not reduce **atomic count** materially.
- You assume **next power-of-two** along a narrow 2D axis is always faster than **`min(cap, exact extent)`** without measuring—narrow shapes often favor **exact** tile width.

## What To Verify After Applying

- **MLIR**: **`scf.for`** upper bounds / static tile sizes **shrink** when extent is smaller.
- **Profile**: dominant **LD/ST** or loop counts **track** the new tile, not the old padding.
- **Correctness**: boundaries, dtypes, and **all dispatch branches** still covered.
- **JIT**: non-power-of-two **`BLOCK_*`** still accepted on target.

## Related Patterns

- [discrete_memory_access.md](discrete_memory_access.md)
- [tiling.md](tiling.md)

## Expected Performance Impact

- **Indexed / masked loops** that were **padding-dominated**: often **proportional** drop in **scalar trips** and related **memop** counts when IR bounds follow the new tile.
- **Core indexed global path**: **floor** still set by **discrete access**; expect **bounded** upside from tile alignment alone.
- **Atomic-heavy** paths: **may be flat** to tile-only tweaks unless contention or trip count actually changes.
