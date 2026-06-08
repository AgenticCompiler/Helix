# Block Pointer Dimensionality Pattern

## Summary

Use `tl.make_block_ptr` to model multidimensional contiguous tensor dimensions directly, enabling wider DMA transfers and reducing scalar address-generation overhead compared to flattened 1D offsets.

## Use When

- A high-dimensional contiguous tensor is accessed through flattened one-dimensional offsets that stride through an inner dimension.
- An inner dimension is processed by an explicit loop or decoded from `program_id` even though it could be included in the block shape.
- Profiling or IR suggests the 1D pointer path produces strided or non-coalesced loads across a dimension that is actually contiguous in memory.
- You have a `report.txt` output from `extracted_bin_data` (or you have already extracted simulation data and are about to analyze it). Focus on its overall content section.
- `report.txt` overall `[Pipe Distribution]` section shows **high SCALAR-to-VECTOR ratio** — `SCALAR_cycles% / VECTOR_cycles% > 10`, indicating heavy scalar address computation dominates execution time.
- `report.txt` overall `[Pipe Distribution]` section shows **high MTE3** — `MTE3_cycles% > 10%` — and high scalar–MTE3 serialization `%(SCALAR&MTE3/SCALAR) > 20%`, meaning address generation and data transfer are pipelined poorly.
- `report.txt` overall `[Pipe Distribution]` section shows **MTE2–MTE3 near-total overlap** — `%(MTE2&MTE3/MTE2) > 50%`, forcing the two memory transfer engines to service the same request serially.
- `report.txt` overall `[Pipeline Flows]` section shows **both XToY and YToX flows** for some pair (e.g., `SCALARToMTE3` + `MTE3ToSCALAR`), indicating pipeline stages are serialized in a cycle.
- `report.txt` overall `[Pipe Distribution]` section shows **low SCALAR–VECTOR overlap** — `%(SCALAR&VECTOR/SCALAR) < 2%` — while SCALAR is high `> 10%`, meaning scalar address generation is blocking vector execution.
- `report.txt` overall `[Pipe Distribution Over Each Core]` section lists **very few cores active** relative to hardware capacity, suggesting flat 1D grid decomposition is too coarse.
- `report.txt` overall `[Pipeline Flows]` MTE2ToVECTOR count = 0 despite the kernel loading from global memory — the 1D pointer path routes data through SCALAR→VECTOR instead of MTE2→VECTOR. Multidimensional `make_block_ptr` enables the fast MTE2→VECTOR path.

### Exclusion Signals

- `report.txt` overall `[Pipe Distribution]` section shows **SCALAR and VECTOR already well-overlapped** — `%(SCALAR&VECTOR/VECTOR) > 60%`; block_ptr cannot further improve overlap.
- The kernel uses **gather/scatter access** — non-contiguous indirect access via `tl.gather` or `index_ptr` violates the contiguous-memory assumption of `make_block_ptr`.
- `report.txt` overall `[Pipe Distribution]` section shows **MTE2 negligible** — `MTE2_cycles% < 0.5%` — meaning memory access volume is too small to be the bottleneck; likely compute-bound or control-bound instead.

## Signals

### Code

- Manual pointer arithmetic reconstructs multi-dimensional coordinates from a single flat `program_id`.

### Profiling (from `extracted_bin_data/report.txt`)

In the signals below, `PIPE_cycles%` refers to the `cycles%` column for pipe `PIPE` in `[Pipe Distribution]`. For example, given `SCALAR cycles=4594017 (45.1%)`, `SCALAR_cycles%` = 45.1%.

#### Strong Signals

1. **High SCALAR-to-VECTOR ratio** — `SCALAR_cycles% / VECTOR_cycles% > 10`. Flat 1D pointer arithmetic generates heavy scalar address computation (`//`, `%`, `* stride`) that dwarfs useful vector work. Multidimensional `make_block_ptr` replaces scalar address generation with DMA descriptors, sharply reducing SCALAR.

2. **High MTE3 with SCALAR–MTE3 serialization** — `MTE3_cycles% > 10%` AND `%(SCALAR&MTE3/SCALAR) > 20%`. High MTE3 plus high SCALAR–MTE3 overlap means scalar address computation and data transfer are serialized. Block_ptr eliminates this by letting the DMA engine handle addressing autonomously.

3. **MTE2–MTE3 near-total overlap** — `%(MTE2&MTE3/MTE2) > 50%`. When the two memory transfer engine levels overlap almost completely, the 1D pointer path is forcing them to service the same request serially. Multidimensional block_ptr lets MTE2 and MTE3 operate independently.

#### Moderate Signals

4. **Pipeline flow cycle** — Pipeline Flows contain both `XToY` and `YToX` for some pair (e.g., `SCALARToMTE3` + `MTE3ToSCALAR`, or `SCALARToVECTOR` + `VECTORToSCALAR`). A cycle means pipeline stages are serialized. The optimization goal is to eliminate cycles and approach the ideal flow `MTE2ToVECTOR → VECTORToMTE3`.

5. **Low SCALAR–VECTOR overlap with high SCALAR** — `%(SCALAR&VECTOR/SCALAR) < 2%` AND `SCALAR_cycles% > 10%`. Scalar address generation is blocking vector execution. After block_ptr, SCALAR&VECTOR/SCALAR typically rises as the two run more concurrently.

#### Weak Signals

6. **Few cores active** — The `[Pipe Distribution Over Each Core]` section lists very few cores relative to the hardware capacity. Flat 1D grid decomposition is often too coarse; upgrading to multidimensional block_ptr enables finer-grained grid partitioning that naturally uses more cores.

#### Exclusion Signals (any one suggests this pattern will NOT help)

1. **SCALAR and VECTOR already well-overlapped** — `%(SCALAR&VECTOR/VECTOR) > 60%` means scalar and vector are already running concurrently; block_ptr cannot further improve overlap.
2. **Access pattern is gather/scatter** — Index-driven non-contiguous access (e.g., `tl.gather`, indirect loads via `index_ptr`) violates the contiguous-memory assumption of `make_block_ptr`.
3. **MTE2 is negligible** — `MTE2_cycles% < 0.5%` means memory access volume is too small to be the bottleneck; the issue is likely compute-bound or control-bound instead.

#### Decision Rule

- **Trigger** if any strong signal is present.
- **Also trigger** if two or more moderate signals co-occur.
- **Skip** if any exclusion signal is present.

## What To Verify After Applying

- Confirm every field in `tl.make_block_ptr` — `shape`, `strides`, `offsets`, `block_shape`, and `order` — matches the actual tensor layout. One wrong field can silently benchmark a different access pattern.
- Verify that `boundary_check` and `padding_option` produce correct results on tail blocks.
- When an inner dimension was previously part of grid partitioning, verify the grid reduction is correct and that per-program work density improves.

---

## Detail

### Before (flattened 1D offset)

```python
pid = tl.program_id(0)
# Flattened offset strides through inner dimension — compiler sees one long strided access
offs = pid * BLOCK + tl.arange(0, BLOCK)
vals = tl.load(x + offs, mask=offs < total)
```

### After (multidimensional block pointer)

```python
pid_t = tl.program_id(0)
ptr = tl.make_block_ptr(
    base=x,
    shape=(T, H),
    strides=(stride_t, stride_h),
    offsets=(pid_t * BLOCK_T, 0),
    block_shape=(BLOCK_T, BLOCK_H),
    order=(1, 0),
)
tile = tl.load(ptr, boundary_check=(0, 1), padding_option="zero")
```

### Vectorizing an inner dimension

If an inner loop only walks a small dimension, include that dimension in the loaded tile and compute with an extra tensor axis:

```python
# Before: explicit inner loop over small dim
pid = tl.program_id(0)
for d in range(D):
    vals = tl.load(x + pid * stride_pid + d * stride_d + tl.arange(0, BLOCK))

# After: include D in the block shape
pid = tl.program_id(0)
ptr = tl.make_block_ptr(
    base=x, shape=(N, D), strides=(stride_n, stride_d),
    offsets=(pid * BLOCK_N, 0), block_shape=(BLOCK_N, D), order=(1, 0),
)
tile = tl.load(ptr, boundary_check=(0, 1))
```

Update broadcasting and grid mapping together; if the inner dimension was part of grid partitioning, removing that grid axis may be part of the optimization.
