# i64/i32 Comparison Optimization Pattern

## Summary

Rewrite explicit integer compare-heavy logic into a form that is more vector-friendly on Ascend NPU, especially when scalarized compares are blocking fast masking or selection.

## Use When

- Explicit `i64` or `i32` comparisons feed `tl.where`, conditional assignments, reused masks, or other hot-path selection logic.
- The comparison is outside the compiler's normal inline `tl.load(..., mask=...)` or `tl.store(..., mask=...)` fast path.
- `fp32` comparison preserves the mask semantics for the compared value range.
- You have a `report.txt` output from `extracted_bin_data` (or you have already extracted simulation data and are about to analyze it). Focus on its overall content section.
- `report.txt` overall `[Pipe Distribution]` shows high SCALAR-to-VECTOR ratio: `%(SCALAR) / %(VECTOR) > 10`.
- `report.txt` overall `[Key Ratios]` shows a high `SCALAR:VECTOR` ratio, such as `SCALAR:VECTOR_instr` much larger than `4:1`.
- `report.txt` overall `[VECTOR Unit]` shows low or zero utilization, and the top VECTOR instructions are mask-like operations such as `MOVEMASK`.
- `report.txt` overall `[TRACE Events]` contains many mask/control-related scalar events such as `CMP_IMM`, `JUMPC`, `JUMPCMP`, `MOVEMASK`, or `SIGNEXT`.
- UB conflicts are low and MTE2/MTE3 activity does not explain the regression by itself, making scalarized mask/control work a plausible cause.

## Signals

### Code

- Integer comparisons produce explicit boolean masks used in `tl.where`, conditional assignments, or similar hot-path logic.
- The comparison is written outside the compiler's normal `tl.load` or `tl.store` mask fast path.
- The code still compares integer operands directly even though vector-friendly `fp32` comparison would preserve semantics.

### Profile

- `report.txt` overall `[Pipe Distribution]` shows high SCALAR-to-VECTOR ratio, for example `%(SCALAR) / %(VECTOR) > 10`. This supports `vec-cmp` only when the code has explicit integer masks, because scalarized integer compares can spend most cycles building control/mask state instead of doing useful vector selection.
- `report.txt` overall `[Key Ratios]` shows a high `SCALAR:VECTOR` ratio, such as `SCALAR:VECTOR_instr` much larger than `4:1` or `SCALAR:VECTOR_cycles > 10:1`. This matches `vec-cmp` when integer compare masks feed hot-path `tl.where`, conditional assignments, or reused masks, because those masks should become cheaper if the comparison is lowered through vector-friendly `fp32` compare.
- `report.txt` overall `[VECTOR Unit]` shows low or zero utilization, and the top VECTOR instructions are mask-like operations such as `MOVEMASK`. This suggests the vector pipe is mostly receiving or materializing masks rather than performing sustained vector compute, which is the failure mode `vec-cmp` tries to avoid.
- `report.txt` overall `[TRACE Events]` or `[SCALAR Instr Types]` contains many mask/control-related scalar events such as `CMP_IMM`, `JUMPC`, `JUMPCMP`, `MOVEMASK`, `SIGNEXT`, `ZEROEXT`, or `AND`. These are direct signatures of integer compare, branch/control, integer widening, and mask materialization; they strengthen the `vec-cmp` diagnosis when they line up with explicit compare-mask code.
- UB conflicts are low and MTE2/MTE3 activity does not explain the regression by itself. This matters because `vec-cmp` targets scalarized mask/control work; if UB, memory transfer, gather/scatter, layout, or address/index math dominates instead, the report points to another pattern.
- Treat the `report.txt` evidence as a trigger only when it matches the code signal; if `DIV`, `REM`, `MADD`, `ADD`, or stride arithmetic dominate around flat-index decoding or pooling coordinate math, treat `vec-cmp` as secondary and prefer the matching address/indexing pattern first.

## Problem Description

On Huawei Ascend NPU devices, integer comparison operations (`i64` and `i32`) cannot utilize vector processing units and degrade to scalar computation, significantly reducing performance.

## Optimization Strategy

Convert integer comparisons to `fp32` type to leverage `vec_cast` and `vec_cmp` instructions for vectorized operations, achieving better performance through hardware acceleration.

### Key Principles

1. **Identify explicit comparison operations**: Look for code that performs integer comparisons outside of `tl.load`/`tl.store` mask parameters
2. **Convert to fp32**: Cast integer operands to `fp32` before comparison
3. **Preserve semantics**: Ensure the logical result of the comparison remains unchanged
4. **Avoid redundant changes**: Skip comparisons already in `fp32` or auto-vectorized contexts

### Important Notes

- **`tl.load` and `tl.store` masks**: The compiler automatically optimizes comparison operations in mask parameters. Manual conversion is NOT needed for these cases.
- **Explicit comparisons**: Only target comparison operations that produce explicit boolean masks used in conditional logic (e.g., `tl.where`, conditional assignments)
- **Type safety**: When converting types, ensure numerical precision is maintained for downstream operations
- **Compare helpers and NaN semantics**: When hot-path compare helpers such as `tl.maximum()` or `tl.minimum()` appear in the optimized kernel, inspect similar call sites for omitted `propagate_nan`. Add `propagate_nan=tl.PropagateNan.ALL` only when the round intentionally wants explicit, consistent NaN propagation, and record that this can change NaN-input behavior.

## Detection Pattern

Look for code patterns like:

```python
# Problematic: i64 comparison outside tl.load/tl.store
mask = offsets < n_elements  # i64 comparison
result = tl.where(mask, x, y)  # Used in conditional logic

# Problematic: i32 comparison
valid = x_ids > y_ids  # i32 comparison
output = tl.where(valid, data, 0.0)
```

## Optimization Example

### Before Optimization

```python
@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)

    # Problem: i64 comparison degrades to scalar on NPU
    valid_indices = offsets < n_elements

    # Use comparison result in conditional logic
    output = tl.where(valid_indices, x * 2, 0.0)

    tl.store(output_ptr + offsets, output)
```

### After Optimization

```python
@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptr + offsets)

    # Optimization: Convert to fp32 for vectorized comparison
    offsets_fp32 = tl.cast(offsets, tl.float32)
    n_elements_fp32 = tl.cast(n_elements, tl.float32)
    valid_indices = offsets_fp32 < n_elements_fp32  # Vectorized fp32 comparison

    # Use comparison result in conditional logic
    output = tl.where(valid_indices, x * 2, 0.0)

    tl.store(output_ptr + offsets, output)
```

## Another Example

### Before Optimization

```python
# Problem: i64 comparison in tl.load mask degrades to scalar on NPU
# Note that though value of the comparison is used as the argument of tl.load, it is outside the tl.load statement
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
```

### After Optimization

```python
# Optimization: Convert to fp32 for vectorized comparison
offset_fp32 = tl.cast(offsets, tl.float32)
n_elements_fp32 = tl.cast(n_elements, tl.float32)
mask_fp32 = offset_fp32 < n_elements_fp32
x = tl.load(x_ptr + offsets, mask=mask_fp32)
```

## Avoid When

- The comparison appears only as an inline `tl.load` or `tl.store` boundary mask; the compiler normally handles this case.
- The compared integer values can exceed the exact `fp32` integer range, or exact large-integer ordering matters.
- The report points primarily to address math, layout, UB conflicts, gather/scatter, or memory transfer rather than compare scalarization.
- The comparison is not on the hot path or the operands are already compared as `fp32`.
- Nearby hot-path compare helpers such as `tl.maximum()` or `tl.minimum()` require specific NaN semantics; add `propagate_nan=tl.PropagateNan.ALL` only as an intentional behavior change and record the NaN-input behavior.

## What To Verify After Applying

- Verify the comparison result still feeds the same hot-path conditional logic after the dtype rewrite.
- Verify both operands are cast in a way that preserves semantic equivalence for the downstream mask usage.
- Re-check downstream dtype expectations and confirm the comparison is no longer a scalarization bottleneck.
- Re-check `extracted_bin_data/report.txt` when available and confirm the scalar/vector balance improved instead of merely moving the bottleneck.
- Re-check `tl.maximum()` and `tl.minimum()` call sites on the hot path and document any intentional `propagate_nan` choice as a semantics change.
