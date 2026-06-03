# i64/i32 Comparison Optimization Pattern

## Summary

Rewrite explicit integer compare-heavy logic into a form that is more vector-friendly on Ascend NPU, especially when scalarized compares are blocking fast masking or selection.

## Use When

- Explicit `i64` or `i32` comparisons appear on the hot path outside the compiler's normal fast load/store mask cases.
- Comparison-heavy control flow or masking looks like a real vectorization blocker rather than just minor boundary handling.
- You have a `report.txt` output from `extracted_bin_data` and its overall pipe sections suggest scalarized compare work is blocking useful vector work.

## Signals

### Code

- Integer comparisons produce explicit boolean masks used in `tl.where`, conditional assignments, or similar hot-path logic.
- The comparison is written outside the compiler's normal `tl.load` or `tl.store` mask fast path.
- The code still compares integer operands directly even though vector-friendly `fp32` comparison would preserve semantics.

### Profile

- `report.txt` overall `[Pipe Distribution]` section shows **dominant scalar work with little useful vector work**, for example `SCALAR_instr_pct > 75%`, `SCALAR_cycles_pct > 75%`, and `VECTOR_cycles_pct < 15%`.
- `report.txt` overall `[Key Ratios]` section shows a very high scalar-to-vector ratio, for example `SCALAR:VECTOR_instr > 10:1` or `SCALAR:VECTOR_cycles > 10:1`, while the hot code has explicit integer comparisons that can explain the scalar work.
- `report.txt` `[TRACE Events]` or `[SCALAR Instr Types]` sections contain compare/mask-building scalar activity such as `CMP_IMM`, `JUMPC`, `JUMPCMP`, `MOVEMASK`, `SIGNEXT`, `ZEROEXT`, `AND`, or repeated scalar integer arithmetic near mask construction.
- `report.txt` `[VECTOR Unit]` shows very few vector instructions, or mainly mask-related instructions such as `MOVEMASK`, suggesting compare/mask work did not become useful vector compute.
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

1. **Comparisons in `tl.load`/`tl.store` masks** - already auto-optimized:
   ```python
   # No change needed - compiler handles this
   x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
   ```

2. **Already using fp32 comparisons** - no optimization needed:
   ```python
   # Already optimal
   offsets_fp32 = tl.cast(offsets, tl.float32)
   valid = offsets_fp32 < threshold
   ```

3. **Non-performance-critical code** - optimization overhead may not be justified

4. **`report.txt` does not support a scalarized-compare bottleneck** - if `SCALAR` is not meaningful, `VECTOR` is already well utilized, or the scalar work is better explained by address generation, pooling coordinate math, gather/scatter, or memory movement, prefer the matching pattern instead.

## What To Verify After Applying

- Verify the comparison result still feeds the same hot-path conditional logic after the dtype rewrite.
- Verify both operands are cast in a way that preserves semantic equivalence for the downstream mask usage.
- Re-check downstream dtype expectations and confirm the comparison is no longer a scalarization bottleneck.
- Re-check `extracted_bin_data/report.txt` when available and confirm the scalar/vector balance improved instead of merely moving the bottleneck.
- Re-check `tl.maximum()` and `tl.minimum()` call sites on the hot path and document any intentional `propagate_nan` choice as a semantics change.
