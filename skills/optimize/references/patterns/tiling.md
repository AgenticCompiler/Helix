# Hierarchical Tiling Optimization Pattern (UB Overflow Prevention)

## Problem Description

**Root Cause:**
- Ascend NPU has limited on-chip Unified Buffer (192KB on Atlas 800T A2/A3)
- Large `BLOCK_SIZE` values cause excessive memory consumption per program instance
- When loading multiple tensors or performing complex operations within a block, UB usage can overflow

**Symptoms:**
- Runtime errors indicating UB overflow
- Kernel failures with large block sizes
- Memory access violations on NPU

## Optimization Strategy

Introduce **hierarchical tiling** (also called sub-blocking) to further subdivide large blocks:

### Key Principles

1. **Two-level blocking**: Separate task scheduling from memory management
   - Keep main `BLOCK_SIZE` for task scheduling (coreDim compliance)
   - Introduce `BLOCK_SIZE_SUB` for processing data in smaller batches

2. **Process in loops**: Use inner loops to process sub-blocks sequentially
   - Reduce peak memory usage by processing data in smaller chunks
   - Maintain reasonable coreDim values for efficient task scheduling
   - Control UB usage through smaller batch sizes

3. **Balance performance and memory**:
   - Small enough to fit within UB capacity
   - Large enough to maintain reasonable performance
   - Aligned with memory access patterns (32-byte alignment)

## Detection Pattern

Look for code with these characteristics:

1. **Large BLOCK_SIZE values** (> 8192)
2. **Multiple tensor loads** within a single block
3. **Complex operations** that require intermediate storage
4. **UB overflow errors** at runtime

### Problematic Code Patterns

```python
# Problem: Large block size + multiple loads = UB overflow
@triton.jit
def kernel(inp, mask1, mask2, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Loading multiple large tensors at once
    mask = offsets < N
    data1 = tl.load(inp + offsets, mask=mask)
    data2 = tl.load(mask1 + offsets, mask=mask)
    data3 = tl.load(mask2 + offsets, mask=mask)

    # UB overflow here!
    result = complex_operation(data1, data2, data3)
    tl.store(out + offsets, result, mask=mask)
```

## Code Example

### Before Optimization (UB Overflow)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Loading all data at once causes UB overflow
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)

    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)
```

**Issues:**
- Single large block loads all data at once
- Multiple tensors occupy UB simultaneously
- Risk of overflow with large BLOCK_SIZE values

### After Optimization (Hierarchical Tiling)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                      BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE

    # Calculate the number of sub-blocks to process
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)

    # Process in blocks to avoid UB overflow
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N

        # Load and process data in batches (smaller UB footprint)
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)

        # First write the original data
        tl.store(out + offsets, input_vals, mask=mask)

        # Then overwrite the target value at positions that need filling
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)
```

**Improvements:**
- Data processed in smaller sub-blocks
- Reduced peak UB usage
- Maintains task scheduling efficiency with large main BLOCK_SIZE

### Host Code Configuration

```python
def masked_fill(inp, mask, value):
    N = inp.numel()

    # Two-level blocking strategy
    MAIN_BLOCK_SIZE = 32768  # Ensure coreDim compliance (N / 32768 < 65535)
    SUB_BLOCK_SIZE = 1024    # Control UB usage (process in smaller chunks)

    grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
    masked_fill_kernel[grid](inp, mask, value, out, N,
                           MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
    return out
```

## Guidelines for Sub-Block Size Selection

**SUB_BLOCK_SIZE should be:**

1. **UB-safe**: Small enough to fit within UB capacity
   - Simple element-wise operations: 1024-2048
   - Operations with multiple tensors: 512-1024
   - Complex reductions: 256-512

2. **Performance-aware**: Large enough to maintain reasonable performance
   - Avoid too-small blocks that increase loop overhead
   - Balance between memory usage and computation efficiency

3. **Alignment-aware**: Divisible by or aligned with memory access patterns
   - 32-byte alignment requirement
   - Vector width considerations (typically 128/256 elements)

## When NOT to Apply

1. **Small BLOCK_SIZE** No significant memory pressure
2. **Simple operations** with single tensor - UB usage is minimal
3. **Already optimized** with sub-blocking present

## Implementation Checklist

- [ ] Identify kernels with large BLOCK_SIZE values
- [ ] Check for multiple tensor loads or complex operations
- [ ] Calculate appropriate SUB_BLOCK_SIZE based on operation type
- [ ] Add inner loop to process sub-blocks
- [ ] Update function signature to include BLOCK_SIZE_SUB parameter
- [ ] Update host code to pass both block sizes

## Expected Performance Impact

**Memory Usage:**
- UB usage reduced by factor of `BLOCK_SIZE / BLOCK_SIZE_SUB`
- Enables processing larger arrays without overflow

**Performance Trade-offs:**
- **Pros**: Enables kernels that would otherwise overflow UB
- **Cons**: Small loop overhead for sub-block iteration (typically < 5%)
- **Net effect**: Enables functionality that was previously impossible

**Typical Results:**
- UB overflow kernels become functional
- Performance impact: -5% to +10% depending on operation complexity
- Enables larger batch sizes and more efficient task scheduling

## Code Transformation Pattern

**Step 1: Add sub-block parameter**
```python
# Before
def kernel(..., BLOCK_SIZE: tl.constexpr):

# After
def kernel(..., BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
```

**Step 2: Calculate sub-block count**
```python
num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
```

**Step 3: Wrap computation in loop**
```python
for sub_block_idx in range(num_sub_blocks):
    # Compute sub-block offsets
    sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
    offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)

    # Load, compute, store for this sub-block
    ...
```

**Step 4: Update host code**
```python
# Before
kernel[grid](..., BLOCK_SIZE)

# After
kernel[grid](..., BLOCK_SIZE, BLOCK_SIZE_SUB)
```
