# Hierarchical Tiling Optimization Pattern (UB Overflow Prevention)

## Summary

Use hierarchical tiling to reduce per-program working-set size so tiles, intermediates, and multi-tensor live state fit Unified Buffer (UB) safely.

Keep the kernel's high-level scheduling shape, but add an inner sub-block layer to cap peak memory footprint.
Ascend UB is fixed-size on-chip memory (commonly around the 192KB class on many Atlas targets), so footprint cliffs are often hard limits rather than gentle slowdowns.

## Use When

- Large block sizes or live intermediates risk UB overflow.
- Kernel structure is mostly correct, but memory footprint per program is too large.
- Runtime failures or instability appear when widening tiles.
- You need `BLOCK_SIZE` for scheduling/core-dim behavior and `BLOCK_SIZE_SUB` for memory safety.

## Avoid When

- UB pressure is not the bottleneck.
- Kernel still needs first-order structural rewrite (for example manual reduction should first become regular tiled `tl.dot`).
- Footprint is already safe and the next issue is overlap (prefer `software-pipeline`).

## Signals

### Code

- Multiple tensors and temporaries are simultaneously live in one tile iteration.
- Large `BLOCK_SIZE` values trigger overflow or access violations.
- Performance degrades sharply when increasing tile width.

### Runtime/Profile

- UB overflow or memory-pressure failures on larger tiles.
- Throughput does not scale with larger tiles due to footprint saturation.

### Detection heuristics

- Very large `BLOCK_SIZE` values (often `> 8192`) with multiple live tensors per iteration.
- Complex per-block compute that creates additional intermediate tensors.
- Growing tile width quickly triggers runtime faults or severe instability.

## Optimization Strategy

1. Keep a larger outer block for task scheduling.
2. Add a smaller sub-block for UB-safe processing.
3. Iterate sub-blocks sequentially inside each outer block.
4. Tune sub-block size for UB safety, alignment, and loop-overhead balance.

When choosing outer `BLOCK_SIZE`, include launch/core-dimension constraints in host planning (for example avoid exploding logical program count beyond practical hardware limits), then use `BLOCK_SIZE_SUB` to control memory residency.

## Reference Example

### Before (single large block)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    fill_mask = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
    cur_inp = tl.load(inp + offsets, mask=(~fill_mask) & mask, other=0)
    tl.store(out + offsets, cur_inp, (~fill_mask) & mask)
    tl.store(out + offsets, value, fill_mask & mask)
```

### After (hierarchical tiling)

```python
@triton.jit
def masked_fill_kernel(inp, expand_mask, value, out, N,
                      BLOCK_SIZE: tl.constexpr, BLOCK_SIZE_SUB: tl.constexpr):
    pid = tl.program_id(axis=0)
    base_offset = pid * BLOCK_SIZE
    num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)
    for sub_block_idx in range(num_sub_blocks):
        sub_offset = base_offset + sub_block_idx * BLOCK_SIZE_SUB
        offsets = sub_offset + tl.arange(0, BLOCK_SIZE_SUB)
        mask = offsets < N
        input_vals = tl.load(inp + offsets, mask=mask, other=0)
        fill_mask_vals = tl.load(expand_mask + offsets, mask=mask, other=0).to(tl.int1)
        value_to_write = tl.full([BLOCK_SIZE_SUB], value, dtype=input_vals.dtype)
        final_vals = tl.where(fill_mask_vals, value_to_write, input_vals)
        tl.store(out + offsets, final_vals, mask=mask)
```

### Host launch sketch

```python
MAIN_BLOCK_SIZE = 32768  # scheduling/coreDim-friendly
SUB_BLOCK_SIZE = 1024    # UB-footprint control
grid = lambda meta: (triton.cdiv(N, MAIN_BLOCK_SIZE),)
masked_fill_kernel[grid](inp, mask, value, out, N, MAIN_BLOCK_SIZE, SUB_BLOCK_SIZE)
```

## Sub-block sizing guidance

- **UB safety first:** choose a size that keeps peak live tensors below UB limits.
- **Performance balance:** avoid overly small slices that add excessive loop overhead.
- **Alignment awareness:** prefer transfer-friendly sizes and alignment-consistent chunking.
- **Alignment awareness:** prefer transfer-friendly sizes and alignment-consistent chunking (32-byte aligned lanes and common vector-width-friendly multiples).
- **Operation complexity:** more live operands/intermediates generally requires smaller sub-blocks.

Practical starting bands:

- Simple elementwise paths: `1024-2048`
- Multi-tensor paths: `512-1024`
- Heavier reductions/intermediate-heavy paths: `256-512`

## Practical Notes

- This pattern often turns "failing at large tiles" into a stable baseline for later tuning.
- Once capacity is safe, revisit tile ladders and pipeline/warp settings; best sub-block size is often workload-specific.
- Typical tradeoff is a small inner-loop overhead versus eliminating overflow and unlocking larger outer scheduling blocks.

## Expected impact

- **Memory:** peak UB residency scales down approximately with `BLOCK_SIZE / BLOCK_SIZE_SUB`.
- **Stability:** overflow-prone kernels become runnable in otherwise failing regimes.
- **Performance:** common tradeoff is a small loop-overhead tax (often low single-digit percent) versus significant stability and broader tunable launch space.

## Transformation checklist

1. Add `BLOCK_SIZE_SUB` to kernel signature.
2. Compute `num_sub_blocks = tl.cdiv(BLOCK_SIZE, BLOCK_SIZE_SUB)`.
3. Wrap load/compute/store in a sub-block loop.
4. Update host launch to pass both block parameters.

## What To Verify After Applying

- Correctness across full tiles and boundary tails.
- No UB overflow or memory-pressure faults on largest representative cases.
- Kernel and host signatures both pass new tiling parameters consistently.
- Parent-vs-child benchmark confirms stability and acceptable overhead.

## Related Patterns

- `software-pipeline`
- `program-multiple-rows`
- `layout-store-and-block-pointers`
- `classic-matmul`
