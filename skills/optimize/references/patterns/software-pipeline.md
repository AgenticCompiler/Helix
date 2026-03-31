# Software Pipelining and Block Pointer Optimization Pattern

## Problem Description

Huawei Ascend NPUs (DaVinci architecture) use a **Decoupled Access-Compute (DAC)** design. This means memory movement (MTE engines) and computation (Cube/Vector cores) happen on different hardware units.

In standard Triton code, the kernel often executes synchronously: it waits for a data load to finish before starting the calculation. This results in "stalls" or "gaps" in the `msprof` timeline where the expensive Cube/Vector cores sit idle while waiting for the MTE engine to fetch data from Global Memory (HBM).

## Optimization Strategy

Implement **Software Pipelining (Double Buffering)** combined with **Block Pointers**. This allows the NPU to fetch the *next* tile of data from memory while simultaneously performing computation on the *current* tile.

### Key Principles

1.  **Use `tl.make_block_ptr`**: Replaces manual pointer arithmetic. It allows the hardware to utilize specialized 2D DMA controllers, reducing Scalar Unit overhead.
2.  **Pre-fetching**: Load the first tile (or first few tiles) before entering the main loop.
3.  **Overlapped Loop**: Inside the loop, load data for iteration `i+1` before (or during) the computation of iteration `i`.
4.  **Pointer Advancement**: Use `tl.advance` to move pointers forward. This is more efficient for the Scalar unit than recalculating offsets.

### Important Notes

-   **UB Size Constraints**: Pipelining requires keeping multiple tiles in the Unified Buffer (UB) simultaneously. Ensure `2 * (Tile_M * Tile_K * dtype_size)` fits within the available UB (usually 192KB-256KB).
-   **Loop Latency**: Pipelining is most effective when the computation time and memory transfer time are roughly balanced.

## Detection Pattern

Look for code patterns where loads and math are interleaved sequentially inside a loop:

```python
# Problematic: Synchronous "Load-then-Compute"
for k in range(0, K, BLOCK_SIZE_K):
    # Scalar unit calculates offsets here
    a = tl.load(a_ptr + k * stride + offsets)
    b = tl.load(b_ptr + k * stride + offsets)
    # Cube core waits for the loads above to complete
    res = tl.dot(a, b, res)
```

## Optimization Example

### Before Optimization

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # ... (program ID and offset logic)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # Sequential Load (MTE2)
        a = tl.load(a_ptr + k_offsets)
        b = tl.load(b_ptr + k_offsets)
        # Sequential Compute (Cube) - Waits for MTE2
        accumulator = tl.dot(a, b, accumulator)
```

### After Optimization (Pipelined with Block Pointers)

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    # 1. Initialize Block Pointers (Hardware optimized)
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(K, 1),
                                   offsets=(0, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(N, 1),
                                   offsets=(0, 0), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N), order=(1, 0))

    # 2. Prefetch first tile (Start MTE2 transfer early)
    a_tile = tl.load(a_block_ptr)
    b_tile = tl.load(b_block_ptr)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        # 3. Advance pointers for the NEXT iteration
        a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_SIZE_K])
        b_block_ptr = tl.advance(b_block_ptr, [BLOCK_SIZE_K, 0])

        # 4. Compute CURRENT tile while loading NEXT tile
        # The Triton compiler maps this to overlapped Cube and MTE instructions
        accumulator = tl.dot(a_tile, b_tile, accumulator)

        # Load next tile (Async)
        a_tile = tl.load(a_block_ptr)
        b_tile = tl.load(b_block_ptr)
```

## When NOT to Apply

1.  **Tiny Inner Loops**: If the loop only runs 1 or 2 times, the pre-fetch overhead might exceed the savings.
2.  **Extreme Memory Pressure**: If the tile size is so large that the Unified Buffer cannot hold two sets of tiles (current and next).
3.  **Dependency Chains**: If Tile `i+1` depends on the result of the computation of Tile `i`.

## Implementation Checklist

- [ ] Replace raw pointer arithmetic with `tl.make_block_ptr`.
- [ ] Pull the first `tl.load` outside the loop to initialize the pipeline.
- [ ] Inside the loop, use `tl.advance` to update pointers.
- [ ] Ensure the final `tl.dot` or math operation uses the previously loaded tile.
- [ ] Verify that the total memory used by all active tiles (current + next) does not exceed NPU UB capacity.
