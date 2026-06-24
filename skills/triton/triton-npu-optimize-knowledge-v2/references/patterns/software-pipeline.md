# Software Pipelining and Block Pointer Optimization Pattern

## Summary

Use this pattern to increase overlap between memory movement and compute in an already tiled hot loop.

The focus is not changing algorithm shape, but changing loop scheduling so data for the next iteration is fetched while current tiles are being computed.

## Use When

- The kernel already has a stable tiled loop, but execution still looks like synchronous load-then-compute.
- Profiling shows wait-heavy behavior or visible compute gaps while memory engines fetch next tiles.
- Block-pointer structure can replace repeated manual pointer arithmetic on the hot path.
- UB can hold the active tile set needed for prefetch/pipeline depth.

## Avoid When

- Inner loop trip count is tiny and pipeline setup overhead dominates.
- UB headroom is insufficient for multiple live tile sets.
- Iteration `i+1` depends on compute results from iteration `i` in a way that prevents overlap.
- The kernel still needs first-order structural rewriting (for example manual reduction should become regular tiled `tl.dot` first).

## Signals

### Code

- Tiled loops issue `tl.load` then immediately compute, repeatedly, with little decoupling.
- Pointer arithmetic and offset rebuilds dominate loop body setup.
- `tl.make_block_ptr` / `tl.advance` are absent despite regular tiled access.

### Profile

- Timeline shows Cube/Vector idle gaps while waiting for memory transfers.
- Improvements from pure tiling changes plateau before overlap is improved.
- Wait-dominant behavior persists even after basic launch geometry cleanup.

## Repairs

### Convert to block-pointer traversal

Use `tl.make_block_ptr` for tiled operands and `tl.advance` for deterministic pointer movement between iterations.

### Prefetch before entering steady-state loop

Load the first tile (or first pipeline stage) before the main overlapped loop starts.

### Overlap compute with next-tile fetch

In each iteration, compute on currently staged tiles while issuing loads for the next tiles.

### Tune pipeline depth under UB constraints

Increase active-stage depth only when UB capacity and register pressure remain safe.

### Simplified code sketch

```python
a_ptrs = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                           offsets=(m0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
b_ptrs = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                           offsets=(0, n0), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

# Prefetch first tile.
a_tile = tl.load(a_ptrs)
b_tile = tl.load(b_ptrs)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for _ in range(0, K, BLOCK_K):
    a_ptrs = tl.advance(a_ptrs, [0, BLOCK_K])
    b_ptrs = tl.advance(b_ptrs, [BLOCK_K, 0])
    acc = tl.dot(a_tile, b_tile, acc)  # compute current tile
    a_tile = tl.load(a_ptrs)           # fetch next tile
    b_tile = tl.load(b_ptrs)
```

## Synthesized Guidance

- Apply software pipelining after tiled structure is already sound; it is an overlap optimization, not a substitute for foundational kernel design.
- Start with shallow prefetching and validate correctness/perf before adding deeper pipeline stages.
- Pair block pointers with overlap changes; this usually reduces scalar setup overhead and clarifies pipeline intent.
- If overlap gains are weak, check whether bottleneck has shifted to layout/store shape or working-set pressure.

## Related Patterns

- `classic-matmul`
- `tiling`
- `compile_hint`
- `layout-store-and-block-pointers`

## What To Verify After Applying

- Correctness is unchanged across all benchmark regimes and boundary tiles.
- Hot loop now computes on staged tiles while fetching subsequent tiles.
- UB/register usage remains within safe limits at the chosen pipeline depth.
- Parent-vs-child benchmarks confirm net latency improvement, not just timeline cosmetics.
