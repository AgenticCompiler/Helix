# Software Pipelining and Block Pointer Optimization Pattern

## Summary

Use software pipelining to overlap memory transfer and compute in an already well-tiled hot loop.

On Ascend's decoupled access-compute design, this usually means staging tiles with block pointers, prefetching ahead, and computing tile `i` while loading tile `i+1`.

## Use When

- Loop structure is already tiled and semantically stable.
- Execution still looks like synchronous load-then-compute.
- Profiles show wait-heavy gaps while memory engines feed compute units.
- UB headroom can hold the required live tile sets.

## Avoid When

- Inner-loop trip count is tiny (pipeline setup dominates).
- UB cannot hold multi-stage live tiles safely.
- Iteration dependencies prevent overlap.
- Kernel still needs first-order structural rewrite (for example convert manual reduction to regular tiled `tl.dot` first).

## Signals

### Code

- `tl.load` and compute are serialized every iteration.
- Manual pointer arithmetic dominates loop body.
- `tl.make_block_ptr` / `tl.advance` are absent despite regular tiled access.

### Profile

- Cube/Vector idle gaps while MTE transfers run.
- Tiling-only changes plateau before overlap is addressed.

## Problem framing

Ascend NPUs use decoupled engines for memory movement and compute. If code issues load and compute strictly in sequence, compute units wait for transfers. Software pipelining aims to keep both sides active by staging future tiles before current compute completes.

## Optimization Strategy

1. Replace raw pointer arithmetic with `tl.make_block_ptr`.
2. Prefetch first tile(s) before entering the steady-state loop.
3. Advance pointers via `tl.advance`.
4. Compute on current staged tiles while issuing loads for next tiles.
5. Tune `num_stages` and `num_warps` conservatively under UB/register limits.

## Reference patterns

### Serialized pattern (before)

```python
for k in range(0, K, BLOCK_K):
    a = tl.load(a_ptr + k_offsets)
    b = tl.load(b_ptr + k_offsets)
    acc = tl.dot(a, b, acc)
```

### Overlapped pattern (after)

```python
a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                offsets=(m0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                offsets=(0, n0), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))

a_tile = tl.load(a_block_ptr)  # prefetch
b_tile = tl.load(b_block_ptr)
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

for _ in range(0, K, BLOCK_K):
    a_block_ptr = tl.advance(a_block_ptr, [0, BLOCK_K])
    b_block_ptr = tl.advance(b_block_ptr, [BLOCK_K, 0])
    acc = tl.dot(a_tile, b_tile, acc)  # compute current
    a_tile = tl.load(a_block_ptr)      # fetch next
    b_tile = tl.load(b_block_ptr)
```

## Practical Notes

- Pipeline depth is not monotonic: both shallower and deeper `num_stages` can regress depending on tile footprint and transfer balance.
- Block-pointer conversion and overlap changes work best together: less scalar address overhead plus clearer staging intent.
- If overlap edits are flat, another pattern is likely primary (layout, tiling, launch geometry, or scalar-control cleanup).
- Keep branch-local evidence: a stage setting that wins on one tile/config family may fail on another.
- UB budgeting matters: double-buffer style overlap usually implies keeping roughly two tile sets live; validate with a quick footprint estimate such as `2 * (tile_elems * dtype_bytes)` for dominant operands against device UB capacity (commonly ~192KB to 256KB class depending on target).
- Pipelining is most effective when compute and transfer latencies are of similar order; if one side dominates completely, stage-depth increases often become noise or regressions.

## What To Verify After Applying

- Correctness across full and boundary tiles.
- Hot loop genuinely computes while loading the next tile.
- UB/register usage stays within safe limits at chosen stage depth.
- Parent-vs-child benchmarks show real latency gain.

## Related Patterns

- `classic-matmul`
- `tiling`
- `layout-store-and-block-pointers`
- `compile_hint`
