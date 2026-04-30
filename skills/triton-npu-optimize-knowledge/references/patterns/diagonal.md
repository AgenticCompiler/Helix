# Diagonal Block Traversal Pattern

## Summary

While it is good to access data from L2 cache as much as possible, having multiple
kernels accessing the *same* data from the L2 cache may cause bank conflicts that slow down operations.
One can use the diagonal access pattern to replace the usual swizzle pattern to alleviate this problem.
The example applies this technique to matrix multiplication, but it may be applicable in other contexts.

## Use When

- Large tiled matrix-style work shows poor locality or bank-conflict-like behavior even though the basic tiling is already reasonable.
- Many programs touch the same cache regions at the same time, so changing block traversal order may improve effective L2 use.

## Signals

### Code

- Traditional row-major or horizontal block assignment makes many cores touch the same left-matrix cache region at once.
- The matrix already spans many blocks along both `M` and `N`, so traversal order is a plausible performance lever rather than a cosmetic rewrite.
- The right-hand matrix is large enough that ordinary block traversal can churn L2 and lower reuse.

## Detail

### Diagonal Matmul

See example below

```python
@triton.autotune(
configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "BLOCK_TRESHHOLD": 8}),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def matmul_kernel(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_TRESHHOLD: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0

    '''
    With ordinary horizontal work partitioning, the task-block numbering is:
    [0,  1,  2,  3,  4,  5,  6,  7]
    [8,  9,  10, 11, 12, 13, 14, 15]
    [16, 17, 18, 19, 20, 21, 22, 23]
    [24, 25, 26, 27, 28, 29, 30, 31]
    [32, 33, 34, 35, 36, 37, 38, 39]
    [40, 41, 42, 43, 44, 45, 46, 47]
    [48, 49, 50, 51, 52, 53, 54, 55]
    [56, 57, 58, 59, 60, 61, 62, 63]
    Core 0 handles tasks 0, 20, 40, and 60 (4 blocks).
    Core 1 handles tasks 1, 21, 41, and 61 (4 blocks).
    Core 2 handles tasks 2, 22, 42, and 62 (4 blocks).
    ...
    Core 19 handles tasks 19, 39, and 59 (3 blocks).
    
    For large shapes, the traditional horizontal partitioning above causes two problems:
    1. Many cores access the same left-matrix cache region at the same time, creating bank conflicts
       and lowering hardware efficiency.
    2. By the time one full row of mat_c is complete, all right-matrix data for that row has already
       been consumed. When the right matrix is large, it can exceed L2-cache capacity, triggering
       cache eviction and reload traffic. Later rows then see more cache misses, which lowers L2 hit
       rate and hurts kernel performance.
    Using 8 x 8 diagonal partitioning lets each 8 x 8 block region progress along the diagonal, which
    significantly improves both problems above.

    The example below uses 8 x 8 diagonal partitioning. In practice, `BLOCK_TRESHHOLD` is tuned to
    choose the best threshold. Under 8 x 8 diagonal partitioning, each 8 x 8 region is numbered like this:
    [0,  8,  16, 24, 32, 40, 48, 56]
    [57, 1,  9,  17, 25, 33, 41, 49]
    [50, 58, 2,  10, 18, 26, 34, 42]
    [43, 51, 59, 3,  11, 19, 27, 35]
    [36, 44, 52, 60, 4,  12, 20, 28]
    [29, 37, 45, 53, 61, 5,  13, 21]
    [22, 30, 38, 46, 54, 62, 6,  14]
    [15, 23, 31, 39, 47, 55, 63, 7]
    
    When the M dimension spans more than 8 base blocks, diagonal partitioning can significantly reduce
    bank conflicts. When the right matrix exceeds L2-cache capacity, diagonal partitioning can also
    improve L2 reuse. So when both M and N exceed 8 blocks, enabling diagonal partitioning is often
    beneficial, especially when the right matrix is larger than L2.
    '''
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    # Enable diagonal partitioning when the task count is large enough to benefit from it.
    if NUM_BLOCKS_M >= BLOCK_TRESHHOLD and NUM_BLOCKS_N >= BLOCK_TRESHHOLD:
        for block_idx in range (
            pid, NUM_BLOCKS, num_cores
        ):
            # 8 x 8 diagonal partitioning implementation
            curThresholdM = BLOCK_TRESHHOLD if block_idx < (NUM_BLOCKS_M // BLOCK_TRESHHOLD * BLOCK_TRESHHOLD) * NUM_BLOCKS_N else NUM_BLOCKS_M % BLOCK_TRESHHOLD
            curThresholdM_thresholdN = curThresholdM * BLOCK_TRESHHOLD
            curThresholdN = BLOCK_TRESHHOLD if block_idx % (NUM_BLOCKS_N * BLOCK_TRESHHOLD) < (curThresholdM * NUM_BLOCKS_N) // curThresholdM_thresholdN * curThresholdM_thresholdN else NUM_BLOCKS_N % BLOCK_TRESHHOLD
            localRelativeBlock = block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) % (BLOCK_TRESHHOLD * curThresholdM)
            task_m_idx = localRelativeBlock % curThresholdM + block_idx // (BLOCK_TRESHHOLD * NUM_BLOCKS_N) * BLOCK_TRESHHOLD
            # Compute the least common multiple to make the block-coordinate mapping easier.
            x, y = curThresholdM, curThresholdN if curThresholdM > curThresholdN else curThresholdN, curThresholdM
            while y != 0:
                x, y = y, x % y
            lcm = curThresholdM * curThresholdN // x
            task_n_idx = (localRelativeBlock + (localRelativeBlock // lcm)) % curThresholdN + block_idx % (BLOCK_TRESHHOLD * NUM_BLOCKS_N) // curThresholdM_thresholdN * BLOCK_TRESHHOLD
            
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            
            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                    k_start + tl.arange(0, BLOCK_K)
                )[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (k_start + tl.arange(0, BLOCK_K)) < K
                )[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
                    n_start + tl.arange(0, BLOCK_N)
                )[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)
    else:
        # Traditional sequential partitioning
        for block_idx in range (
            pid, NUM_BLOCKS, num_cores
        ):
            task_m_idx = block_idx // NUM_BLOCKS_N
            task_n_idx = block_idx % NUM_BLOCKS_N
            m_start = task_m_idx * BLOCK_M
            n_start = task_n_idx * BLOCK_N
            
            mat_c_block = tl.zeros((BLOCK_M, BLOCK_N),dtype = tl.float32)
            for k_start in range(0, K, BLOCK_K):
                mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
                    k_start + tl.arange(0, BLOCK_K)
                )[None, :]
                mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                    (k_start + tl.arange(0, BLOCK_K)) < K
                )[None, :]
                mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
                tl.compile_hint(mat_a_block, "dot_pad_only_k")
                mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
                    n_start + tl.arange(0, BLOCK_N)
                )[None, :]
                mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
                    (n_start + tl.arange(0, BLOCK_N)) < N
                )[None, :]
                mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
                tl.compile_hint(mat_b_block, "dot_pad_only_k")
                mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
            mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
                n_start + tl.arange(0, BLOCK_N)
            )[None, :]
            mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
                (n_start + tl.arange(0, BLOCK_N)) < N
            )[None, :]
            tl.store(mat_c + mat_c_offset, mat_c_block.to(tl.bfloat16), mask = mat_c_mask)
```
