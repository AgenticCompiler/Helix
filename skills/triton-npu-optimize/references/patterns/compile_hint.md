# Compiler Hint Pattern

## Summary

Try the following compile hints:

  - Before call to `tl.dot` in matrices, use `tl.compile_hint(a, "dot_pad_only_k")` for matrices
    involved in the product.
  - Use `tl.multiple_of` (resp. `tl.max_contiguous`) to specify tensor slices that are known to be
    aligned (resp. contiguous).

## Use When

- The kernel structure already looks close to good, but the compiler still lacks explicit alignment or contiguity information.
- `tl.dot` tiles, slices, or pointer math are known to satisfy stronger layout assumptions than the code currently expresses.

## Signals

### Code

- `tl.dot` inputs are already aligned in `M` and `N`, so only the `K` direction still needs padding hints.
- Pointer slices are known contiguous or aligned, but the code does not yet communicate that with `tl.max_contiguous` or `tl.multiple_of`.

## What To Verify After Applying

- Verify the alignment or contiguity assumptions encoded in the hint are actually true for the rewritten slices.
- Verify the compiler hints changed lowering or performance without changing the logical result.

## Detail

### dot_pad_only_k

Try using "dot_pad_only_k" to specify that in a Cube operation, only the `k` direction need
to be padded (the `m` and `n` directions are already aligned). For example: in the following
code for matmul:

```python
for k_start in range(0, K, BLOCK_K):
    mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
        k_start + tl.arange(0, BLOCK_K)
    )[None, :]
    mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
        (k_start + tl.arange(0, BLOCK_K)) < K
    )[None, :]
    mat_a_block = tl.load(mat_a + mat_a_offset, mask = mat_a_mask, other = 0.0)
    tl.compile_hint(mat_a_block, "dot_pad_only_k")   # add compile hint
    mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + ( 
        n_start + tl.arange(0, BLOCK_N)
    )[None, :]
    mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
        (n_start + tl.arange(0, BLOCK_N)) < N
    )[None, :]
    mat_b_block = tl.load(mat_b + mat_b_offset, mask = mat_b_mask, other = 0.0)
    tl.compile_hint(mat_b_block, "dot_pad_only_k")  #add compile hint
    mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)
```

### max_contiguous and multiple_of

Set `tl.max_contiguous` to specify the loaded data is contiguous. Set `tl.multiple_of`
to specify the loaded data is aligned up to multiple of the second parameter. For example:

```python
@triton.jit
def write_req_to_token_pool_triton_optimize(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices,
    pre_lens,
    seq_lens,
    extend_lens,
    out_cache_loc,
    req_to_token_ptr_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_token = tl.program_id(1)

    req_pool_index = tl.load(req_pool_indices + pid_batch)
    pre_len = tl.load(pre_lens + pid_batch)
    seq_len = tl.load(seq_lens + pid_batch)
    extend_len = seq_len - pre_len

    cumsum_start = 0
    for i in range(pid_batch):
        cumsum_start += tl.load(extend_lens + i)

    token_start = pid_token * BLOCK_SIZE

    offset = tl.arange(0, BLOCK_SIZE)
    actual_offset = token_start + offset
    mask = actual_offset < extend_len

    src_ptr = out_cache_loc + cumsum_start + actual_offset
    src_ptr = tl.max_contiguous(tl.multiple_of(src_ptr, BLOCK_SIZE), BLOCK_SIZE)  # used here
    value = tl.load(src_ptr, mask=mask)
    dst_ptr = (
        req_to_token_ptr
        + req_pool_index * req_to_token_ptr_stride
        + actual_offset
        + pre_len
    )
    dst_ptr = tl.max_contiguous(tl.multiple_of(dst_ptr, BLOCK_SIZE), BLOCK_SIZE)  # used here

    tl.store(dst_ptr, value, mask=mask)
```
