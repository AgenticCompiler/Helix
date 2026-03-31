import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * block_size_m + tl.arange(0, block_size_m)
    offs_n = pid_n * block_size_n + tl.arange(0, block_size_n)
    offs_k = tl.arange(0, block_size_k)

    accumulator = tl.zeros((block_size_m, block_size_n), dtype=tl.float32)

    for k_start in range(0, k, block_size_k):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < m) & ((k_start + offs_k[None, :]) < k)
        b_mask = ((k_start + offs_k[:, None]) < k) & (offs_n[None, :] < n)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)

    c = accumulator.to(tl.float32)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)

    grid = (
        triton.cdiv(m, 16),
        triton.cdiv(n, 16),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        block_size_m=16,
        block_size_n=16,
        block_size_k=16,
    )
    return c

