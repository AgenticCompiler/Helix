---
name: tilelang-npu-convert-pytorch-operator
description: Convert one PyTorch operator into a TileLang NPU-backed PyTorch operator, preserve the trailing input-helper block, and validate the converted output through standalone or differential testing.
---

# Convert PyTorch Operator (TileLang)

Convert one PyTorch operator file into a PyTorch-facing operator backed by a real TileLang Ascend NPU kernel path.

## TileLang API Reference

Import pattern:
```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

Kernel definition:
```python
@T.prim_func
def kernel(A: T.Tensor((M, N), dtype), ...):
    with T.Kernel(block_count, is_npu=True) as (cid, vid):
        ...
```

Memory: `T.alloc_shared` (UB), `T.alloc_L1`, `T.alloc_L0A/L0B/L0C` (fragment)
Data movement: `T.copy(src, dst)`
Compute: `T.Parallel` (+/-/*/T.exp/T.max), `T.gemm_v0`, `T.reduce_sum/max/min`, `T.tile.*`
JIT: `@jit(out_idx=[-1])` on factory function

## Constraints

- Keep the public API PyTorch-facing.
- Target Ascend NPU only; no CUDA/CPU/MPS fallback.
- Preserve the trailing input-helper block.
- Use npu-gen-test + npu-run-eval for validation.

## TODO

- Document TileLang-specific conversion patterns (e.g., how to map common PyTorch ops to TileLang kernel structure).
- Add validation command examples for TileLang kernels.
