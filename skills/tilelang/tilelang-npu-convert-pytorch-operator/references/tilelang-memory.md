# TileLang Memory Allocation

| Level | API | Purpose |
|-------|-----|---------|
| Parameter | `T.Tensor((M, N), dtype)` | Kernel I/O, shape-and-dtype declared |
| UB (developer) | `T.alloc_shared((M, N), dtype)` | Unified Buffer, type inferred by compiler |
| UB (expert) | `T.alloc_ub((M, N), dtype)` | Unified Buffer, explicit allocation |
| L1 (Cube input) | `T.alloc_L1((M, N), dtype)` | L1 buffer for gemm operands |
| L0A (Cube) | `T.alloc_L0A((M, N), dtype)` | Cube left-matrix fragment |
| L0B (Cube) | `T.alloc_L0B((M, N), dtype)` | Cube right-matrix fragment |
| L0C (Cube) | `T.alloc_L0C((M, N), dtype)` | Cube accumulator fragment |
| Fragment (dev) | `T.alloc_fragment((M, N), dtype)` | Developer-mode alias for fragment |
| Scalar var | `T.alloc_var(dtype, init=value)` | Local scalar variable |
