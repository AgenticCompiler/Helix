---
name: tilelang-npu-api-reference
description: TileLang Ascend NPU API reference — shared documentation for both convert and optimize workflows.
---

# TileLang Ascend NPU API Reference

Shared API reference for TileLang Ascend NPU kernel development. Used by both convert and optimize workflows.

## Reference Documents

| File | Contents |
|------|----------|
| [tilelang-kernel-basics.md](references/tilelang-kernel-basics.md) | Shared infrastructure: `@tilelang.jit`, `T.Kernel`, loops, `pass_configs`, cache, autotune |
| [tilelang-memory-developer.md](references/tilelang-memory-developer.md) | Layer 1 memory: `T.alloc_shared`, `T.alloc_fragment`, `T.alloc_var`, `T.copy` |
| [tilelang-memory-expert.md](references/tilelang-memory-expert.md) | Layer 3 memory: `T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0*` |
| [tilelang-compute-developer.md](references/tilelang-compute-developer.md) | Layer 1 compute: `T.gemm_v0`, `T.reduce_*`, `T.Parallel` + symbolic math |
| [tilelang-compute-expert.md](references/tilelang-compute-expert.md) | Layer 2 extended `T.tile.*` + Layer 3 sync primitives |
