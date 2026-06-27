---
name: tilelang-npu-cann-ext-api-patterns
description: Expert-mode optimization pattern references for TileLang Ascend NPU kernels that use explicit hardware memory, manual synchronization, and CV scope separation. This skill does not define the optimize workflow; it only provides pattern material for optimize to consume.
---

# TileLang Expert-Mode Optimization Patterns

## Purpose

This skill provides pattern references for optimize runs that use the Expert programming model — explicit hardware memory allocation, manual synchronization, and Cube/Vector scope separation.

## Scope

- This skill does not define optimize workflow behavior.
- The optimize workflow contract remains owned by `tilelang-npu-optimize`.
- This skill provides TileLang-specific Expert-mode patterns for performance-critical kernel tuning.
- Use these patterns when the Developer-mode auto-managed approach is insufficient and the kernel needs precise hardware-level control.

## How To Use This Skill

1. Use this skill only when optimize explicitly stages it.
2. Read `references/patterns/index.md` first.
3. Pick only the most relevant detailed pattern file for the current bottleneck.
4. Avoid bulk-loading all detailed pattern references unless the kernel genuinely shows multiple independent Expert-mode opportunities.

## Quick Orientation

### Expert vs Developer Mode

| | Developer Mode | Expert Mode |
|---|---|---|
| Memory | `T.alloc_shared`, `T.alloc_fragment` | `T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C` |
| Sync | Auto (pass_configs) | `T.set_flag`/`T.wait_flag`, `T.barrier_all` |
| Scope | Compiler-inferred | `T.Scope("C")`, `T.Scope("V")` |
| Pass configs | All auto-passes ON | See below |

### Switching pass_configs for Expert Mode

When moving from Developer to Expert mode, change `pass_configs`:

```python
# Developer (default for convert)
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
}

# Expert — recommended for full manual control
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,   # required
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,         # recommended
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,    # doesn't matter
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,      # recommended
}
```

| Key | Expert | Why | What changes |
|-----|--------|-----|--------------|
| `AUTO_CV_COMBINE` | `False` **(required)** | You use `T.Scope("C")` / `T.Scope("V")` manually — if the compiler also splits, it will conflict | Must wrap Cube work in `T.Scope("C")`, Vector work in `T.Scope("V")` |
| `AUTO_SYNC` | `False` (recommended) | Turning it off forces you to see missing barriers; leaving it on only adds redundant sync, not correctness bugs | Write `T.barrier_all()`, `T.set_flag` / `T.wait_flag` by hand |
| `AUTO_CV_SYNC` | `False` (recommended) | Same — off so you verify your cross-core handshakes are complete; on just adds redundant sync | Write `T.set_cross_flag` / `T.wait_cross_flag` at Cube↔Vector boundaries |
| `MEMORY_PLANNING` | either | Pure memory optimization — no impact on correctness or your manual control | None |

### CV scope architecture

Every Expert-mode CV-fusion kernel separates Cube and Vector work with `T.Scope`:

```python
with T.Scope("C"):   # Cube Core — matrix multiply
    for k in T.serial(T.ceildiv(K, K_L1)):
        T.copy(A[...], A_L1)
        T.copy(B[...], B_L1)
        T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

with T.Scope("V"):   # Vector Core — element-wise, reductions
    T.copy(C_L0, c_ub)
    for i, j in T.Parallel(block_M, block_N):
        c_ub[i, j] = T.exp(c_ub[i, j])
    T.copy(c_ub, C[...])
```

### Sync handshake pattern

For each buffer handoff between Cube and Vector, use matching `set_flag`/`wait_flag` pairs:

```python
# MTE3 done copying → Cube can use the data
T.set_flag("mte3", "m", 0)    # in MTE pipeline
T.wait_flag("mte3", "m", 0)   # in Cube scope

# Cube done computing → MTE can copy result out
T.set_flag("m", "mte3", 1)    # in Cube scope
T.wait_flag("m", "mte3", 1)   # in MTE pipeline
```

Pipe names: `"fix"`, `"mte1"`, `"mte2"`, `"mte3"`, `"m"`, `"v"`, `"s"`, `"ALL"`

### Cross-core sync

For Cube ↔ Vector cross-core synchronization:

```python
T.set_cross_flag("MTE3", 0)
T.wait_cross_flag(0)
```

### Double-buffering

Allocate `_0` and `_1` variants of L1/UB buffers. Toggle with `cur = k % 2; nxt = 1 - cur`. Prefetch the next tile while computing the current one. Requires manual sync to coordinate MTE prefetch and Cube compute.

See [references/patterns/double-buffer.md](references/patterns/double-buffer.md).

### Explicit memory mapping

| Developer (abstract) | Expert (explicit) | Hardware |
|---|---|---|
| `T.alloc_shared` | `T.alloc_L1` | L1 Buffer |
| `T.alloc_shared` | `T.alloc_ub` | Unified Buffer |
| `T.alloc_fragment` | `T.alloc_L0C` | L0C Accumulator |
| `T.alloc_fragment` | `T.alloc_L0A` | L0A Left Operand |
| `T.alloc_fragment` | `T.alloc_L0B` | L0B Right Operand |

## Reading Contract

- Treat this skill as reference material only.
- Follow optimize workflow, validation, and reporting rules from `tilelang-npu-optimize`.
- Use the detailed pattern files only when round evidence supports the rewrite direction.
