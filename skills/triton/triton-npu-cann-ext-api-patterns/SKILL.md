---
name: triton-npu-cann-ext-api-patterns
description: A5-only specialized optimization pattern references for Triton Ascend NPU kernels that use CANN Triton extension APIs. This skill does not define the optimize workflow; it only provides pattern material for optimize to consume.
---

# CANN Triton Extension API Patterns

## Purpose

This skill is a pattern library for optimize runs that explicitly enable CANN extension API access.

## Scope

- This skill does not define optimize workflow behavior.
- The optimize workflow contract remains owned by `triton-npu-optimize`.
- This skill only provides specialized A5-oriented pattern references for CANN Triton extension APIs.
- Treat the pattern material here as A5-specific unless the detailed pattern reference states otherwise.

## How To Use This Skill

1. Use this skill only when optimize explicitly stages it.
2. Read `references/patterns/index.md` first.
3. Pick only the most relevant detailed pattern file for the current bottleneck.
4. Avoid bulk-loading all detailed pattern references unless the kernel genuinely shows multiple independent extension-API opportunities.

## Quick Orientation

### Kernel structure

Every CV-fusion kernel has two `al.scope` blocks at the top level:

```python
with al.scope(core_mode="cube"):
    for block_idx in range(start, end, step):
        _qk_matmul(...)   # tl.dot + al.fixpipe → qk_ub
        _pv_matmul(...)   # tl.dot + al.fixpipe → pv_ub

with al.scope(core_mode="vector"):
    for block_idx in range(start, end, step):
        _softmax(...)     # qk_ub → p_nz → al.copy → p_l1
        _flash_update(...)# pv_ub → acc
```

Buffers (`qk_ub`, `p_l1`, `pv_ub`) are allocated with `bl.alloc` **before** the scopes and are visible in both.

### Sync handshake pattern

For each buffer handoff between cube and vector, issue a matching `set`/`wait` pair:

```python
# Cube done writing → vector can read:
al.sync_block_set("cube", "vector", event_id, PIPE_FIX, PIPE_V)   # in cube scope
al.sync_block_wait("cube", "vector", event_id, PIPE_FIX, PIPE_V)  # in vector scope

# Vector done writing → cube can read:
al.sync_block_set("vector", "cube", event_id, PIPE_MTE3, PIPE_MTE1)   # in vector scope
al.sync_block_wait("vector", "cube", event_id, PIPE_MTE3, PIPE_MTE1)  # in cube scope
```

See [references/patterns/al_sync.md](references/patterns/al_sync.md) for the full event ID table and pre/post-loop initialization.

### NZ format — key sizes

| Dtype | N0 | M0 | p_l1 shape for [BM, BN] |
|---|---|---|---|
| fp16/bf16 | 16 | 16 | `[BN//16, BM//16, 16, 16]` |
| fp32 | 8 | 16 | `[BN//8, BM//16, 16, 8]` |

Set at kernel launch: `fractal_n0 = 8 if dtype == float32 else 16`.

### fp32 NZ conversion: always outside outlined scopes

For fp32, build softmax output as ND (`p_temp`) inside the VF scope, then convert to NZ with `tl.permute(reshape(...), (2,0,1,3))` outside. See [references/patterns/al_scope_args.md](references/patterns/al_scope_args.md) and [references/patterns/al_copy_fractal.md](references/patterns/al_copy_fractal.md).

### Ping-pong buffering

Allocate `_ping` and `_pong` variants of `qk_ub`, `p_l1`, `pv_ub`. Select with `(sid & 1)` and `(pvid & 1)`. Requires 6 sync helper functions and pre/post-loop credit initialization. See [references/patterns/al_sync.md](references/patterns/al_sync.md).

### 1:2 sub_vec_id lane split

```python
sub_vec_id = al.sub_vec_id()
p_l1_sub = bl.subview(p_l1, [0, sub_vec_id * ((BLOCK_M//2)//FRACTAL_M0), 0, 0], ...)
al.copy(bl.to_buffer(p_nz_4d, UB), p_l1_sub)
```

Each vector lane writes its half of the P matrix to a disjoint L1 sub-region; cube reads the full `[BM, BN]` tile. See [references/patterns/sub_vec_id_1to2.md](references/patterns/sub_vec_id_1to2.md).

### Kernel complexity levels

| Kernel | Sync events | Buffers | Extra features |
|---|---|---|---|
| `fa_fwd_Affinity` | 3 (0, 1, 2) | single-buffer | basic sub_vec_id, non-outlined VF |
| `fa_fwd_parallel` | 12 (0,2,4,6,8,10) | ping-pong | outlined VF scopes, no_inline alpha |
| `fa_fwd_preload` | 12×2 (doubled) | ping-pong + L0C | task-ring, al.parallel, triple-buffer m/l/alpha |

Start with `fa_fwd_Affinity` as the template when adding a new kernel; escalate to `fa_fwd_parallel` when ping-pong is needed for throughput.

## Reading Contract

- Treat this skill as reference material only.
- Follow optimize workflow, validation, and reporting rules from `triton-npu-optimize`.
- Use the detailed pattern files only when round evidence supports the rewrite direction.
