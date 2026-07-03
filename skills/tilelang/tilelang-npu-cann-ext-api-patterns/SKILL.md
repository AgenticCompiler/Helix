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

## Expert API Reference

This skill uses the Expert programming model. Read the Expert API docs before writing any Expert-mode kernel:

- [tilelang-memory-expert.md](../tilelang-npu-api-reference/references/tilelang-memory-expert.md) — `T.alloc_ub`, `T.alloc_L1`, `T.alloc_L0A/L0B/L0C`, Expert `pass_configs`
- [tilelang-compute-expert.md](../tilelang-npu-api-reference/references/tilelang-compute-expert.md) — `T.tile.*`, `T.barrier_all`, `T.set_flag`/`T.wait_flag`, `T.set_cross_flag`/`T.wait_cross_flag`

The Developer docs ([memory-developer](../tilelang-npu-api-reference/references/tilelang-memory-developer.md), [compute-developer](../tilelang-npu-api-reference/references/tilelang-compute-developer.md), [kernel-basics](../tilelang-npu-api-reference/references/tilelang-kernel-basics.md)) cover the shared infrastructure — you should already be familiar with those from convert.

## How To Use This Skill

1. Use this skill only when optimize explicitly stages it.
2. Read `references/pattern_index.md` first.
3. Pick only the most relevant detailed pattern file for the current bottleneck.
4. Avoid bulk-loading all detailed pattern references unless the kernel genuinely shows multiple independent Expert-mode opportunities.

## Lint Verification

Expert-mode kernels with manual sync are error-prone. **Always run the sync linter after applying any pattern**, before attempting on-device compilation:

```bash
python3 scripts/tl_sync_lint.py --tier1 --tier2 --tier3 --tier4 <kernel>.py
```

The linter has four tiers; map findings to on-device symptoms:

| Tier | Flag | Symptom on device |
|------|------|-------------------|
| **0** | `FLAG_IMBALANCE`, `CROSS_DEADLOCK`, `NO_INTRA_FENCE` | **Deadlock** — AICore timeout (`507014`), ~10 min hang |
| **1** (`--tier1`) | `DATA_RACE` | **Precision / correctness errors** — output mismatches ref |
| **2** (`--tier2`) | `LATE_RELEASE` | **Throughput-collapse deadlock** — buffer-reuse credit released too late |
| **3** (`--tier3`) | `PRIME_UNDERFLOW` | **First-iteration deadlock** — unconditional wait_flag whose credit is only primed under a guard |
| **4** (`--tier4`) | `SHARED_MAILBOX` | **Illegal vec config crash** — alloc_shared used as cross-lane mailbox (error `507015`) |

Tier 0 checks set/wait flag balance (liveness). Tier 1 checks happens-before between cross-pipe buffer accesses. Tier 2 catches deferred-credit late releases. Tier 3 catches first-iteration credit underflow. Tier 4 detects illegal cross-vid memory sharing.

## Patterns

All Expert-mode patterns are documented in `references/patterns/`. Start with [pattern_index.md](references/pattern_index.md) then read the specific pattern file for your bottleneck. If patterns don't cover what you need, explore the full Expert API docs linked above — the memory-expert and compute-expert references cover the complete surface.

## Reading Contract

- Treat this skill as reference material only.
- Follow optimize workflow, validation, and reporting rules from `tilelang-npu-optimize`.
- Use the detailed pattern files only when round evidence supports the rewrite direction.
