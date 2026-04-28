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

## Reading Contract

- Treat this skill as reference material only.
- Follow optimize workflow, validation, and reporting rules from `triton-npu-optimize`.
- Use the detailed pattern files only when round evidence supports the rewrite direction.
