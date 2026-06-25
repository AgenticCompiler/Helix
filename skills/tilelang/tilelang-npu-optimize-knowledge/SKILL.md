---
name: tilelang-npu-optimize-knowledge
description: Reference-only TileLang optimize knowledge for pattern triage and evidence-backed symptom routing. This skill does not define optimize workflow or own round artifacts.
---

# TileLang Optimize Knowledge

Reference-only optimize knowledge for TileLang Ascend NPU kernel optimization.

## Purpose

- Provide pattern cards and symptom references for TileLang kernel optimization.
- Used by `tilelang-npu-optimize` during the pattern triage phase.
- Read-only reference; does not own workflow or round artifacts.

## Content (to be populated)

Patterns and symptoms that are TileLang-specific should be added under `references/patterns/` and `references/symptoms/`. Common Ascend NPU patterns that apply regardless of kernel language can reference `npu-analyze-round-performance`.

## Related skills

- `tilelang-npu-optimize` — the optimize workflow that consumes this knowledge.
