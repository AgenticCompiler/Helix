---
name: triton-npu-optimize-knowledge-v3
description: Generic reference-only optimize knowledge (v3 working copy) for pattern triage and evidence-backed symptom routing. This skill does not define optimize workflow or own round artifacts; prefer this tree for ongoing knowledge updates.
---

# Optimize Knowledge (v3)

## Purpose

This skill is the generic optimize knowledge library for reusable pattern and symptom references. It is a **copy of `triton-npu-optimize-knowledge`** intended as the **v3 update target**; keep `triton-npu-optimize-knowledge` unchanged unless you are intentionally aligning the bases.

## Scope

- This skill is reference-only.
- This skill does not define optimize workflow behavior.
- This skill does not own `opt-round-N/perf-analysis.md`, `attempts.md`, `summary.md`, or `opt-note.md`.
- `triton-npu-optimize` owns optimize workflow and validation rules.
- `npu-analyze-round-performance` owns round-level performance diagnosis.
- For bench-log-to-pattern synthesis workflow (including full-card rewrites and removal of temporary inventory/narrative scaffolding), follow `skills/triton-npu-kernel-bench-logs/SKILL.md` rather than inventing a local process.
- During synthesis rewrites, preserve valid pre-existing pattern knowledge (examples, technique catalogs, implementation notes) unless round evidence clearly invalidates it.

## Reading Order

1. For code-structure-first triage, read `references/pattern_index.md`.
2. For profile- or IR-backed routing, read `references/symptom_index.md`.
3. Read only the one or two most relevant detailed cards after the index narrows the candidate set.

## Reasoning Rules

- Treat pattern cards and symptom cards as routing aids, not a hard rule engine.
- Return to the caller skill for diagnosis, optimization choice, and recordkeeping.
- Keep specialized packs such as `triton-npu-cann-ext-api-patterns` separate unless the caller explicitly needs them.
