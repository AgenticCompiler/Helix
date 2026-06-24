---
name: torch-npu-optimize-knowledge
description: Reference-only Torch NPU optimize pattern library for operator-target triage. This skill does not define optimize workflow or own round artifacts.
---

# Torch NPU Optimize Knowledge

## Purpose

This skill is a pattern library for optimize runs that explicitly stage Torch NPU operator-level pattern references.

## Scope

- This skill is reference-only.
- This skill does not define optimize workflow behavior.
- This skill does not own `opt-round-N/perf-analysis.md`, `attempts.md`, `summary.md`, or `opt-note.md`.
- `triton-npu-optimize` owns optimize workflow and validation rules.
- `npu-analyze-round-performance` owns round-level performance diagnosis.

## How To Use This Skill

1. Use this skill only when optimize explicitly stages Torch NPU operator-level guidance.
2. Read `references/pattern_index.md` first.
3. Read only the one or two most relevant detailed pattern files after the index narrows the candidate set.

## Reading Contract

- Treat pattern cards as routing aids, not a hard rule engine.
- Return to the caller skill for diagnosis, optimization choice, and recordkeeping.
- Use this skill only when the caller explicitly stages Torch NPU guidance.
