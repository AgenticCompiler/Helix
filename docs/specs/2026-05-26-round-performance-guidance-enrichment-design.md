# Round Performance Guidance Enrichment Design

## Summary

Enrich the existing Triton Ascend round-performance analysis references with a few practical diagnosis techniques that are currently under-documented:

- theoretical lower-bound estimation for transfer and compute
- code-mapping interpretation
- `Block Dim` mismatch guidance for over- and under-partitioned kernels

Also extend the generic symptom cards so these signals can route the reader toward the right optimization patterns without changing optimize workflow ownership.

## Problem

The current round-performance references already explain how to read profiler layers and how to connect symptoms to likely implementation issues, but they stop short of several practical analysis moves that operator developers use repeatedly:

- estimating whether measured MTE or compute time is far from a simple theoretical lower bound
- using code mapping to notice when a load-looking region is unexpectedly scalar-heavy
- recognizing that `Block Dim` can be misfit because it is too large for the effective hardware width, not only because it is too small

Without this guidance, the skill still points in the right direction, but it gives less help when turning raw profiling artifacts into concrete next actions.

## Goals

- Add generic, reusable guidance for theoretical lower-bound estimation.
- Add optional code-mapping-output guidance and stronger text-artifact interpretation.
- Strengthen mapping from profiler signals to optimization directions.
- Extend existing symptom cards with evidence cues from these sources.
- Keep the guidance generic and architecture-aware instead of baking one chip's example values into universal rules.

## Non-Goals

- Do not change `triton-npu-analyze-round-performance` workflow or output contract.
- Do not add new CLI commands, parsers, or artifact formats.
- Do not create new pattern cards for this pass.
- Do not add chip-specific constants as normative defaults.

## Decision

### 1. Keep the workflow unchanged

`skills/triton-npu-analyze-round-performance/SKILL.md` stays focused on evidence order, artifact ownership, and output expectations.

### 2. Enrich the profiling-analysis reference

Update `skills/triton-npu-analyze-round-performance/references/ascend-npu-profiling-analysis.md` to cover:

- how to estimate simple transfer and compute lower bounds
- why measured transfer time may exceed the lower bound even in healthy kernels
- how to use that gap as a diagnosis hint rather than a pass/fail test
- how to read code-mapping outputs for load/store regions that are unexpectedly scalar-heavy

### 3. Enrich the optimization-guidance reference

Update `skills/triton-npu-analyze-round-performance/references/ascend-npu-optimization-guidance.md` to turn the new signals into action:

- scalar-heavy code-mapping evidence should reinforce scalar-overhead or degraded-vectorization hypotheses
- transfer time far above the lower bound, especially when the working set is small enough to fit on chip, should strengthen tiling and redundant-movement hypotheses
- `Block Dim` should be treated as a fit question, including over-partitioning for vector-heavy kernels

### 4. Extend existing symptom cards

Update the existing cards instead of creating a new symptom taxonomy:

- `high-scalar-overhead`
- `high-transfer-pressure`
- `weak-pipeline-overlap`

These cards should mention the new evidence types in concise routing language only.

### 5. Regenerate generated knowledge

After editing the symptom cards, regenerate the checked-in `symptom_index.md`.

## Rationale

This approach adds the new operator-analysis value exactly where the current skill family expects it:

- deeper reading guidance belongs in round-performance references
- reusable routing cues belong in symptom cards
- workflow instructions should stay stable unless behavior actually changes
