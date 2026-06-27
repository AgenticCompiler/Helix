# Latency Optimizer Skill Fusion Design

## Context

The external `latency-optimizer` skill under `/home/wxx/AgentWorkSpace/Triton_Skills-main/skills/latency-optimizer` contains Ascend Triton latency optimization guidance and many focused reference documents. This repository already has a `triton-npu-optimize` skill that owns optimize workflow semantics, round validation, profiling escalation, and a pattern reference library.

## Decision

Fuse the external skill as optimize pattern knowledge instead of adding a new top-level skill.

The current repository treats `skills/triton-npu-optimize` as the public optimization entrypoint. Adding a separate `latency-optimizer` skill would duplicate the workflow loop, verification authority, and pattern-selection role. The external skill also contains constraints that conflict with current project semantics, such as restricting agents to only its references and invoking a different verifier workflow directly.

## Integration Shape

- Keep `triton-npu-optimize/SKILL.md` as the optimize workflow owner.
- Add grouped pattern references under `skills/triton/triton-npu-optimize/references/patterns/` for latency-specific knowledge that is not already covered.
- Update the pattern index with symptom-based entries and boundaries.
- Route validation through the existing round and `compare-perf` authority instead of direct verifier commands from the external skill.
- Preserve English project-facing prose and avoid copying Chinese source text verbatim.

## Pattern Mapping

Already represented by existing references:

- `auto_tiling.md` maps to `autotune.md` and `tiling.md`.
- `load-order.md` maps to `reorder-load.md`.
- `task_single_row_to_multi_rows.md` maps to `program-multiple-rows.md`.
- `vector_compare.md` maps to `vec-cmp.md`.
- Parts of `optimization-patterns.md` map to `tiling.md`, `cache_use.md`, `classic-matmul.md`, and `compile_hint.md`.

New grouped references:

- `scalar-latency-traps.md`: constexpr parameters, modulo removal, loop pointer recurrence removal, single-position `tl.where`, int32 vector arithmetic, and cumsum axis splitting.
- `layout-store-and-block-pointers.md`: store merge, store transpose degradation, inner-dimension vectorization, high-dimensional block pointers, vec-to-cube transpose ordering, and matmul-bias handling.
- `grid-flatten-and-ub-buffering.md`: flatten-parallel, logical-to-physical grid mapping, UB aggregate write, and UB bulk read.
- `attention-cv-pipeline.md`: Cube/Vector pipeline scheduling, precomputed masks, scale-mask fusion, exp-vs-exp2 softmax consistency, and A5 compile parameters.

## Validation

Add tests that assert the optimize skill exposes the fused latency pattern groups through the pattern index and that each new reference contains the key trigger terms needed by agents.
