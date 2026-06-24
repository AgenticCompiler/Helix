# Autotune Pattern Decision Card Design

## Summary

Rewrite the high-priority `autotune` optimization pattern from a short example page into a decision card that routes agents through automatic autotune first, `hints` second, and hand-written `triton.Config` search spaces last.

The card should keep enough Triton-Ascend detail to prevent common misuse, but it should stay optimized for quick optimization decisions rather than becoming a full API manual.

After the first rewrite, the card still drifted too far toward reference-style explanation. The intended steady state is a compact decision card with minimal examples, not a long-form tutorial.

## Goal

Make the `autotune` pattern actively steer optimization rounds away from repeated manual tiling guesses and toward the preferred Triton-Ascend autotune workflow.

## User-Visible Behavior

- The `autotune` pattern remains marked `priority: high`.
- Its summary explains that the pattern is a search-space selection strategy, not just a decorator example.
- Its `Use When` section favors kernels whose performance problem is mainly unresolved block-size or split-size selection.
- The authored card explains three routes:
  - automatic autotune with `configs=[]`
  - semi-automatic autotune with `hints`
  - custom autotune with explicit `triton.Config` lists
- The card explicitly recommends the order:
  1. try automatic autotune
  2. add `hints` if the kernel is autotune-friendly but parser inference is incomplete
  3. hand-write configs only when the search space must be constrained manually
- The card avoids duplicating the same routing logic across a separate flow section, long recognition checklists, and route sections.
- The card documents Triton-Ascend-specific constraints such as preferring `BLOCK_*`, `multibuffer`, and `unit_flag`, and not treating GPU-only `num_warps` or `num_stages` as the main tuning knobs.

## Design

### Decision Structure

The rewritten card should answer one routing question first:

- is this kernel structurally ready for autotune-driven search, or does it need a different optimization pattern first

From there, it should give agents a stable escalation path:

1. Identify which `tl.constexpr` parameters are actually free to tune.
2. Use automatic autotune when split and tiling parameters can be inferred from `tl.program_id`, `tl.arange`, loop structure, and bounds masks.
3. Use `hints` when the kernel semantics still fit auto-generated search, but the DSL shape obscures the axis-to-parameter mapping.
4. Use custom `triton.Config` lists when semantic constraints or parameter coupling mean the search space must be authored manually.

This structure makes the pattern actionable during optimization triage instead of reading like a loose list of examples.

### Content Boundaries

The pattern should not copy the full source document verbatim. Instead, it should compress that material into:

- concise `Use When` and `Avoid When` bullets
- a small number of route sections that each explain when to use that route
- one compact failure-recognition section for knowing when automatic parsing is unlikely to work
- a short Ascend-specific workflow and fallback policy
- one compact example for each of the three routes

The long-form API details, exhaustive parameter descriptions, large benchmark tables, and repeated route restatements should stay out of the pattern card unless they directly change routing behavior.

### Ascend-Specific Guidance

The card should preserve the most decision-relevant platform guidance:

- Triton-Ascend autotune supports block-size-style meta-parameters and NPU-specific options such as `multibuffer`.
- Agents should not default to GPU-centric `num_warps` / `num_stages` tuning guidance on Ascend.
- Update kernels that write accumulative results may need `reset_to_zero` or equivalent hooks during autotune evaluation.
- Debugging should start with `TRITON_PRINT_AUTOTUNING=1` so the chosen routing path and generated configs are visible.

## Validation

- Regenerate `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`.
- Run the pattern-index generator with `--check`.
- Run the targeted optimize-pattern tool tests that verify checked-in index consistency.

## Scope Boundaries

- Do not hand-edit `pattern_index.md`.
- Do not turn the pattern card into a full Triton-Ascend autotune reference manual.
- Do not change symptom cards or the pattern-index generator.
- Do not weaken the recommendation to prefer autotune over repeated manual tiling experiments.
