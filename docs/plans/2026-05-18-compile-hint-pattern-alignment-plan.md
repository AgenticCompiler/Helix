# Compile Hint Pattern Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reframe the authored `compile_hint` pattern card so its name, title, summary, and generated index entry all describe the same late-stage compiler-hint strategy.

**Architecture:** Keep the existing `compile_hint` taxonomy stable and update only the authored pattern card plus the generated pattern index. Reuse the stronger framing already present in older optimize-knowledge variants instead of inventing a new split or rename.

**Tech Stack:** Markdown pattern cards, Python pattern-index generator

---

### Task 1: Rewrite The Authored Pattern Card

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/skills/triton/triton-npu-optimize-knowledge/references/patterns/compile_hint.md`
- Reference: `/Users/cdj/Projects/triton-agent/skills/triton/triton-npu-optimize-knowledge-v3/references/patterns/compile_hint.md`

- [ ] **Step 1: Rewrite the title and opening sections to match the broader hinting strategy**

Replace the current compressed framing with a title and opening sections that explicitly present `compile_hint` as one late-stage compiler/lowering hint pattern covering `dot_pad_only_k`, `multiple_of`, and `max_contiguous`.

- [ ] **Step 2: Preserve the existing examples while tightening the pattern semantics**

Keep the existing detail examples, but make the authored card clearly state:

```text
- use hints only after structure is already stable
- use only provable alignment/contiguity assumptions
- treat parent-vs-parent comparison as the success criterion
```

- [ ] **Step 3: Review the rewritten card for contract compliance**

Verify the card still contains:

```text
## Summary
## Use When
```

and that `## Summary` describes what the pattern is while `## Use When` describes when to apply it.

### Task 2: Regenerate And Verify The Pattern Index

**Files:**
- Modify: `/Users/cdj/Projects/triton-agent/skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`
- Script: `/Users/cdj/Projects/triton-agent/skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py`

- [ ] **Step 1: Regenerate the checked-in pattern index**

Run:

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md
```

Expected: the generated `compile_hint` summary in `pattern_index.md` matches the new authored framing.

- [ ] **Step 2: Confirm the checked-in index is current**

Run:

```bash
python3 skills/triton/triton-npu-optimize-knowledge/scripts/build_pattern_index.py \
  --patterns-dir skills/triton/triton-npu-optimize-knowledge/references/patterns \
  --output skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md \
  --check
```

Expected: success exit with no drift reported.

- [ ] **Step 3: Review diff boundaries before reporting completion**

Confirm the change stays scoped to:

```text
- docs/specs/2026-05-18-compile-hint-pattern-alignment-design.md
- docs/plans/2026-05-18-compile-hint-pattern-alignment-plan.md
- skills/triton/triton-npu-optimize-knowledge/references/patterns/compile_hint.md
- skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md
```

and note any unrelated pre-existing worktree changes separately instead of overwriting them.
