# Compile Hint Pattern Alignment Design

## Summary

Re-align the authored `compile_hint` optimize pattern card so its title, summary, and usage guidance match the broader strategy the card actually teaches.

Keep the existing file path and pattern identifier, but rewrite the card to clearly present `tl.compile_hint(..., "dot_pad_only_k")`, `tl.multiple_of(...)`, and `tl.max_contiguous(...)` as one late-stage compiler-hint pattern rather than as a narrowly named single-API note.

## Goal

Reduce reader confusion caused by the current mismatch between the `compile_hint` name and the card body, while preserving the current knowledge taxonomy and cross-references.

## User-Visible Behavior

- The `compile_hint` pattern card continues to exist at the same authored path.
- The card title and opening sections explicitly frame the pattern as late-stage compiler or lowering hints.
- The card keeps all three existing hint families in one place:
  - `dot_pad_only_k`
  - `multiple_of`
  - `max_contiguous`
- The generated pattern index summary for `compile_hint` reflects the revised framing.

## Design

### Card Framing

The current card reads as if `compile_hint` should only describe the `tl.compile_hint(...)` API, but the actual content groups several related mechanisms that all communicate stronger layout facts to the compiler.

This change should make that grouping explicit instead of narrowing the content to the literal API name. The card should describe one practical optimization pattern:

- apply small, provable compiler/lowering hints only after the main kernel structure is already stable
- use those hints to encode alignment, contiguity, or dot-padding facts the compiler may not infer safely

The title should therefore become more semantically aligned with the content, for example by referring to compiler hints rather than only `compile_hint` as an API surface.

### Taxonomy Stability

This change should keep the authored filename `compile_hint.md` and preserve the generated pattern identifier `compile_hint`.

The repository already references this pattern name from other cards and generated indexes, and the mismatch is better solved by reframing the card than by renaming or splitting it in this pass.

### Content Shape

The authored card should keep the existing pattern-card contract:

- `## Summary`
- `## Use When`
- optional supporting sections such as `## Avoid When`, `## Signals`, `## What To Verify After Applying`, and detail/examples

The rewritten `## Summary` should describe what the pattern is: a late-stage hinting pattern that covers both `tl.compile_hint` and alignment/contiguity assertions.

The rewritten `## Use When` should describe when to apply it: only after structure is already strong, and only when the asserted facts are provable for the active path.

### Index Regeneration

After the card is updated, regenerate the checked-in `pattern_index.md` so the generated summary matches the new framing.

## Validation

- Review the rewritten `compile_hint.md` for title/content alignment.
- Regenerate `skills/triton/triton-npu-optimize-knowledge/references/pattern_index.md`.
- Run the pattern-index generator with `--check` to confirm the checked-in index is current.

## Scope Boundaries

- Do not rename `compile_hint.md` in this change.
- Do not split the content into multiple new pattern cards in this change.
- Do not change the meaning of other pattern cards that reference `compile_hint`.
- Do not hand-edit `pattern_index.md`; regenerate it from the authored card.
