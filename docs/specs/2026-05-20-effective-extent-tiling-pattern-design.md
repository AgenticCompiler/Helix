# Effective-Extent Tiling Pattern Design

## Goal

Record how to capture masked-tile oversizing guidance in the optimize knowledge base with a pattern name and structure that match the real scope.

## Decision

Rename `constexpr-tile-discrete-access` to `effective-extent-tiling` and rewrite the card so it focuses on one central rule:

- choose tile widths from the live logical extent
- do not assume masks or `next_power_of_2()` automatically make an oversized tile cheap
- treat indexed paths and copy-like contiguous paths as two common manifestations of the same tiling mistake

## Why

The old name no longer matches the card body:

- `constexpr` is an implementation detail, not the optimization idea
- `discrete-access` is too narrow now that the card also covers copy-only contiguous axes

The real shared mechanism is broader:

- a tile is larger than the effective extent
- masks preserve semantics but may not reduce real loop, transfer, or vector-path work proportionally
- performance improves when the tile tracks the live extent more closely

## Required Documentation Changes

- Rename `skills/triton/triton-npu-optimize-knowledge/references/patterns/constexpr-tile-discrete-access.md` to `effective-extent-tiling.md`.
- Rewrite the card to keep:
  - the generic masked-tile rule
  - the copy-axis `next_power_of_2()` warning on Ascend
  - the user-provided exact-width vs padded-width example
- Remove repeated restatements of the same rule from multiple sections.
- Update cross-references and generated indexes to use `effective-extent-tiling`.
- Regenerate the checked-in pattern and symptom indexes.

## Non-Goals

- Do not split copy-only tail-width guidance into a separate standalone pattern.
- Do not move this content into `compile_hint`.
- Do not keep duplicate root-cause prose in `Use When`, `Problem Description`, and `Detection Pattern`.
