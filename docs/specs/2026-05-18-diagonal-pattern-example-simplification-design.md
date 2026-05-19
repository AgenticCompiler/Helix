# Diagonal Pattern Example Simplification Design

## Summary

Simplify the example in `skills/triton-npu-optimize-knowledge/references/patterns/diagonal.md` so the card teaches the diagonal traversal idea directly instead of embedding a full, duplicated matmul implementation.

The revised card should keep the optimization meaning intact while making the example short enough to scan during pattern triage.

## Goal

Make the `diagonal` pattern example explain the traversal change clearly:

- what ordinary block traversal looks like
- what diagonal traversal changes
- when the diagonal mapping is worth considering

## User-Visible Behavior

- The pattern summary, `Use When`, and `Signals` sections remain semantically the same.
- The long `Detail` section is replaced by a compact explanation plus a smaller example.
- The example keeps Triton-flavored code so the card still feels actionable to optimize agents.
- The example focuses on `block_idx -> (task_m_idx, task_n_idx)` mapping, not a full matmul walkthrough.
- The repeated load / dot / store body is collapsed into a short placeholder comment so the example emphasizes traversal rather than arithmetic details.

## Design

### Example Scope

The simplified example should show one minimal kernel skeleton with:

- block-count setup
- the threshold gate that decides whether diagonal traversal is enabled
- a compact helper or inline mapping for ordinary traversal
- a compact helper or inline mapping for diagonal traversal
- one short comment that the resulting `(task_m_idx, task_n_idx)` drives the usual tile compute body

This keeps the example aligned with real Triton kernels while removing the repeated masked load and store mechanics that do not help explain the pattern choice.

### Content To Remove

Remove or compress:

- the full duplicated matmul accumulation body in both branches
- the oversized block-numbering narrative embedded inside a triple-quoted comment
- the inline greatest-common-divisor / least-common-multiple arithmetic when it is not necessary to show the idea

If the card still needs one concrete picture of diagonal ordering, prefer a tiny illustrative grid in prose rather than a long embedded tutorial comment.

## Validation

- Re-read `diagonal.md` to confirm the simplified example still explains when diagonal traversal changes behavior.
- Verify no generated files need regeneration; `pattern_index.md` should remain untouched because the card summary and `Use When` contract are not changing.

## Scope Boundaries

- Do not change the pattern's title, summary meaning, or `Use When` criteria.
- Do not turn the card into a generic matmul tutorial.
- Do not update unrelated pattern cards or generated indexes unless a verification step proves it is necessary.
