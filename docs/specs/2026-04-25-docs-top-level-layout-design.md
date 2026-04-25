# Docs Top-Level Layout Design

## Summary

- Move the remaining date-prefixed topic documents out of the `docs/` root into `docs/notes/`.
- Move the design review snapshot out of the `docs/` root into `docs/reviews/`.
- Add a lightweight `docs/README.md` so the layout is discoverable without reading `AGENTS.md`.
- Update durable repo guidance so future work does not recreate a crowded `docs/` root.

## Goals

- Make `docs/` easier to scan by keeping the top level category-oriented.
- Preserve the existing distinction between specs, plans, bug reports, and other topical notes.
- Keep this cleanup as a path migration rather than a content rewrite.

## Non-Goals

- Do not rewrite the historical content of the moved documents.
- Do not create deep category trees for individual features.
- Do not reorganize `docs/specs/`, `docs/plans/`, or `docs/bug-reports/`.

## Target Layout

- `docs/specs/`
  - design documents
- `docs/plans/`
  - implementation plans
- `docs/notes/`
  - dated behavior, backend, workflow, and refactor notes
- `docs/reviews/`
  - audits and review snapshots
- `docs/bug-reports/`
  - bug-focused reports

## Migration Rules

- Every date-prefixed topical document currently in the `docs/` root moves into `docs/notes/` unchanged.
- The historical design review snapshot moves into `docs/reviews/` unchanged.
- Repo-owned references to the moved files are updated to their new paths.
- Future repo guidance should point new topical documents at `docs/notes/` instead of the `docs/` root.

## Verification

- Confirm the `docs/` root contains category directories plus the layout index.
- Search for stale references to old `docs/<date>-*.md` paths.
- Confirm the moved files exist under `docs/notes/` and `docs/reviews/`.
