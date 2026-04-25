# Docs Layout

This repository keeps documentation under `docs/`, grouped by purpose instead of letting every dated note accumulate at the top level.

## Layout

- `docs/specs/`
  - short design documents written before behavior changes
- `docs/plans/`
  - implementation plans used to execute approved designs
- `docs/notes/`
  - dated behavior, workflow, backend, and refactor notes
- `docs/reviews/`
  - audits, review reports, and similar historical analysis
- `docs/bug-reports/`
  - bug review and regression-tracking documents

## Naming

- Keep spec and plan filenames aligned with their topic.
- Keep note and review documents date-prefixed when they capture a point-in-time decision or snapshot.
- Prefer moving a document into the right category over adding more unrelated files directly under `docs/`.
