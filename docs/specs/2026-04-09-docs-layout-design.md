# Docs Layout Design

## Summary

- Move repository-owned planning documents from `docs/superpowers/plans/` to `docs/plans/`.
- Move repository-owned design/spec documents from `docs/superpowers/specs/` to `docs/specs/`.
- Keep behavior and design notes that already live directly under `docs/` unchanged.
- Update repository rules so future work stores specs under `docs/specs/` and implementation plans under `docs/plans/`.

## Goals

- Keep all repository documentation under `docs/`.
- Remove the mixed convention where some repo docs live under `docs/` and others under `docs/superpowers/`.
- Make future document placement explicit in repo-owned guidance.

## Non-Goals

- Do not rewrite the content of existing design or plan documents beyond path-related updates.
- Do not reorganize unrelated doc categories such as `docs/bug-reports/`.
- Do not modify external superpowers skill files outside this repository.

## Target Layout

- `docs/specs/`
  - repository design/spec documents used before implementation
- `docs/plans/`
  - repository implementation plans used during execution
- `docs/`
  - user-visible behavior, design notes, backend notes, and focused workflow docs

## Migration Rules

- Every file currently under `docs/superpowers/plans/` moves to `docs/plans/` with the same filename.
- Every file currently under `docs/superpowers/specs/` moves to `docs/specs/` with the same filename.
- Empty legacy directories under `docs/superpowers/` are removed after the move.
- Repo-owned references to the old layout are updated to the new layout.

## Durable Repo Rules

- New design/spec docs go under `docs/specs/`.
- New implementation plans go under `docs/plans/`.
- User-visible behavior docs continue to live under `docs/` with date-prefixed names when appropriate.
- `AGENTS.md` should document the new locations so future work does not recreate the old split layout.

## Verification

- Confirm the expected files exist under `docs/specs/` and `docs/plans/`.
- Confirm the moved files no longer exist under `docs/superpowers/`.
- Search the repository for stale `docs/superpowers/plans` and `docs/superpowers/specs` references.
- Run the repository verification commands after the migration.
