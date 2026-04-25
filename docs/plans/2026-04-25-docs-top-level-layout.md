# Docs Top-Level Layout Plan

**Goal:** Clear the crowded `docs/` root by moving dated topical documents into category directories and updating repo guidance to match.

**Architecture:** Treat this as a docs-only migration. Add the layout index and durable rules first, then move files without rewriting their content, and finally update stale references.

**Tech Stack:** Markdown docs, shell moves, `rg`

## Task 1: Document The Target Layout

**Files:**
- Modify: `AGENTS.md`
- Create: `docs/README.md`
- Create: `docs/specs/2026-04-25-docs-top-level-layout-design.md`
- Create: `docs/plans/2026-04-25-docs-top-level-layout.md`

- [ ] Add durable repo rules for `docs/notes/` and `docs/reviews/`.
- [ ] Add a short `docs/README.md` index describing the layout.

## Task 2: Move The Scattered Top-Level Docs

**Files:**
- Create: `docs/notes/*`
- Create: `docs/reviews/*`
- Delete: moved date-prefixed files from the `docs/` root

- [ ] Create `docs/notes/` and `docs/reviews/`.
- [ ] Move the date-prefixed topical docs from `docs/` into `docs/notes/`.
- [ ] Move the design review snapshot from `docs/` into `docs/reviews/`.

## Task 3: Update References

**Files:**
- Modify: repo-owned Markdown files that still reference old top-level doc paths

- [ ] Rewrite references from `docs/YYYY-MM-DD-...` to `docs/notes/YYYY-MM-DD-...` for moved topical docs.
- [ ] Rewrite references to the review snapshot to `docs/reviews/2026-04-16-design-doc-review.md`.

## Task 4: Verify

- [ ] Confirm the new layout with `find docs -maxdepth 2 -type f | sort`.
- [ ] Search for stale references to the old top-level paths with `rg`.
