# Docs Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move repo-owned spec and plan documents into `docs/specs/` and `docs/plans/`, then update repository guidance so future documents use the new locations.

**Architecture:** Treat this as a docs-only migration. Create the new target directories first, move the existing files without changing their filenames, then update the repo guidance that controls future document placement and verify that no stale old-layout references remain.

**Tech Stack:** Markdown docs, shell file moves, `rg`, repository verification commands

---

### Task 1: Create The New Docs Directories And Capture The Target State

**Files:**
- Create: `docs/specs/`
- Create: `docs/plans/`
- Modify: `docs/specs/2026-04-09-docs-layout-design.md`
- Modify: `docs/plans/2026-04-09-docs-layout.md`

- [ ] **Step 1: Confirm the existing docs inventory**

Run: `find docs/superpowers -maxdepth 3 -type f | sort`
Expected: A concrete list of files currently under `docs/superpowers/plans/` and `docs/superpowers/specs/`

- [ ] **Step 2: Create the destination directories**

Run: `mkdir -p docs/specs docs/plans`
Expected: The destination directories exist

- [ ] **Step 3: Keep this spec and plan in the new locations**

Expected: The migration spec is stored under `docs/specs/` and the plan is stored under `docs/plans/`

### Task 2: Move Existing Plan And Spec Files

**Files:**
- Modify: `docs/specs/*`
- Modify: `docs/plans/*`
- Delete: `docs/superpowers/plans/*`
- Delete: `docs/superpowers/specs/*`

- [ ] **Step 1: Move plan files into `docs/plans/`**

Run: `mv docs/superpowers/plans/*.md docs/plans/`
Expected: The plan files now live under `docs/plans/`

- [ ] **Step 2: Move spec files into `docs/specs/`**

Run: `mv docs/superpowers/specs/*.md docs/specs/`
Expected: The spec files now live under `docs/specs/`

- [ ] **Step 3: Remove empty legacy directories**

Run: `rmdir docs/superpowers/specs docs/superpowers/plans docs/superpowers`
Expected: Empty legacy directories are removed cleanly

### Task 3: Update Repo Rules For Future Documents

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the durable docs-location rules**

Add explicit repo rules such as:

```md
- Keep design/spec documents under `docs/specs/`.
- Keep implementation plans under `docs/plans/`.
- Keep behavior and focused workflow documents under `docs/notes/`.
```

- [ ] **Step 2: Re-read the updated rules**

Expected: `AGENTS.md` clearly prevents future specs and plans from being written under `docs/superpowers/`

### Task 4: Verify The Migration

**Files:**
- Modify: none

- [ ] **Step 1: Search for stale old-layout references**

Run: `rg -n "docs/superpowers/plans|docs/superpowers/specs" .`
Expected: No repo-owned references remain

- [ ] **Step 2: Verify the new file layout**

Run: `find docs -maxdepth 3 -type f | sort`
Expected: Plan files are under `docs/plans/` and spec files are under `docs/specs/`

- [ ] **Step 3: Run lint**

Run: `uv run --group dev ruff check`
Expected: PASS

- [ ] **Step 4: Run static typing**

Run: `uv run pyright`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `uv run python -m unittest discover -s tests -v`
Expected: PASS
