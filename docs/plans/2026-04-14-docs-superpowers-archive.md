# Docs Superpowers Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the remaining repository-owned documents from `docs/superpowers/` into `docs/plans/` and `docs/specs/`, then remove the empty legacy directories.

**Architecture:** Treat this as a content move, not a rewrite. Preserve filenames, move plans into `docs/plans/`, move specs into `docs/specs/`, and only adjust references if they are truly stale rather than historical.

**Tech Stack:** Markdown docs, shell move commands, repository search

---

### Task 1: Move The Remaining Files

**Files:**
- Create: `docs/plans/2026-04-13-baseline-contract-plan.md`
- Create: `docs/plans/2026-04-14-optimize-user-prompt-plan.md`
- Create: `docs/specs/2026-04-13-optimize-baseline-contract-design.md`
- Delete: `docs/superpowers/plans/2026-04-13-baseline-contract-plan.md`
- Delete: `docs/superpowers/plans/2026-04-14-optimize-user-prompt-plan.md`
- Delete: `docs/superpowers/specs/2026-04-13-optimize-baseline-contract-design.md`

- [ ] **Step 1: Verify destination filenames are available**

Run: `find docs/plans -maxdepth 1 -type f | sed 's#.*/##' | sort` and `find docs/specs -maxdepth 1 -type f | sed 's#.*/##' | sort`
Expected: none of the remaining `docs/superpowers/` filenames already exist in the destination directories

- [ ] **Step 2: Move the files**

Run: move the two plan docs into `docs/plans/` and the one spec doc into `docs/specs/`
Expected: files exist only in their new locations

- [ ] **Step 3: Remove empty legacy directories**

Run: `rmdir docs/superpowers/specs docs/superpowers/plans docs/superpowers`
Expected: `docs/superpowers/` no longer exists

### Task 2: Check For Stale References

**Files:**
- Modify only if needed: files returned by repository search

- [ ] **Step 1: Search for legacy references**

Run: `rg -n "docs/superpowers|superpowers/specs|superpowers/plans" .`
Expected: only intentional historical references remain, or any truly stale references are identified for cleanup

- [ ] **Step 2: Verify repository state**

Run: `git status --short`
Expected: moved files and any intentional reference cleanups are visible, with no unexpected deletions outside the docs move
