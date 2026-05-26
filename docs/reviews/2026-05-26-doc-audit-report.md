# Document Audit Report

> Audit date: 2026-05-26
> Scope: ALL docs/ files (113 plans + 171 specs + 66 notes + README + AGENTS.md)
> Auditor: Sisyphus

## Summary

Reviewed all documentation against current codebase state. Found and fixed **3 categories of issues**: inconsistencies, missing info, and superseded documents lacking notices.

| Category | Fixed | Description |
|---|---|---|
| Inconsistencies | 2 | README missing commands; AGENTS.md outdated backend list |
| Missing content | 1 | README missing log-check section |
| Superseded notices | 3 | Added notices to docs describing removed features |

---

## Detailed Findings

### 1. README.md — Missing commands

**Problem:** `log-check` and `log-check-batch` are fully implemented CLI commands (`CommandKind.LOG_CHECK`, `CommandKind.LOG_CHECK_BATCH`) but were missing from the README Command Map and had no dedicated section.

**Fix:** Added both commands to the Command Map and Quick Start examples, and added a new "Run Log Strategy Validation" section with usage examples and common options.

### 2. AGENTS.md — Outdated backend list

**Problem:** Line 7 stated "The supported backends are `codex`, `opencode`, `pi`, and `claude`", omitting `openhands` and `traecli` which are both supported in the current CLI.

**Fix:** Updated to "The supported backends are `codex`, `opencode`, `pi`, `claude`, `openhands`, and `traecli`."

### 3. Superseded documents (notices added)

| Document | Superseded by | Reason |
|---|---|---|
| `docs/specs/2026-04-09-optimize-analysis-driven-design.md` | `docs/specs/2026-04-22-optimize-layered-analysis-default-design.md` | Proposed `--require-analysis` flag; later spec removed it and made layered analysis default |
| `docs/plans/2026-04-09-optimize-analysis-driven.md` | Same as above | Implementation plan for the now-removed `--require-analysis` feature |
| `docs/plans/2026-04-02-optimize-continue-mode.md` | `docs/specs/2026-04-09-optimize-resume-mode-design.md` | Described `--continue` flag which was replaced by `--resume {auto,continue,fresh}` |

### 4. Documents already properly marked as superseded (pre-existing)

These documents already had superseded/historical notices — verified as correct:

- `docs/notes/2026-03-31-triton-agent-cli.md` — superseded notice present
- `docs/notes/2026-04-07-cli-optimize-refactor-layering.md` — "Superseded note" present
- `docs/specs/2026-04-10-optimize-supervisor-round-gate-design.md` — superseded notice present
- `docs/specs/2026-04-13-optimize-supervise-mode-design.md` — superseded note present
- `docs/specs/2026-04-13-optimize-supervised-log-archive-design.md` — superseded note present

### 5. Notes on docs maintained as historical artifacts

The following categories of documents are intentionally kept as historical snapshots and do not need updating:

- **Early notes (Mar 31 – Apr 7):** Document design decisions that were implemented. They describe the *why* behind current code structure. Adding a superseded notice to every old note would add noise rather than value.
- **Implementation plans (all dates):** These are task-by-task execution guides that authors followed during implementation. They are historical artifacts of *how* something was built. Plans for completed work do not need to match the current codebase state.
- **Early specs later refined by subsequent specs:** Many spec topics (baseline, optimize-status, verify, backends) evolved through multiple iterative specs. The latest spec in each chain represents the current design. Earlier specs document the design evolution.

### 6. Spec evolution chains (for reference)

| Topic | Evolution chain |
|---|---|
| `optimize-status` | Apr 9 (5 specs) → Apr 13 (single-workspace) → Apr 20 (best-round-warning, markdown-notes, name-sorting) → Apr 21 (verified-speedups) |
| `baseline` | Apr 13 (contract, prep) → Apr 14 (state-contract) → Apr 21 (field-map) → Apr 22 (baseline-skill) |
| `verify` | Apr 20 (single-workspace) → Apr 21 (batch, verify-batch) |
| `backends` | Apr 9 (package refactor) → Apr 13 (cli-dedup) → Apr 14 (openhands) → Apr 17 (traecli) |
| `optimize-supervisor` | Apr 10 (round-gate) → Apr 13 (supervise-mode, log-archive) → Apr 14 (alias, check-loop) → Apr 16 (round-perf-analysis) |

---

## Remaining Suggestions (not fixed)

1. **Most early specs and plans** reference old module paths (e.g., `src/triton_agent/supervisor.py`, `src/triton_agent/runtime.py`) that no longer exist. The Apr 16 review already identified these. These docs are intentionally preserved as historical design records; updating every path reference would lose the historical context.
2. **Some very old plans** (Mar 31 – Apr 3) reference `superpowers:subagent-driven-development` (now `subagent-driven-development` without the `superpowers:` prefix). These are execution instructions for agentic workers; updating them is unnecessary since the plans are completed.
