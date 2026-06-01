---
name: triton-npu-pattern-validation-loop
description: Close the loop from PERF_PATTERN_SYNTHESIS to validated optimize-batch runs. Agent reads synthesis, updates persistent repo-local skills, builds validation workspaces, runs optimize-batch, audits pattern application, and iterates until success.
---

# Pattern Validation Loop

## Goal

Turn commit-analysis output into **verified** optimization knowledge:

1. **Read** `PERF_PATTERN_SYNTHESIS.md` and update a **persistent** skills workdir under the repo.
2. **Build** optimize-batch workspaces with pre-optimization operators and tests.
3. **Run** `optimize-batch` with `--skills-source-dir` so optimize copies from that workdir.
4. **Audit** pattern IDs in round artifacts; archive passes under `_completed/`.
5. **Iterate** edits in the workdir and re-run until success or `max_iterations`.

## Prerequisites

- `PERF_PATTERN_SYNTHESIS.md` in the repo.
- Git history for pre-optimization snapshots.
- `triton-agent` CLI for subprocess `optimize-batch`.

## Paths (resolve at runtime)

- `REPO` = Git repository root (where you run commands)
- `SKILLS` = persistent loop skills workdir (default `REPO/pattern-validation-skills/`)
- `KNOWLEDGE` = `SKILLS/triton-npu-optimize-knowledge/` — **edit pattern cards here only**
- `SKILL` = staged orchestration skill (read contracts; do not edit knowledge here)
- `STATE` = `REPO/.triton-agent/pattern-validation-loop-state.json`
- `BATCH` = batch root (default `REPO/pattern-validation-batch/`)

On first `pattern-validation-loop` run, the CLI seeds `KNOWLEDGE` from the triton-agent install
bundle if missing. The directory **stays on disk** for the whole loop and after completion.

Read before acting:

- [references/skill-update-contract.md](references/skill-update-contract.md)
- [references/workspace-scaffold-contract.md](references/workspace-scaffold-contract.md)
- [references/iteration-contract.md](references/iteration-contract.md)

## Phase A — Initialize loop state

```bash
python3 "$SKILL/scripts/init_loop_state.py" \
  --repo "$REPO" \
  --synthesis PERF_PATTERN_SYNTHESIS.md \
  --batch-dir pattern-validation-batch \
  --skills-dir pattern-validation-skills \
  --base origin/main \
  --min-rounds 10 \
  --max-iterations 5
```

## Phase B — Read synthesis and update skills (each iteration)

1. Read `PERF_PATTERN_SYNTHESIS.md` (and `PERF_KNOWLEDGE_BASE.md` if needed).
2. Edit pattern cards under **`$KNOWLEDGE/references/patterns/`** only.
3. Regenerate index:

```bash
python3 "$KNOWLEDGE/scripts/build_pattern_index.py" \
  --patterns-dir "$KNOWLEDGE/references/patterns" \
  --output "$KNOWLEDGE/references/pattern_index.md"
```

4. Record changes in `STATE`.

**Do not** edit `$REPO/.codex/skills/` or the triton-agent install tree for knowledge updates.

## Phase C — Plan and scaffold workspaces

Follow [workspace-scaffold-contract.md](references/workspace-scaffold-contract.md). When a
repo `source_path` contains multiple kernels, use **Step 2b** (manual split) — group by
launch entrypoint and call chain, not by raw kernel count; do not copy the whole file into
one workspace or use blind auto-split scripts.

## Phase D — Run optimize batch

From `REPO`:

```bash
triton-agent optimize-batch \
  -i "$BATCH" \
  --resume fresh \
  --reset-optimize \
  --min-rounds 10 \
  --max-concurrency 1 \
  --show-output \
  --skills-source-dir "$SKILLS" \
  --agent <backend>
```

**Required:** `--skills-source-dir "$SKILLS"` copies matching skill subdirectories from the
persistent workdir into each workspace before optimize (overwriting install-bundle copies).

Later iterations:

```bash
python3 "$SKILL/scripts/reset_workspace_rounds.py" --batch-root "$BATCH"

triton-agent optimize-batch -i "$BATCH" \
  --resume continue \
  --min-rounds 10 \
  --skills-source-dir "$SKILLS" \
  --show-output \
  --agent <backend>
```

## Phase E — Audit

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" \
  --archive-passed \
  --json > "$BATCH/audit-report.json"
```

## Phase F — Iterate or complete

On audit failure: edit `$KNOWLEDGE` → regenerate index → reset active rounds → Phase D with the
same `--skills-source-dir "$SKILLS"`.

On success (`active_remaining` empty): write `$BATCH/VALIDATION_SUMMARY.md`.

## Non-Negotiable Rules

- All knowledge edits live under `$SKILLS`; the directory is never deleted by the loop.
- Every optimize-batch run must pass `--skills-source-dir "$SKILLS"`.
- Do not hand-edit `pattern_index.md`.
- Do not delete `baseline/` when iterating; use `reset_workspace_rounds.py` on active workspaces.

## Related Skills

- `triton-npu-analyze-commit-perf` — produces synthesis input
- `triton-npu-optimize` / `optimize-batch` — optimization execution
