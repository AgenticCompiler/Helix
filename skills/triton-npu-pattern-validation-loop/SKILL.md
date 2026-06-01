---
name: triton-npu-pattern-validation-loop
description: Close the loop from PERF_PATTERN_SYNTHESIS to validated optimize-batch runs. Agent reads synthesis, updates staged pattern skills, builds validation workspaces, runs optimize-batch, audits pattern application, and iterates until success.
---

# Pattern Validation Loop

## Goal

Turn commit-analysis output into **verified** optimization knowledge:

1. **Read** `PERF_PATTERN_SYNTHESIS.md` (any structure) and decide skill updates.
2. **Build** optimize-batch workspaces with pre-optimization operators and tests.
3. **Run** `optimize-batch` for a fixed minimum number of rounds.
4. **Audit** whether expected pattern IDs appear in round artifacts.
5. **Iterate** skills and re-run until success or `max_iterations`.

This skill defines **process and contracts**. The orchestrating agent performs synthesis
interpretation, file layout, and Git snapshot selection — not fixed-format parsers.

## Prerequisites

- `PERF_PATTERN_SYNTHESIS.md` exists (typically from `analyze-commit-perf`).
- Git repo on the analyzed branch; know `base_revision..HEAD` for snapshots.
- `triton-agent` CLI available for subprocess `optimize-batch`.
- NPU bench optional; audit can pass on pattern citation alone.

## Paths (resolve at runtime)

- `REPO` = Git repository root
- `SKILL` = staged copy of this skill
- `KNOWLEDGE` = staged `triton-npu-optimize-knowledge`
- `STATE` = `REPO/.triton-agent/pattern-validation-loop-state.json`
- `BATCH` = batch root (default `REPO/pattern-validation-batch`)

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
  --base origin/main \
  --min-rounds 10 \
  --max-iterations 5
```

## Phase B — Read synthesis and update skills (each iteration start)

1. Read **`PERF_PATTERN_SYNTHESIS.md` end-to-end**. Use `PERF_KNOWLEDGE_BASE.md` when you
   need file paths, commits, or diff detail. Do **not** assume fixed section names or IDs.
2. Compare against staged `KNOWLEDGE/references/pattern_index.md`.
3. Edit pattern cards under `KNOWLEDGE/references/patterns/` per
   [skill-update-contract.md](references/skill-update-contract.md).
4. Regenerate index:

```bash
python3 "$KNOWLEDGE/scripts/build_pattern_index.py" \
  --patterns-dir "$KNOWLEDGE/references/patterns" \
  --output "$KNOWLEDGE/references/pattern_index.md"
```

5. Record changes:

```bash
python3 "$SKILL/scripts/record_iteration.py" \
  --state "$STATE" --phase skill-update \
  --note "summary of pattern edits"
```

**Do not** hand-edit `pattern_index.md`.

## Phase C — Plan and scaffold workspaces (agent-driven)

Follow [workspace-scaffold-contract.md](references/workspace-scaffold-contract.md).

You must:

1. Decide which operators deserve a validation workspace and why.
2. Extract **pre-optimization** snapshots with Git (`base_revision..HEAD`).
3. Find and copy tests (and optional benches / import dependencies).
4. Write each `validation-meta.json` with `expected_patterns` aligned to staged pattern IDs.
5. Optionally write `$BATCH/manifest.json` as your own index.

Optional helpers (use only if **you** already wrote manifest JSON):

- `scripts/scaffold_batch.py --manifest ... --output "$BATCH"`
- `scripts/generate_manifest.py` — heuristic only; synthesis may not match its parser

Record scaffold:

```bash
python3 "$SKILL/scripts/record_iteration.py" \
  --state "$STATE" --phase scaffold \
  --note "workspaces created: ..."
```

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
  --agent <backend>
```

Later iterations (baseline already exists):

```bash
python3 "$SKILL/scripts/reset_workspace_rounds.py" --batch-root "$BATCH"

triton-agent optimize-batch -i "$BATCH/<workspace>" \
  --resume continue --min-rounds 10 --show-output --agent <backend>
```

Prefer `--max-concurrency 1` until stable.

## Phase E — Audit pattern application

After optimize-batch, audit **active** workspaces only (`_completed/` is skipped).

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" \
  --archive-passed \
  --json > "$BATCH/audit-report.json"
```

When a workspace passes, `--archive-passed` moves it to `$BATCH/_completed/<name>/`.
The next `optimize-batch -i "$BATCH"` run schedules only remaining active workspaces.

Also read failing active workspaces: `opt-round-*/attempts.md`, `summary.md`, and compare to
each `validation-meta.json`.

```bash
python3 "$SKILL/scripts/record_iteration.py" \
  --state "$STATE" --phase audit \
  --audit-report "$BATCH/audit-report.json"
```

## Phase F — Decide next action

| Audit result | Action |
|--------------|--------|
| All active workspaces passed and archived | Write `$BATCH/VALIDATION_SUMMARY.md`; mark complete |
| Some active `missing_patterns` | Phase B targeted fixes → reset rounds on **active only** → Phase D → re-audit |
| `round_count == 0` on an active workspace | Fix harness / snapshot / tests before skill iteration |
| Iteration ≥ `max_iterations` with active failures | Stop with failure summary |

Loop is **complete** when `audit-report.json` → `active_remaining` is empty and every
target workspace sits under `_completed/`.

```bash
python3 "$SKILL/scripts/record_iteration.py" \
  --state "$STATE" --phase complete
```

## Non-Negotiable Rules

- **You** interpret synthesis; do not require `G1-I1` tables or run blind manifest generation.
- Edit **staged** pattern cards; regenerate `pattern_index.md` after each edit batch.
- Do not treat repo-local-only lessons as required skill outcomes.
- Do not delete `baseline/` when iterating skills; use `reset_workspace_rounds.py` on **active** workspaces only.
- Do not re-run optimize on workspaces already under `_completed/`.
- Do not skip audit because optimize-batch exited zero.
- Run real shell commands; do not fabricate audit JSON or round artifacts.

## Helper scripts (machine checks only)

| Script | Role |
|--------|------|
| `init_loop_state.py` | Create/update loop state |
| `record_iteration.py` | Append history events |
| `reset_workspace_rounds.py` | Clear rounds; keep baseline and workspace files |
| `audit_batch.py` | Check `expected_patterns`; `--archive-passed` moves passes to `_completed/` |
| `batch_layout.py` | Shared active/completed workspace layout helpers (imported by scripts) |
| `scaffold_batch.py` | Optional: materialize manifest you authored |
| `generate_manifest.py` | Optional: rough manifest when synthesis matches strict table format |

## Related Skills

- `triton-npu-analyze-commit-perf` — produces synthesis input
- `triton-npu-optimize` / `optimize-batch` — optimization execution
- `triton-npu-optimize-knowledge` — pattern library under test
