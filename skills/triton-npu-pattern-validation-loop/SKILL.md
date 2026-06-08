---
name: triton-npu-pattern-validation-loop
description: Close the loop from PERF_PATTERN_SYNTHESIS to validated optimize-batch runs. CLI orchestrates optimize; agents prepare workspaces and analyze evidence.
---

# Pattern Validation Loop

## Goal

Turn commit-analysis output into **verified** optimization knowledge:

1. **Prepare agent** reads `PERF_PATTERN_SYNTHESIS.md` and updates persistent repo-local skills.
2. **Prepare agent** builds optimize-batch workspaces with pre-optimization operators and tests.
3. **CLI** runs `pattern-validation-verify`, then **`optimize-batch`** with `--skills-source-dir`.
4. **CLI** collects round evidence into `audit-report.json`.
5. **Analyze agent** reviews evidence, updates skills if needed, archives passed workspaces.
6. Repeat optimize → evidence → analyze until success or `max_iterations`.

## Prerequisites

- `PERF_PATTERN_SYNTHESIS.md` in the repo (pattern promotion targets).
- `PERF_KNOWLEDGE_BASE.md` in the repo (kernel-scoped lessons; drives `workspace-plan.json`).
- Git history for pre-optimization snapshots.
- `triton-agent` CLI for the loop, `pattern-validation-plan`, and `pattern-validation-verify`.

You only need to place the markdown reports; the CLI generates `workspace-plan.json` before the
prepare agent runs.

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

## CLI entrypoint

```bash
uv run triton-agent pattern-validation-loop \
  -i "$REPO" \
  --show-output \
  --agent <backend>
```

The CLI runs **prepare agent → verify → optimize-batch → evidence collection → analyze agent**
and repeats optimize/analyze until complete or `--max-iterations`.

## Phase A — Initialize loop state (prepare agent or CLI)

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

## Phase B — Prepare agent: synthesis and skills

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

## Phase C — Prepare agent: scaffold workspaces

When `PERF_KNOWLEDGE_BASE.md` exists, read
[knowledge-base-scaffold-contract.md](references/knowledge-base-scaffold-contract.md) first:

The CLI already ran `triton-agent pattern-validation-plan` when `PERF_KNOWLEDGE_BASE.md` exists.
Read `$BATCH/workspace-plan.json` (regenerate with `pattern-validation-plan` only if missing).

Then follow [workspace-scaffold-contract.md](references/workspace-scaffold-contract.md):

- **One launch function → one workspace** named after the **primary kernel**
  (for example `chunk_bwd_kernel_dv_local/chunk_bwd_kernel_dv_local.py`).
- Split knowledge-base lessons **per kernel**, not per source file.
- If one launch calls multiple kernels (branches), merge them into one operator file.

After scaffolding, sync dependencies into each workspace (literal directory name `deps`, never `{deps}`):

- **Default:** inject repo `sys.path` at the top of the operator; keep original `from fla...` / `from src.kernels.fla...` imports.
- **Verify:** import smoke (`python -c "import <operator_stem>"`) per workspace.
- **Fallback:** copy `fla.*` closure into `deps/fla/` only when smoke fails or you pass `--copy-deps`.

```bash
python3 "$SKILL/scripts/sync_workspace_dependencies.py" \
  --batch-root "$BATCH" --repo "$REPO"
```

`pattern-validation-loop` runs this automatically before verify. Then:

```bash
triton-agent pattern-validation-verify -i "$BATCH"
```

Fix every reported issue before the prepare agent finishes. The CLI runs the same command
again before optimize as a hard gate.

## Simulate optimize plan loop (integrated prepare + simulate)

Use `pattern-validation-simulate` for an **end-to-end** dry-run loop: workspace plan,
prepare agent (when the batch is empty), verify, simulate agents, skill-audit on
`pattern-validation-skills`, then repeat until aligned. It does **not** run real
`optimize-batch` unless you pass `--run-optimize`.

```bash
uv run triton-agent pattern-validation-simulate -i "$REPO" \
  --synthesis PERF_PATTERN_SYNTHESIS.md \
  --knowledge-base PERF_KNOWLEDGE_BASE.md \
  --batch-dir pattern-validation-batch \
  --skills-dir pattern-validation-skills \
  --max-iterations 5 \
  --show-output \
  --agent opencode
```

Bootstrap (once per command, before simulate iterations):

1. CLI generates `workspace-plan.json` when `PERF_KNOWLEDGE_BASE.md` exists.
2. If no active workspaces, CLI launches the **same prepare agent** as
   `pattern-validation-loop` (scaffold + verify). Use `--skip-prepare` only when the batch
   is already scaffolded.
3. CLI syncs deps and runs `pattern-validation-verify` (`--skip-verify` to skip).

Each simulate iteration:

1. Sync deps.
2. **Simulate agents** (one per workspace): same staged optimize skills as a real worker plus the
   operator `.py` and `test_*.py.txt` only. Ground truth (`expected_patterns`, etc.) lives in
   `pattern-validation-batch/batch-evaluation.json`, not inside workspace directories. They
   must **not** read PERF markdown. Write `simulate-plan/report.json` with `ranked_patterns`,
   `proposed_code_changes` (required `unified_diff` and per-hit `edits_by_pattern`),
   `code_plan_quality`, and `skills_alignment`.
3. If all workspaces pass **CLI structural validation** and the **skill-audit agent** confirms
   `skills_alignment: aligned` with `code_plan_quality: concrete`, the loop completes. The CLI
   never finishes on simulate self-assessment alone.
4. Otherwise a **skill-audit agent** reads `$BATCH/simulate-plan-report.json`, reviews proposed
   code changes (not only pattern hits), edits
   `$SKILLS/triton-npu-optimize-knowledge/references/patterns/`, regenerates
   `pattern_index.md`, and may mark `.triton-agent/pattern-validation-simulate-state.json`
   complete.
5. Repeat until aligned or `--max-iterations`.

State file: `.triton-agent/pattern-validation-simulate-state.json`.

After completion the CLI prints a suggested **manual** `optimize-batch` command. Pass
`--run-optimize` only when you want the CLI to run real optimize after the simulate loop.
Use `--max-iterations 1` for one simulate → skill-audit cycle (skill-audit always runs).

## Phase D — CLI optimize batch

The **CLI** runs `optimize-batch` (not the prepare/analyze agents). It injects a prompt that
each workspace may include `test_*.py.txt` **reference** files (dtype and shapes only; not
runnable pytest). The optimize agent should use them when authoring real `test_*.py`.

Typical flags:

- `--skills-source-dir "$SKILLS"`
- `--min-rounds 10 --concurrency 1 --show-output`
- `TRITON_AGENT_STALL_TIMEOUT_SECONDS=0` in the environment

Optional passthrough from loop start: `--target-chip`, `--test-mode`, `--bench-mode`.

## Phase E — Evidence collection (CLI)

The CLI writes `$BATCH/audit-report.json` using `audit_batch.py`. The report aggregates
`opt-round-*/attempts.md`, `summary.md`, and related paths. `heuristic_suggested_pass` is a
hint only — not an automatic pass/fail gate.

```bash
python3 "$SKILL/scripts/audit_batch.py" \
  --batch-root "$BATCH" \
  --output "$BATCH/audit-report.json"
```

## Phase F — Analyze agent

1. Read `$BATCH/audit-report.json` and open round artifacts when needed.
2. Judge whether synthesis-backed mechanisms were applied (not only pattern ID substring hits).
3. Update `$KNOWLEDGE` if another iteration is needed; regenerate `pattern_index.md`.
4. When confident a workspace passed, archive with `audit_batch.py --archive-passed`.
5. When all targets pass, write `$BATCH/VALIDATION_SUMMARY.md` and `record_iteration.py --phase complete`.

After a successful analyze agent, the CLI deletes each active workspace's `simulate-plan/`
directory (dry-run artifacts are already summarized in `audit-report.json` and batch-level
`simulate-plan-report.json` when present). Then, between iterations, the CLI runs
`reset_workspace_rounds.py` and another `optimize-batch`.

## Non-Negotiable Rules

- All knowledge edits live under `$SKILLS`; the directory is never deleted by the loop.
- Prepare and analyze agents **must not** run `triton-agent optimize-batch`.
- Only **one** operator `.py` at each workspace root; copy helper modules under `deps/` (see workspace-scaffold-contract).
- Run `triton-agent pattern-validation-verify -i "$BATCH"` after scaffolding (prepare agent + CLI gate).
- If `optimize-batch` has per-workspace failures, the CLI still collects evidence and runs the analyze agent.
- Do not copy entire multi-kernel source files into one workspace when synthesis validates separate launch entrypoints.
- Do not hand-edit `pattern_index.md`.
- Do not delete `baseline/` when iterating; use `reset_workspace_rounds.py` on active workspaces.

## Related Skills

- `triton-npu-analyze-commit-perf` — produces synthesis input
- `triton-npu-optimize` / `optimize-batch` — optimization execution (CLI-driven in this loop)
