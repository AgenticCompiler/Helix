# Distill Trajectory Analysis Design

## Status

Deferred. This document preserves the optimization-trajectory analysis approach
explored on the `feat/skills_update` branch before the mainline `distill`
refactor. It is not implemented in the current CLI.

## Problem

Distillation updates pattern cards from baseline-vs-optimized evidence, but it
does not capture **how** an optimizer reached the final answer across many rounds.
Trajectory analysis targets methodology: detours, wasted rounds, ordering
dependencies, and reusable skip rules.

## Proposed Scope

Two layers:

1. **Per-operator trajectory analysis** — read `opt-note.md` first, optionally
   deep-read `triton-agent-logs/` for regression/U-turn/abandoned rounds.
2. **Cross-operator merge** — combine multiple `trajectory_analysis.md` reports
   into a playbook and a concise reusable path skill.

This is complementary to distill, not a replacement. Distill extracts *what*
changed in code; trajectory analysis extracts *how the search should have
proceeded*.

## Per-Operator Workflow (Draft)

### Phase 0 — Timeline from opt-note

Build a round table (theme, speedup, promoted/not) and classify rounds:

- Breakthrough (>1.05x, promoted)
- Incremental (1.02–1.05x)
- Noise (~1.0x)
- Regression (<0.98x or not promoted after loss)
- Abandoned (correctness/compile dead-end)

### Phase 1 — Surface detours (cheap)

From opt-note alone, flag:

- Parameter-sweep chains (3+ tiny tuning rounds)
- U-turns (promoted change later reverted)
- Noise rounds
- Correctness-debug loops

### Phase 2 — Log-assisted deep read (selective)

Only for regression, U-turn, abandoned, or suspiciously thin themes:

- Locate round window in batch logs via `grep opt-round-N`
- Read ≤2000 lines per round, ≤8 rounds total
- Detect rushed-fallback pattern (primary hypothesis fails, weak substitute submitted)

### Phase 3 — Optimal path

Construct minimal effective path: remove noise, collapse sweeps, reorder by
dependency.

### Phase 4 — Principles

Extract sequence rules, skip rules, prerequisites, anti-patterns.

### Output

`trajectory_analysis.md` under the operator directory with timeline, detours,
abandoned approaches, optimal path, and principles.

## Cross-Operator Merge (Draft)

Input: directory of operator workspaces each containing `trajectory_analysis.md`.

1. Index scan — operator name, round count, speedup, first 3 themes only.
2. Group by pattern signature (e.g. reduction-stat ops share dead-code → autotune → grid-cap).
3. Align timelines within groups; identify consensus steps vs divergent steps.
4. Deep-read principles only when resolving conflicts.
5. Output:
   - `playbook.md` — full evidence and tables
   - `playbook.skill.md` — concise optimizer-facing path skill

## Integration Options (Future)

| Option | Pros | Cons |
|--------|------|------|
| Separate CLI subcommand `triton-agent distill-trajectory` | Clear boundary | More CLI surface |
| Optional `--trajectory` phase after distill | Single entrypoint | Long runs, mixed concerns |
| Standalone scripts only | Fast to iterate | No staging/skill contract |

Recommended first step when implementing: standalone scripts under `scripts/`
using the same agent staging as `distill`, then promote to CLI once stable.

## Explicit Non-Goals (for v1)

- No automatic modification of pattern cards from trajectory output
- No diff-summary handoff into simulate/analyze (see separate deferred design)
- No dependency on grouped editable skill layout — use flat
  `<skills-dir>/<language>-npu-optimize-knowledge/` like mainline distill

## Related Artifacts from Branch Prototype

The pre-refactor branch contained prompt builders and scripts:

- `build_trajectory_analysis_prompt`
- `build_trajectory_merge_prompt`
- `scripts/run_trajectory_analysis.py`
- `scripts/run_trajectory_merge.py`
- `scripts/run_trajectory_pipeline.py`

When re-implementing, port prompts into
`skills/common/ascend-npu-distill-patterns/references/` or a dedicated
`ascend-npu-optimize-trajectory` skill rather than large Python prompt strings.
