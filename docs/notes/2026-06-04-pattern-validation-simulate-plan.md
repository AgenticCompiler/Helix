# Pattern Validation Simulate Plan Loop

## Purpose

`triton-agent pattern-validation-simulate` is an **integrated** end-to-end command. It does not
run real `optimize-batch` unless `--run-optimize` is set.

Bootstrap (once): workspace plan from knowledge → **prepare agent** when the batch is empty →
dependency sync → scaffold verify. CLI/prepare may read PERF markdown; simulate agents may not.

Each cycle: **simulate agents** (skills + operator only; CLI hides `validation-meta.json` during
the agent run; no PERF reports) → **skill-audit**
(updates `pattern-validation-skills` from simulate reports, including proposed code diffs) →
repeat until `skills_alignment: aligned`, `code_plan_quality: concrete`, or `--max-iterations`.

After real optimize, the **analyze agent** compares each workspace's `simulate-plan/report.json`
(`proposed_code_changes`) against `baseline/` and `opt-round-*` operator edits (no
`batch-evaluation.json` or PERF). After a successful analyze in `pattern-validation-loop`, the
CLI removes workspace `simulate-plan/` directories before the next `optimize-batch` iteration
(batch-level `simulate-plan-report.json` is kept).

## Outputs

| Path | Content |
|------|---------|
| `<workspace>/simulate-plan/report.json` | Pattern ranking, unified diff / `edits_by_pattern`, `code_plan_quality`, skills alignment |
| `<batch>/simulate-plan-report.json` | Aggregated batch report |
| `.triton-agent/pattern-validation-simulate-state.json` | Loop iteration history |

## Typical workflow

1. Ensure `PERF_PATTERN_SYNTHESIS.md` (and usually `PERF_KNOWLEDGE_BASE.md`) exist in the repo.
2. Run `pattern-validation-simulate` once; the CLI plans, prepares when needed, verifies, then
   runs simulate → skill-audit iterations.
3. Run the printed `optimize-batch` command manually (or `--run-optimize`).

## Flags

- `--synthesis` / `--knowledge-base` — used by CLI for plan/prepare only (not passed to simulate
  or skill-audit agents). Synthesis must exist on disk; knowledge drives workspace-plan when present.
- `--base` / `--skip-launch` — passed through to workspace-plan generation when knowledge exists.
- `--max-iterations` — simulate → skill-audit cycles (default: 5); use `1` for one simulate pass only.
- `--skip-prepare` — do not launch the prepare agent when the batch is empty (batch must exist).
- `--skip-verify` — skip scaffold verify before simulate iterations.
- `--run-optimize` — run real `optimize-batch` after loop completion (off by default).
