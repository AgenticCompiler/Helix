# Pattern Validation Simulate Plan Loop

## Purpose

`triton-agent pattern-validation-simulate` is a **standalone** loop. It does not change
`pattern-validation-loop` orchestration.

Each cycle: **simulate agents** (per workspace, same inputs as optimize-batch) → **skill-audit
agent** (updates `pattern-validation-skills` from reports) → repeat until all workspaces report
`skills_alignment: aligned` or `--max-iterations` is reached.

## Outputs

| Path | Content |
|------|---------|
| `<workspace>/simulate-plan/report.json` | Pattern ranking, hit rationale, proposed changes, skills alignment |
| `<batch>/simulate-plan-report.json` | Aggregated batch report |
| `.triton-agent/pattern-validation-simulate-state.json` | Loop iteration history |

## Typical workflow

1. Prepare batch (manually or via prepare agent + verify).
2. `pattern-validation-simulate` until the loop completes.
3. Run the printed `optimize-batch` command manually (or `--run-optimize`).

## Flags

- `--max-iterations` — simulate → skill-audit cycles (default: 5); use `1` for one simulate pass only.
- `--skip-verify` — skip scaffold verify before the first simulate iteration.
- `--run-optimize` — run real `optimize-batch` after loop completion (off by default).
