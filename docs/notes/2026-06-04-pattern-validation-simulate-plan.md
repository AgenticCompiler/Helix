# Pattern Validation Simulate Plan

## Purpose

`triton-agent pattern-validation-simulate` is a **standalone** fast path. It does not change
`pattern-validation-loop` orchestration.

The command runs one dry-run agent per active validation workspace with the **same** inputs
as `optimize-batch` (staged skills, `skills-source-dir`, operator workspace, test/bench modes
in metadata), but the agent only writes `simulate-plan/report.json` — no compile, no rounds.

## Outputs

| Path | Content |
|------|---------|
| `<workspace>/simulate-plan/report.json` | Per-workspace pattern ranking, hit rationale, proposed changes, skills alignment |
| `<batch>/simulate-plan-report.json` | Aggregated batch report + suggested manual `optimize-batch` command |

## Typical workflow

1. Prepare batch (manually or via prepare agent + verify) as today.
2. `pattern-validation-simulate` → review `simulate-plan-report.json` and fix pattern cards under `pattern-validation-skills/`.
3. When satisfied, run the printed `optimize-batch` command manually (or pass `--run-optimize`).

## Flags

- `--skip-verify` — skip scaffold verify before simulate agents.
- `--run-optimize` — chain into real `optimize-batch` after all simulate plans succeed (off by default).
