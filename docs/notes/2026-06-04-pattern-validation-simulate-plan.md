# Pattern Validation Simulate Plan Loop

## Purpose

`triton-agent pattern-validation-simulate` is an **integrated** end-to-end command. It does not
run real `optimize-batch` unless `--run-optimize` is set.

Bootstrap (once): workspace plan from knowledge → **prepare agent** when the batch is empty →
dependency sync → scaffold verify.

Each cycle: **simulate agents** (per workspace) → **skill-audit agent** (updates
`pattern-validation-skills`) → repeat until `skills_alignment: aligned` or `--max-iterations`.

## Outputs

| Path | Content |
|------|---------|
| `<workspace>/simulate-plan/report.json` | Pattern ranking, hit rationale, proposed changes, skills alignment |
| `<batch>/simulate-plan-report.json` | Aggregated batch report |
| `.triton-agent/pattern-validation-simulate-state.json` | Loop iteration history |

## Typical workflow

1. Ensure `PERF_PATTERN_SYNTHESIS.md` (and usually `PERF_KNOWLEDGE_BASE.md`) exist in the repo.
2. Run `pattern-validation-simulate` once; the CLI plans, prepares when needed, verifies, then
   runs simulate → skill-audit iterations.
3. Run the printed `optimize-batch` command manually (or `--run-optimize`).

## Flags

- `--synthesis` / `--knowledge-base` — same defaults as `pattern-validation-loop`
  (`PERF_PATTERN_SYNTHESIS.md`, `PERF_KNOWLEDGE_BASE.md`). Synthesis is required on disk;
  knowledge is optional but drives workspace-plan regeneration and agent prompts when present.
- `--base` / `--skip-launch` — passed through to workspace-plan generation when knowledge exists.
- `--max-iterations` — simulate → skill-audit cycles (default: 5); use `1` for one simulate pass only.
- `--skip-prepare` — do not launch the prepare agent when the batch is empty (batch must exist).
- `--skip-verify` — skip scaffold verify before simulate iterations.
- `--run-optimize` — run real `optimize-batch` after loop completion (off by default).
