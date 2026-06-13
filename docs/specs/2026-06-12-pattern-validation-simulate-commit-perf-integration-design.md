# Pattern Validation Simulate + Commit Perf Integration

## Summary

`pattern-validation-simulate` becomes the single end-to-end entrypoint from Git commit
extraction through pattern-card embedding in `pattern-validation-skills`. The CLI runs
commit performance analysis when PERF reports are missing (or when `--force` is set),
then continues with the existing simulate → skill-audit loop.

`analyze-commit-perf` remains available as a standalone extraction-only command.

## User-visible behavior

1. **Extract (default):** When `PERF_PATTERN_SYNTHESIS.md` or `PERF_KNOWLEDGE_BASE.md`
   is missing, or when `--force` is passed, launch the commit-perf agent using the same
   `--base`, `--pull-request`, `--target-chip`, and `--include-ir` flags as the simulate
   command.
2. **Embed:** Run workspace plan → prepare (when needed) → verify → simulate → skill-audit
   iterations until skills are aligned or `--max-iterations` is reached.

Use `--skip-extract` when PERF reports already exist and should not be regenerated.

## Removed / simplified

- `run_simulate_plan_batch` (unused one-shot helper).
- Duplicate `resolve_git_worktree` in `commit_perf_analysis/launcher.py` (reuse shared helper).

## Non-goals

- Do not remove the standalone `analyze-commit-perf` subcommand.
- Do not remove `pattern-validation-plan` (still useful for manual plan inspection).
