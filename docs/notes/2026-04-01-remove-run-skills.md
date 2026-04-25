# Remove `test-run` And `bench-run` Skills

## Summary

- Delete the dedicated `skills/test-run/` and `skills/bench-run/` directories.
- Keep the user-facing `run-test` and `run-bench` CLI subcommands.
- Treat these run commands as direct local CLI behavior rather than skill-backed agent workflows.

## Why

- `run-test` and `run-bench` no longer launch code agents.
- Result archiving, perf parsing, and comparison logic already live in local Python modules.
- The remaining run-skill files only duplicate or contradict the real behavior.

## Required cleanup

- Remove the two skill directories from `skills/`.
- Stop advertising `test-run` and `bench-run` as active skill names in code and docs.
- Update generation and optimize guidance to reference the `run-test` and `run-bench` CLI subcommands directly.
- Keep existing CLI subcommands, artifact names, and validation flow unchanged.
