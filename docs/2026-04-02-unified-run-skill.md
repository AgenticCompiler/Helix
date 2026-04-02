# Unified `run` Skill Ownership

## Summary

- Move `run-test`, `run-bench`, `compare-result`, and `compare-perf` execution logic into a unified `skills/run-validation/` skill.
- Keep the CLI as a thin wrapper that parses arguments, validates paths, dynamically loads the skill-side Python modules, and renders results.
- Make `skills/run-validation/scripts/run-command.py` the canonical helper entrypoint for skills that need to invoke project commands from the current checkout.

## Why

- The repository already treats `skills/` as the source of truth for workflow behavior.
- Keeping run and compare execution logic inside the unified `run` skill makes that boundary explicit instead of letting local execution drift into CLI-owned modules.
- The CLI stays easier to reason about when it owns orchestration and user-facing validation, while the skill owns execution semantics.

## Stable Boundary

- `src/triton_agent/cli.py` owns argparse, path validation, prompt construction, and result printing.
- `skills/run-validation/scripts/*.py` own local execution, remote execution, metadata parsing, result archiving, and comparison behavior.
- `skills/run-validation/scripts/run-command.py` is a standalone helper CLI, not a wrapper that imports `triton_agent.cli`.
- `src/triton_agent/run_skill.py` is the only bridge layer. It resolves script paths under `skills/run-validation/scripts/` and dynamically loads them by file path.
- The dependency direction is one-way only: `triton_agent` may import `skills/run-validation/scripts`, but the skill scripts must not import `triton_agent`.
