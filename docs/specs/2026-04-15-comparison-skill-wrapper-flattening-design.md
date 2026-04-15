# Comparison Skill Wrapper Flattening Design

## Summary

- Remove the redundant `compare_result.py` and `compare_perf.py` skill wrappers under `skills/operator-eval/scripts/`.
- Remove the redundant `src/triton_agent/comparison.py` package bridge and let the comparison command module load skill implementations directly.
- Preserve the current CLI surface, exit codes, and `triton_agent -> skills` dependency direction.

## Goals

- Eliminate one-hop forwarding modules that add no behavior.
- Keep comparison logic in the executable command path instead of preserving a fake package API layer.
- Keep skill scripts free of `triton_agent` imports.

## Non-Goals

- Do not refactor `src/triton_agent/test_runner.py` or `src/triton_agent/bench_runner.py` in this change.
- Do not change local or remote comparison semantics.

## Design

- Update `src/triton_agent/commands/comparison.py` so it:
  - loads the `test_runner` skill module for result comparisons
  - loads the `bench_runner` skill module for perf comparisons
  - keeps argument validation and error presentation in the same file
- Delete `skills/operator-eval/scripts/compare_result.py`.
- Delete `skills/operator-eval/scripts/compare_perf.py`.
- Delete `src/triton_agent/comparison.py`.
- Adjust focused tests to target `triton_agent.commands.comparison` directly instead of a package bridge.

## Verification

- Run `uv run python -m unittest tests.test_comparison_commands tests.test_run_skill_loader -v`
- Run `uv run --group dev ruff check`
- Run `uv run pyright`
