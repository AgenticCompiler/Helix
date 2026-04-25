# CLI Generation Refactor

## Summary

- Complete the thin-CLI refactor by moving the remaining generation command flow out of `src/triton_agent/cli.py`.
- This phase covers:
  - `gen-test`
  - `gen-bench`
- Extract generation-specific request construction, output-path resolution, overwrite protection, and agent launch orchestration into dedicated modules.

## Motivation

- After the optimize, execution, and comparison extractions, the largest remaining logic in `cli.py` is the generation flow.
- That flow still mixes:
  - input-path validation
  - output-path derivation
  - overwrite behavior
  - prompt construction
  - `AgentRequest` creation
  - skill staging
  - agent launch
- Moving it out finishes the command-layer split and leaves `cli.py` close to a real entrypoint module.

## User-Visible Behavior

- CLI behavior should remain unchanged for `gen-test` and `gen-bench`.
- Existing defaults for output file names, test and bench modes, overwrite protection, remote context in prompts, verbose skill staging logs, and exit codes must remain stable.

## Target Structure

Recommended additions for this phase:

```text
src/triton_agent/
  commands/
    generation.py
  generation.py
```

Intended responsibilities:

- `commands/generation.py`
  - handle `gen-test` and `gen-bench`
  - keep CLI-facing path validation and error-to-exit behavior
- `generation.py`
  - resolve generation output paths
  - enforce overwrite behavior
  - build generation `AgentRequest`
  - run the generation request with staged skills and existing runner infrastructure

## Boundary Rules

- `cli.py` should only:
  - build the parser
  - normalize aliases
  - dispatch all command kinds to handlers
  - expose only lightweight compatibility wrappers when tests or other modules still need them
- generation runtime should not depend on `argparse`
- shared runner and output helpers should keep their current roles

## Testing

- Preserve current `tests/test_cli.py` behavior coverage.
- Add focused tests for generation handlers and generation runtime helpers so future changes do not need to patch through `cli.py`.

## Verification

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`
