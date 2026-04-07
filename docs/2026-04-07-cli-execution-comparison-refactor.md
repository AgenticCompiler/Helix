# CLI Execution And Comparison Refactor

## Summary

- Continue the thin-CLI refactor after the optimize extraction by moving the remaining execution and comparison command branches out of `src/triton_agent/cli.py`.
- This phase covers:
  - `run-test`
  - `run-bench`
  - `compare-result`
  - `compare-perf`
- Keep generation commands in `cli.py` for now because they still share the generic agent-request construction path.

## Motivation

- After the optimize extraction, `cli.py` still contains the full command flow for local and remote execution plus comparison operations.
- Those branches are user-facing command workflows, not parser concerns, and they keep the entrypoint larger than necessary.
- Moving them now gives the project a clearer command-layer shape without prematurely refactoring the remaining generation path.

## User-Visible Behavior

- CLI behavior should remain unchanged.
- Existing argument validation, metadata fallback, remote execution behavior, output rendering, and exit codes must stay the same.

## Scope

- Add command handlers for execution and comparison flows.
- Move execution and comparison command routing out of `cli.py`.
- Keep current low-level run-skill wrappers unless extraction naturally suggests a better home.

## Target Structure

Recommended additions for this phase:

```text
src/triton_agent/
  commands/
    optimize.py
    execution.py
    comparison.py
  execution.py
  comparison.py
```

Intended responsibilities:

- `commands/execution.py`
  - handle `run-test` and `run-bench`
  - keep CLI-facing validation and result printing
- `commands/comparison.py`
  - handle `compare-result` and `compare-perf`
  - keep CLI-facing validation and return-code behavior
- `execution.py`
  - expose thin runtime helpers for local and remote test and benchmark execution
  - centralize metadata-resolution helpers for generated harnesses
- `comparison.py`
  - expose thin runtime helpers for compare-result and compare-perf

## Boundary Rules

- `cli.py` should only:
  - build the parser
  - normalize aliases
  - dispatch to command handlers
  - keep the generic generation flow for now
- command handlers may use `argparse.Namespace`
- runtime helper modules should not depend on `argparse`
- output formatting should keep using the shared `render_result` helper

## Testing

- Preserve existing `tests/test_cli.py` coverage as the top-level regression suite.
- Add focused tests for the extracted execution and comparison helper modules if they improve direct verification of behavior and reduce future patching through `cli.py`.

## Verification

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`
