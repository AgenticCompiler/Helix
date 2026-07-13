# CLI And Backend Dedup Refactor Design

## Summary

- Reduce redundant wrapper and branching code in the CLI entrypoint and backend runners.
- Keep user-visible CLI flags, defaults, output formatting, and backend launch semantics unchanged.
- Limit this refactor to low-risk structural cleanup that improves maintainability without widening scope into unrelated execution or optimize redesign.

## Goals

- Treat `src/helix/cli.py` as a thin executable entrypoint instead of a secondary public API surface.
- Remove redundant passthrough helpers from `cli.py` when the same behavior already lives in dedicated modules.
- Replace the CLI's repeated command-option branching with a table-driven command definition that is easier to extend safely.
- Move shared backend runner flow into the backend base class so each backend focuses on command construction and true backend-specific behavior.

## Non-Goals

- Do not change command names, aliases, option defaults, help semantics, or exit-code behavior.
- Do not change prompt construction, skill staging semantics, optimize resume behavior, or result rendering behavior.
- Do not unify execution, comparison, generation, and optimize runtime orchestration into a new shared abstraction in this refactor.
- Do not introduce backend capability registries, plugins, or dynamic backend discovery.

## Current Redundancy

### CLI passthrough wrappers

- `src/helix/cli.py` currently re-exports helpers such as local and remote run wrappers, compare wrappers, output rendering, and backend factory creation.
- These helpers are thin one-line forwards to the real modules and make the entrypoint look like a library API.
- Repository tests already mostly target the real command modules, so the remaining wrapper imports can be removed with focused test updates.

### CLI parser and dispatch branching

- `build_parser()` encodes command capabilities through repeated `if command_kind in {...}` checks.
- `main()` mirrors that with a second long chain mapping `CommandKind` to command handlers.
- This makes command evolution error-prone because the definition of a command is spread across many branches.

### Backend runner skeleton duplication

- `CodexRunner`, `OpenCodeRunner`, `PiRunner`, and `ClaudeRunner` all duplicate the same `run()`, `resume()`, verbose logging, and process mode selection structure.
- The only meaningful per-backend differences are command construction, Codex session extraction and output filtering, and a small amount of session-related flag logic.

## Proposed Design

## CLI Entry Module

- Keep `cli.py` responsible only for:
  - normalizing command aliases
  - building the parser
  - dispatching the parsed command to the right handler
- Remove passthrough helper functions that duplicate behavior from:
  - `helix.execution`
  - `helix.comparison`
  - `helix.generation.outputs`
  - `helix.output`
  - `helix.backends.factory`
- Update tests to import those helpers from their real modules instead of through `cli.py`.

## Table-Driven Command Definitions

- Introduce a small internal command-definition structure inside `cli.py`.
- Each command definition should describe:
  - handler
  - primary input arguments
  - whether it supports output paths
  - whether it supports verbosity
  - whether it supports remote execution
  - whether it supports agent selection
  - whether it supports interact or show-output
  - test and bench mode defaults when applicable
  - optimize-only and batch-only options when applicable
- `build_parser()` should iterate the definitions and attach arguments from shared helpers instead of open-coding large membership checks.
- `main()` should dispatch through the same definition table to keep parser setup and handler lookup derived from one source of truth.

## Backend Base Class

- Extend `src/helix/backends/base.py` with a reusable runner skeleton that provides:
  - shared `run()` implementation
  - shared `resume()` implementation using `build_optimize_resume_prompt()`
  - shared verbose launch logging
  - shared process mode selection
  - overridable hooks for `build_command()`, session-id extraction, and output filtering
- Keep `CodexRunner` responsible for:
  - command construction
  - `_UnifiedDiffFilter`
  - UUID session extraction
- Keep `OpenCodeRunner`, `PiRunner`, and `ClaudeRunner` responsible only for backend-specific command construction and any backend-specific hook overrides.

## Testing Strategy

- Update tests that currently import helper functions from `helix.cli` to import from the dedicated modules instead.
- Add or adjust CLI tests so parser behavior still covers:
  - command aliases
  - command defaults
  - option presence and absence
  - command dispatch through `main()`
- Keep existing backend tests as regression coverage for backend-specific command lines.
- Add focused coverage for any new shared backend-base behavior if current tests do not already exercise it.

## Risks And Mitigations

- Risk: table-driven parser generation may accidentally change which options exist on a command.
  - Mitigation: rely on the existing parser tests in `tests/test_cli.py` and keep command definitions explicit.
- Risk: removing CLI wrappers may break tests or undocumented internal imports.
  - Mitigation: update repository imports in the same change and confirm no remaining internal references.
- Risk: backend base-class refactor could subtly change process-runner parameters.
  - Mitigation: preserve backend-specific hooks and keep Codex-specific behavior isolated.

## Verification

- Run focused regression tests while each stage is in progress.
- Run at least:
  - `uv run python -m unittest tests.test_cli tests.test_execution_commands tests.test_comparison_commands tests.test_generation_commands tests.test_backends_factory tests.test_codex_runner tests.test_opencode_runner tests.test_pi_runner tests.test_claude_runner -v`
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
