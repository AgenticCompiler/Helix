# CLI Optimize Refactor Layering

## Summary

- Refactor the CLI toward a thin entrypoint architecture instead of continuing to grow `src/triton_agent/cli.py`.
- Use a mixed layering model:
  - a thin CLI layer for parser construction, top-level dispatch, and user-facing exit behavior
  - command-group modules for command-specific orchestration
  - focused domain modules for reusable workflow behavior
- Implement the first migration only for the `optimize`, `optimize-batch`, and `optimize-status` command group.

## Motivation

- `src/triton_agent/cli.py` currently combines parser definition, path resolution, optimize request construction, optimize execution lifecycle, batch orchestration, status scanning, and output rendering.
- The file has become the default landing zone for new behavior, which makes future changes harder to reason about and easier to regress.
- The optimize command group already contains multiple independent responsibilities that deserve explicit module boundaries.
- This refactor should reduce the size and cognitive load of `cli.py` without forcing a risky repository-wide rewrite in one step.

## User-Visible Behavior

- Command behavior should remain the same for:
  - `optimize`
  - `optimize-batch`
  - `optimize-status`
- Existing CLI flags, defaults, continuation rules, batch workspace rules, and output format should remain stable.
- The refactor is structural, not semantic, for this change.

## Goals

- Make `src/triton_agent/cli.py` a thin entrypoint.
- Create explicit boundaries around optimize workflow behavior.
- Separate optimize execution, batch orchestration, status inspection, validation, and rendering concerns.
- Preserve current tests and behavior while improving the shape for future refactors.
- Establish a repeatable migration pattern that later command groups can follow.

## Non-Goals

- Do not redesign CLI semantics for generation, run, or compare commands in this change.
- Do not change optimize prompt content, supervisor policy, or remote semantics in this refactor.
- Do not introduce JSON output or new optimize subcommand features.
- Do not force the entire repository into a pure layer-only architecture.

## Design Principles

- Prefer clear module boundaries over minimizing file count.
- Keep command-group flow discoverable: one engineer should still be able to follow an optimize command without jumping across unrelated domains.
- Keep `argparse.Namespace` usage near the CLI edge.
- Keep optimize domain code independent from parser details.
- Keep rendering separate from computation so output changes do not require touching core workflow code.
- Reuse existing public data models and helpers where they already fit.

## Target Architecture

Recommended long-term structure:

```text
src/triton_agent/
  cli.py
  cli_parser.py
  cli_dispatch.py
  commands/
    __init__.py
    optimize.py
    generation.py
    execution.py
    comparison.py
  optimize/
    __init__.py
    models.py
    runtime.py
    batch.py
    status.py
    render.py
    validation.py
```

The first iteration only needs to introduce the optimize-related parts plus any minimal dispatch support required to keep `cli.py` thin enough.

## Layer Responsibilities

### CLI Layer

Files:

- `src/triton_agent/cli.py`
- optionally `src/triton_agent/cli_dispatch.py`

Responsibilities:

- build the parser
- normalize command aliases
- parse arguments
- dispatch to the right command handler
- convert expected failures into concise CLI exit behavior

Rules:

- keep `argparse.Namespace` at this edge or in immediate command handlers
- avoid embedding optimize workflow rules directly here

### Command Layer

Files:

- `src/triton_agent/commands/optimize.py`

Responsibilities:

- translate CLI argument objects into optimize-domain calls
- preserve command-specific exit conventions
- keep one obvious entrypoint per optimize-related subcommand

Rules:

- know CLI semantics
- do not own optimize workflow internals

### Optimize Domain Layer

Files:

- `src/triton_agent/optimize/runtime.py`
- `src/triton_agent/optimize/batch.py`
- `src/triton_agent/optimize/status.py`
- `src/triton_agent/optimize/render.py`
- `src/triton_agent/optimize/validation.py`
- `src/triton_agent/optimize/models.py`

Responsibilities:

- implement optimize workflow behavior independent from parser details
- expose focused functions and data types
- keep optimize-specific models out of the generic CLI module

Rules:

- do not depend on `argparse.Namespace`
- keep data flow explicit through typed parameters and dataclasses
- keep rendering and scanning logic separated

## Optimize Module Boundaries

### `src/triton_agent/optimize/runtime.py`

Owns single-workspace optimize execution:

- build optimize `AgentRequest`
- apply optimize default modes
- resolve continue-mode metadata reuse
- prepare staged skills
- prepare temporary optimize `AGENTS.md`
- launch the runner through `OptimizeSupervisor`
- clean up guidance and copied skills

This module is the home for logic currently represented by:

- `_build_optimize_request()`
- `_run_optimize_request()`
- optimize-specific setup and teardown surrounding `SkillLinkManager` and `OptimizeGuidanceManager`

### `src/triton_agent/optimize/batch.py`

Owns multi-workspace optimize orchestration:

- scan immediate child workspace directories
- identify the operator candidate for each workspace
- build one optimize request per workspace
- run workspaces concurrently
- summarize success or failure for each workspace

This module is the home for logic currently represented by:

- `_run_optimize_batch()`
- `_resolve_batch_optimize_operator_file()`
- `_is_batch_optimize_operator_candidate()`
- `_summarize_batch_optimize_failure()`
- `_PrefixedTextStream`

### `src/triton_agent/optimize/status.py`

Owns read-only optimize status inspection:

- inspect one workspace for optimize artifacts
- locate baseline and round perf files
- parse logged best round from `opt-note.md`
- compute comparable round scores
- derive `ok`, `warning`, and `no-session` states

This module is the home for logic currently represented by:

- `_inspect_optimize_status_workspace()`
- `_select_baseline_perf_file()`
- `_find_round_perf_file()`
- `_parse_logged_best_round()`
- `_round_number()`
- `_mean_value()`

### `src/triton_agent/optimize/render.py`

Owns optimize-related CLI output rendering:

- batch optimize summary rendering
- optimize status float and percent formatting
- optimize status report rendering

This module is the home for logic currently represented by:

- `_render_batch_optimize_results()`
- `_format_optimize_status_float()`
- `_format_optimize_status_percent()`
- `_render_optimize_status_results()`

### `src/triton_agent/optimize/validation.py`

Owns optimize argument validation:

- `--min-rounds` lower bound
- `--max-concurrency` lower bound
- `--continue` incompatibility with explicit mode overrides

This module is the home for logic currently represented by:

- `_validate_optimize_arguments()`

If continue-mode validation grows further, it should extend this module rather than returning to `cli.py`.

### `src/triton_agent/optimize/models.py`

Owns optimize-only data structures:

- `BatchOptimizeWorkspace`
- `BatchOptimizeResult`
- `OptimizeStatusRound`
- `OptimizeStatusWorkspace`

These models should move out of generic CLI scope because they are not shared by all commands.

## Dependency Rules

Desired dependency direction:

- `cli.py` -> `cli_dispatch.py` or command handlers
- `commands/optimize.py` -> `optimize/*`
- `optimize/batch.py` -> `optimize/runtime.py`, `optimize/models.py`, `optimize/render.py`
- `optimize/status.py` -> `optimize/models.py`
- `optimize/render.py` -> `optimize/models.py`
- `optimize/runtime.py` -> existing shared modules such as `models.py`, `prompts.py`, `supervisor.py`, `skills.py`, `optimize_guidance.py`

Avoid:

- `optimize/*` importing `argparse`
- `render.py` importing parser helpers
- `status.py` depending on `batch.py`
- command modules reaching back into `cli.py`

## Incremental Migration Plan

### Phase 1: Extract optimize domain models and helpers

- Move optimize-only dataclasses from `cli.py` into `src/triton_agent/optimize/models.py`.
- Move optimize validation into `src/triton_agent/optimize/validation.py`.
- Move optimize rendering helpers into `src/triton_agent/optimize/render.py`.

### Phase 2: Extract runtime, batch, and status behavior

- Move single optimize execution lifecycle into `src/triton_agent/optimize/runtime.py`.
- Move batch orchestration into `src/triton_agent/optimize/batch.py`.
- Move optimize status inspection into `src/triton_agent/optimize/status.py`.

### Phase 3: Add optimize command handler

- Add `src/triton_agent/commands/optimize.py` as the single command-layer entrypoint for:
  - `optimize`
  - `optimize-batch`
  - `optimize-status`
- Keep command-layer functions small and argument-focused.

### Phase 4: Thin the CLI entrypoint

- Update `src/triton_agent/cli.py` to dispatch optimize subcommands through `commands/optimize.py`.
- Leave non-optimize commands in place for now.
- Preserve alias normalization, parser construction, shared path resolution, and existing helper behavior until later migrations.

## Why This Is Better Than A Single `optimize_commands.py`

- A single extraction file would reduce `cli.py` line count but would likely keep mixed responsibilities together.
- `optimize`, `optimize-batch`, and `optimize-status` are related but not identical workflows.
- Explicit internal modules make future work easier:
  - improving status computation should not touch batch concurrency code
  - changing cleanup behavior should not touch status rendering
  - testing one optimize concern should not require importing all optimize concerns

## Testing Strategy

- Keep current CLI-level tests passing so command behavior remains stable.
- Add or update focused tests for extracted optimize modules where it improves coverage and readability.
- Prefer validating extracted domain logic directly instead of only through `main()`.

Minimum verification after implementation:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m unittest discover -s tests -v`

## Documentation Updates

- Keep this design document as the architectural source of truth for the refactor.
- Update `README.md` only if the refactor changes documented developer-facing structure or testing guidance.
- Update `AGENTS.md` only if the architectural rule for thin CLI layering becomes a durable project rule.

## Risks And Mitigations

### Risk: Hidden coupling remains in `cli.py`

Mitigation:

- move optimize-specific models, validation, rendering, and execution together rather than only moving one top-level function

### Risk: New optimize module becomes a second large catch-all file

Mitigation:

- keep runtime, batch, status, render, and validation as separate modules from the start

### Risk: Tests become brittle during extraction

Mitigation:

- preserve external behavior first
- migrate tests incrementally
- keep function signatures straightforward during the first extraction

## Future Follow-Up

After optimize migration succeeds, reuse the same pattern for other command groups:

- generation commands into a generation command/domain group
- run commands into an execution command/domain group
- compare commands into a comparison command/domain group

That later work can continue reducing `cli.py` until it becomes a stable, thin entrypoint instead of a growth hotspot.
