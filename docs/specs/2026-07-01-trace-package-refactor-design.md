# Trace Package Refactor Design

## Summary

Refactor the current trace-related modules into a dedicated `trace/` package so
trace runtime helpers, trace summary generation, and trace analysis live under
one subsystem instead of being split between `otel_trace.py` and
`trace_analyze/`.

The refactor should keep CLI behavior and trace file formats unchanged while
making module ownership clearer.

## User-Visible Behavior

- Existing command names, especially `trace-analyze`, must remain unchanged.
- Tool-trace environment variable names must remain unchanged.
- Trace log locations and summary file names must remain unchanged.
- Generated trace summary JSON structure must remain unchanged.
- Existing backend trace capture behavior must remain unchanged.

This is a structural refactor only.

## Goals

- Replace the split between root-level `otel_trace.py` and the
  `trace_analyze/` package with one coherent `trace/` subsystem.
- Separate trace runtime helpers from trace summary generation logic.
- Reduce the number of large cross-cutting modules at the package root.
- Keep the initial split simple enough that imports stay easy to follow.

## Non-Goals

- No trace schema redesign.
- No changes to event field names, summary JSON keys, or duration heuristics.
- No rename of the `trace-analyze` CLI subcommand.
- No broader logging or output refactor in the same change.

## Current Problem

Today trace code is split across two ownership locations:

- `src/helix/otel_trace.py`
- `src/helix/trace_analyze/`

That split hides the fact that these files are part of the same subsystem.

`otel_trace.py` also mixes several distinct responsibilities:

- trace constants and run-id generation
- trace path and environment construction
- event appending and event payload helpers
- tool-trace summary generation

The result is a root module that has grown broad, plus a separate analysis
package that already depends on it.

## Design

### New Package Layout

Introduce a dedicated trace package:

```text
src/helix/trace/
  __init__.py
  core.py
  summary.py
  analyze.py
```

Remove:

- `src/helix/otel_trace.py`
- `src/helix/trace_analyze/`

### `trace/core.py`

`core.py` should own trace runtime primitives and shared path helpers.

Move these items into `trace/core.py`:

- `TRACE_PATH_ENV`
- `TRACE_RUN_ID_ENV`
- `TRACE_WORKSPACE_ROOT_ENV`
- `utc_timestamp()`
- `new_trace_run_id()`
- `append_trace_event()`
- `trace_path_from_request()`
- `tool_trace_path()`
- `trace_summary_path()`
- `build_trace_env()`
- `build_tool_trace_env()`
- `build_code_agent_event()`
- `summarize_agent_command()`

This module is the stable import target for backends and workflow orchestration
code that needs to create or append traces.

### `trace/summary.py`

`summary.py` should own tool-trace summary generation.

Move these items into `trace/summary.py`:

- `write_tool_trace_summary()`
- `build_tool_trace_summary()`
- `_read_trace_events()`
- `_tool_trace_capabilities()`
- `_tool_trace_capability_label()`
- `_tool_trace_evidence_gaps()`
- `_detect_trace_source()`
- `_time_ms_by_category()`
- `_build_duration_quality()`
- `_build_top_slow_operations()`

`summary.py` may import `trace_summary_path()` and other helpers from
`trace.core`.

This keeps summary construction logic separate from runtime event writing.

### `trace/analyze.py`

Move the current analysis implementation from:

- `trace_analyze/analyzer.py`

to:

- `trace/analyze.py`

This module should continue to own:

- `analyze_trace()`
- `build_summary()`
- the existing analysis-only helpers

`trace/analyze.py` should import `trace_summary_path()` from `trace.core`.

### `trace/__init__.py`

Keep `__init__.py` intentionally small.

It should re-export only the minimal, high-level public helpers that make
sense to import from the package boundary, such as:

- `analyze_trace`

Do not re-export the entire trace surface by default. Callers should usually
import from `trace.core` or `trace.summary` explicitly so dependencies remain
obvious.

## Import Migration

### Move to `trace.core`

Update these consumers to import from `helix.trace.core`:

- backend trace capture code
- convert/generation/log-check/report trace setup
- trace path helpers used by commands
- tests that currently import from `helix.otel_trace`

### Move to `trace.summary`

Update workflow modules that currently call `write_tool_trace_summary()` to
import from `helix.trace.summary`.

### Move to `trace.analyze`

Update the trace analyze command layer to import analysis helpers from the new
`trace/` package.

## Migration Plan

### Phase 1

Create `trace/core.py` and move trace primitives and path helpers from
`otel_trace.py` without changing behavior.

### Phase 2

Create `trace/summary.py` and move summary-generation logic out of
`otel_trace.py`.

### Phase 3

Move `trace_analyze/analyzer.py` into `trace/analyze.py` and replace
`trace_analyze/__init__.py` with `trace/__init__.py`.

### Phase 4

Update imports across:

- `backends/`
- `commands/`
- `convert/`
- `generation/`
- `log_check/`
- `optimize/`
- `report/`
- `trace_analyze/` replacement call sites
- tests

### Phase 5

Delete the old `otel_trace.py` file and the old `trace_analyze/` package once
all imports are updated.

## Risks And Mitigations

### Risk: call sites may import the wrong new module

Mitigation: keep the split coarse. Runtime helpers live in `core.py`, summary
construction lives in `summary.py`, and offline analysis lives in `analyze.py`.

### Risk: the split may accidentally change shared helper behavior

Mitigation: treat this as a move-only refactor first. Do not redesign helper
logic during the package migration.

### Risk: tests may still patch old import paths

Mitigation: update patch targets and focused tests in the same change as the
module moves.

## Validation

Run at least focused tests for:

- trace helper behavior
- trace analyze command behavior
- backend trace event writing
- convert/generation/log-check/report trace summary writing

Then run the repository standard checks:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

## Future Work

This refactor does not decide whether trace helper re-exports should become a
stable package-level API. If later trace code grows further, a follow-up can
decide whether additional internal modules such as `paths.py` or `events.py`
are worth introducing. The initial refactor should stop at `core.py`,
`summary.py`, and `analyze.py`.
