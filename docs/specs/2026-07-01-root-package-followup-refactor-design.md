# Root Package Follow-Up Refactor Design

## Summary

Continue shrinking `src/helix/` after the first package-layout cleanup so
the remaining root-level modules reflect real cross-cutting ownership instead
of acting as a mixed landing zone for terminal helpers, batch helpers, trace
helpers, report-only helpers, and packaging glue.

This design is a roadmap for the next cleanup pass. It complements the earlier
package-layout refactor and the dedicated trace-package design.

## User-Visible Behavior

- CLI commands, flags, defaults, output, and exit codes must remain unchanged.
- Trace file names, trace summary file names, and trace environment variable
  names must remain unchanged.
- Report generation behavior and hardware detection behavior must remain
  unchanged.
- Verbose output wording and show-output log paths must remain unchanged.
- Batch workspace discovery and NPU affinity semantics must remain unchanged.

This is a structural refactor only.

## Goals

- Further reduce the number of unrelated modules living directly under
  `src/helix/`.
- Group terminal-related helpers, batch-related helpers, and trace-related
  helpers under small subsystem packages.
- Move obviously feature-local helpers into their owning feature packages.
- Keep only truly cross-cutting contracts and stable root-level entry helpers
  at the package root.

## Non-Goals

- No behavior redesign for trace capture, trace analysis, verbose output,
  report generation, or batch affinity.
- No changes to the public CLI command surface.
- No merge of unrelated concerns into generic `utils` or `common` packages.
- No broad command-handler thinning in the same change as the package moves.

## Current Root Modules

After the first refactor pass, the root still contains:

- `cli.py`
- `models.py`
- `paths.py`
- `prompts.py`
- `build_info.py`
- `_setuptools_hooks.py`
- `output.py`
- `show_output_log.py`
- `verbose.py`
- `help_style.py`
- `batch_utils.py`
- `npu_affinity.py`
- `hardware_info.py`
- `otel_trace.py`
- `transient_failures.py`

These do not all have the same ownership. Some are stable shared contracts,
some are subsystem helpers, and some are packaging-specific entry glue.

## Design

### Keep A Small Stable Root

The root package should keep only modules that are genuinely cross-cutting or
are intentionally top-level entry glue:

- `cli.py`
- `models.py`
- `paths.py`
- `prompts.py`
- `build_info.py`
- `_setuptools_hooks.py`
- `transient_failures.py`

Rationale:

- `cli.py` is the executable entrypoint.
- `models.py` owns shared contracts such as `CommandKind`, `AgentRequest`, and
  `AgentResult`.
- `paths.py` is the shared filesystem and generated-artifact path layer.
- `prompts.py` is used across multiple workflows and backends.
- `build_info.py` and `_setuptools_hooks.py` are packaging/build metadata glue.
  They are coupled to `pyproject.toml` command-class configuration and should
  stay stable until there is a dedicated packaging-focused redesign.
- `transient_failures.py` is tiny, but it is shared across backend retry logic
  and optimize recovery; inlining it into `process_runner` would create a less
  honest dependency boundary.

### Introduce `terminal/`

Create a small `terminal/` package for shared CLI/terminal-facing helpers.

Move:

- `output.py` -> `terminal/render.py`
- `show_output_log.py` -> `terminal/logs.py`
- `verbose.py` -> `terminal/verbose.py`
- `help_style.py` -> `terminal/help.py`

These modules are all small, but they are also heavily shared. They are not
feature-local and should not stay scattered at the root.

The point of this move is not to merge them into one file. They should remain
separate modules under one ownership boundary:

- `render.py` for `AgentResult` rendering
- `logs.py` for show-output log path and stream helpers
- `verbose.py` for structured verbose terminal output
- `help.py` for help-text styling

`show_output_log.py` is used by both backend modules and non-backend workflow
modules. This grouping intentionally prioritizes conceptual ownership
("terminal-facing display and log presentation") over consumer distance. A
`backends -> terminal` dependency is acceptable here because the backend code
is participating in CLI-facing output flow rather than exposing backend-private
behavior.

### Introduce `batch/`

Create a `batch/` package for batch workspace discovery and batch NPU affinity
support.

Move:

- `batch_utils.py` -> `batch/discovery.py`
- `npu_affinity.py` -> `batch/affinity.py`

These two modules are already used together by convert, generation, optimize,
and eval-related flows. They form one coherent subsystem and should move
together.

That eval-related set explicitly includes `eval/mcp_server.py`, which currently
imports the NPU affinity parsing helpers even though it is not a batch module
itself.

### Introduce `trace/`

Create a dedicated `trace/` package and remove the split between the
root-level `otel_trace.py` module and the `trace_analyze/` package.

Move:

- `otel_trace.py` -> `trace/core.py` and `trace/summary.py`
- `trace_analyze/analyzer.py` -> `trace/analyze.py`

This follow-up depends on the detailed design in:

- `docs/specs/2026-07-01-trace-package-refactor-design.md`

That design remains the source of truth for the internal split between
`core.py`, `summary.py`, and `analyze.py`.

### Move Report-Only Hardware Helpers

Move:

- `hardware_info.py` -> `report/hardware.py`

The hardware inspection helper is effectively report-owned today. Keeping it at
the root makes the package look more shared than it really is.

### Packaging Glue Stays Put For Now

Keep these modules at the root in this pass:

- `_setuptools_hooks.py`
- `build_info.py`

`_setuptools_hooks.py` is referenced directly from `pyproject.toml`, and
`build_info.py` is its runtime counterpart used by the CLI. They may later move
under a dedicated packaging or build metadata area, but that would require a
separate design that accounts for setuptools import-path expectations and test
patch targets.

## Target Layout

```text
src/helix/
  cli.py
  models.py
  paths.py
  prompts.py
  build_info.py
  _setuptools_hooks.py
  transient_failures.py

  terminal/
    __init__.py
    render.py
    logs.py
    verbose.py
    help.py

  batch/
    __init__.py
    discovery.py
    affinity.py

  trace/
    __init__.py
    core.py
    summary.py
    analyze.py

  report/
    ...
    hardware.py
```

All other existing packages remain at their current locations unless a later
design explicitly moves them.

## Migration Plan

### Phase 1

Introduce `terminal/` and migrate terminal-related imports.

This is the most self-contained follow-up because it mainly affects shared
rendering and verbose-output call sites without changing feature ownership.
Based on the current import surface, this phase is expected to touch roughly
22 repository import sites.

### Phase 2

Introduce `batch/` and migrate batch discovery and NPU affinity imports across:

- `convert/`
- `generation/`
- `optimize/`
- `eval/`
- tests

In particular, this phase should explicitly update `eval/mcp_server.py` in
addition to the batch workflow callers. Based on the current import surface,
this phase is expected to touch roughly 9 repository import sites.

### Phase 3

Move `hardware_info.py` into `report/hardware.py` and update the report
callers. This is expected to touch only the report-owned call site plus its
tests.

### Phase 4

Apply the separate trace-package design:

- create `trace/core.py`
- create `trace/summary.py`
- create `trace/analyze.py`
- remove `otel_trace.py`
- remove `trace_analyze/`

Based on the current import surface, this phase is expected to touch roughly
20 repository import sites: 19 imports of `otel_trace.py` plus the
`trace-analyze` command-layer import of the current `trace_analyze` package.

## Risks And Mitigations

### Risk: moving tiny modules can create noisy import churn

Mitigation: move by subsystem, not by file count. Each phase should update one
ownership area at a time.

### Risk: terminal helpers look small and tempt over-merging

Mitigation: group them under `terminal/`, but keep separate modules for
rendering, help styling, verbose output, and log-path handling.

### Risk: packaging-related root modules are easy to move prematurely

Mitigation: explicitly defer `_setuptools_hooks.py` and `build_info.py` until a
separate packaging-focused design exists.

### Risk: trace work can sprawl during migration

Mitigation: follow the dedicated trace-package design and stop at
`core.py`, `summary.py`, and `analyze.py` for the initial split.

## Validation

For each phase, run focused tests for the affected subsystem first, then run
the repository-standard checks:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

## Future Work

This follow-up design intentionally does not cover command-handler thinning or
broader prompt-layer restructuring. Once the package boundaries settle, those
topics can be revisited separately with less structural noise.

It also does not commit to a `transient_failures.py` rename during the package
moves. After stabilization, that module can be reassessed as a low-priority
naming cleanup, for example to `agent_failures.py` or `retry_signals.py`, but
it should not drive a new dependency move into `backends/process_runner.py`.
