# Source Package Layout Refactor Design

## Summary

Refactor `src/triton_agent/` so modules are grouped by real ownership rather
than accumulated at the package root. The refactor should reduce root-level
sprawl, make feature boundaries easier to read, and keep CLI behavior
unchanged.

## User-Visible Behavior

- CLI commands, flags, defaults, output, and exit codes must remain unchanged.
- Skill staging behavior and backend launch behavior must remain unchanged.
- Generated file names and workspace layouts must remain unchanged.
- Remote execution semantics and SSH preflight behavior must remain unchanged.
- This refactor is structural only. Users should not need to relearn command
  behavior after it lands.

## Goals

- Shrink the number of unrelated modules living directly under
  `src/triton_agent/`.
- Continue the repository's existing direction of grouping code by feature or
  subsystem, such as `optimize/`, `convert/`, `generation/`, and `backends/`.
- Prefer small, direct package names such as `skills/`, `eval/`, and
  `remote/` over abstract layering names.
- Move modules into existing subsystem directories when ownership is already
  clear instead of introducing generic shared directories.

## Non-Goals

- No behavior redesign for optimize, convert, generation, verify, or report.
- No new dependency injection or plugin framework.
- No broad rewrite of `commands/*.py` in the same change as module moves.
- No compatibility shim layer that keeps both old and new import paths alive
  long-term.

## Design

### Root Package Policy

After the refactor, the package root should keep only truly stable shared
objects and a small set of general-purpose helpers. In the initial target
state, the root should keep modules such as:

- `cli.py`
- `models.py`
- `paths.py`
- `help_style.py`
- `batch_utils.py`
- `build_info.py`
- `hardware_info.py`
- `npu_affinity.py`
- `otel_trace.py`
- `output.py`
- `show_output_log.py`
- `prompts.py`
- `transient_failures.py`
- `verbose.py`

`models.py` stays separate because it owns cross-cutting contracts such as
`CommandKind`, `AgentRequest`, and `AgentResult`. It should not be merged into
path or filesystem helpers.

### Paths Consolidation

Merge `resources.py` into `paths.py`.

Today, `resources.py` only resolves application-root and skill-root paths, and
`paths.py` already owns generated artifact path rules. Both modules are path
centric and can live together cleanly as one filesystem-oriented helper
module.

Within `paths.py`, keep the responsibilities separated by section:

- application layout paths
- staged skill paths
- generated artifact paths

### Backend-Owned Utilities

Move the process execution helper into `backends/`:

- `process_runner.py` -> `backends/process_runner.py`

`process_runner.py` is directly backend-facing infrastructure and is only
consumed by backend runner code today. Keeping it under `backends/` makes that
ownership obvious.

Keep `show_output_log.py` and `transient_failures.py` at the package root.

- `show_output_log.py` is used by backends, convert, generation, log-check,
  report, and command-level reporting paths.
- `transient_failures.py` is used by backend retry handling and
  `optimize/recovery.py`.

Those two modules are better described as shared output and retry helpers than
backend-internal implementation details.

### Skills Package

Introduce a `skills/` package for repository-owned skill metadata, loading, and
staging.

Move:

- `skill_catalog.py` -> `skills/catalog.py`
- `skill_loader.py` -> `skills/loader.py`
- `skill_staging.py` -> `skills/selection.py`
- `skills.py` -> `skills/staging.py`

The split between `selection.py` and `staging.py` is intentional:

- `selection.py` decides which skills a workflow should stage
- `staging.py` copies the selected skills into the backend-native workspace

This keeps skill selection policy separate from filesystem mutation.

### Eval Package

Introduce an `eval/` package for run-eval-facing helpers and MCP support.

Move:

- `execution.py` -> `eval/runners.py`
- `mcp.py` -> `eval/mcp.py`
- `run_eval_mcp_server.py` -> `eval/mcp_server.py`

`execution.py` is not generic execution infrastructure; it is specifically a
bridge to the staged run-eval skill helpers. Renaming it under `eval/` makes
that ownership easier to understand.

### Remote Package

Introduce a `remote/` package for remote execution configuration and SSH
preflight behavior.

Move:

- `remote_execution_env.py` -> `remote/env.py`
- `remote_ssh_preflight.py` -> `remote/ssh_preflight.py`

Do not name this package `ssh_utils`. Only one of these modules is SSH
specific; the other is remote execution environment resolution.

### Optimize-Owned Subagent Staging

Fold the root-level `subagents.py` support into `optimize/subagents.py`.

Today the shared subagent staging helpers are only consumed by optimize-side
code and tests. Until another subsystem proves it needs the same abstraction,
the code should live with optimize ownership instead of remaining at the root.

## Target Layout

```text
src/triton_agent/
  cli.py
  models.py
  paths.py
  help_style.py
  batch_utils.py
  build_info.py
  hardware_info.py
  npu_affinity.py
  otel_trace.py
  output.py
  show_output_log.py
  prompts.py
  transient_failures.py
  verbose.py

  backends/
    ...
    process_runner.py

  skills/
    __init__.py
    catalog.py
    loader.py
    selection.py
    staging.py

  eval/
    __init__.py
    runners.py
    mcp.py
    mcp_server.py

  remote/
    __init__.py
    env.py
    ssh_preflight.py

  optimize/
    ...
    subagents.py
```

All other existing packages remain at their current locations unless a later
design explicitly moves them.

## Migration Plan

### Phase 1

Move the clearly backend-owned helper first:

- `process_runner.py`

This is the lowest-risk structural slice because the dependency surface is
small and ownership is already clear.

### Phase 2

Fold root `subagents.py` into `optimize/subagents.py`.

This should happen before wider package moves so optimize-owned staging code is
no longer presented as a global subsystem.

### Phase 3

Create the `skills/` package and update imports across:

- `convert/`
- `generation/`
- `optimize/`
- `report/`
- `log_check/`
- `distill/`
- `scripts/`
- tests

This is the largest import migration and should stay purely structural. Based
on the current import surface, moving the skill-related modules is expected to
touch roughly three dozen repository import sites across `src/`, `tests/`, and
`scripts/`.

### Phase 4

Create the `eval/` and `remote/` packages and update imports across:

- `commands/`
- `verify/`
- `backends/`
- `status/`
- `scripts/`
- tests

At the same time, merge `resources.py` into `paths.py`.

## Risks And Mitigations

### Import churn can create noisy diffs

Mitigation: move modules in small phases and keep each phase structural only.

### Package rename conflicts can break tests or patch targets

Mitigation: update repository imports in the same change as each move and run
focused tests for the touched subsystem before continuing.

### External build and packaging scripts can be missed

Mitigation: explicitly include `scripts/` in the import update checklist for
the `skills/`, `eval/`, `remote/`, and `paths.py` moves. In particular,
`scripts/build-claude-optimize-plugin.py` must be updated together with the
module moves it imports.

### Structural and behavioral refactors can get mixed together

Mitigation: treat file relocation and behavior cleanup as separate phases with
separate verification.

## Validation

For each completed phase, run focused tests for the touched imports first, then
run the repository-standard checks before declaring the phase complete:

- `uv run --group dev ruff check`
- `uv run pyright`
- `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`

## Future Work

After module placement stabilizes, evaluate targeted command thinning in:

- `commands/convert.py`
- `commands/execution.py`
- `commands/optimize.py`

That follow-up should be a separate refactor so behavior changes do not get
mixed into package moves.
