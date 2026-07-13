# Status Command Rename Design

## Summary

- Rename the user-facing command `optimize-status` to `status`.
- Do not keep compatibility aliases for the old command name or old snake_case spelling.
- Move status inspection and status rendering code out of `src/helix/optimize/` into a dedicated `src/helix/status/` package.
- Give status its own command entrypoint module, `src/helix/commands/status.py`, instead of keeping the handler under optimize commands.

## Goals

- Make the CLI name shorter and consistent with the repository's recent `verify` command split.
- Make module boundaries match real ownership so read-only status reporting no longer lives under optimization runtime code.
- Preserve the current status behavior, input semantics, output formats, and verify-state integration while renaming the feature.

## Non-Goals

- Do not change how status selects the numeric best round.
- Do not change status output fields, markdown formatting, or verify-state interpretation.
- Do not change optimize workspace artifacts or verification artifact formats.
- Do not keep deprecated aliases or compatibility shims for the old command or old module paths.

## User-Facing Behavior

- `uv run helix status -i operators_root`
- `uv run helix status -i .`
- `uv run helix status -i operators_root --format markdown`

The old command:

- `optimize-status`

must stop working instead of acting as a hidden alias.

The command remains read-only:

- It summarizes optimization progress from existing optimize artifacts.
- It still accepts either a single workspace directory or a root directory of workspaces.
- It still supports `--format markdown`.
- It still surfaces the latest successful verification signal from `opt-verify/verify-*/verify-state.json`.

## CLI Structure

- Replace `CommandKind.OPTIMIZE_STATUS` with `CommandKind.STATUS`.
- Register the new command name `status` in the parser.
- Remove alias normalization entries for `optimize_status`.
- Add canonical snake_case support only for `status` if the repository still keeps snake_case aliases for canonical command names.
- Move the command out of the `Optimization` help group and into a dedicated `Status` help group.
- Keep the help summary explicit about optimization status so the shorter command name does not become ambiguous.

## Module Boundaries

Create a dedicated package:

- `src/helix/status/__init__.py`
- `src/helix/status/core.py`
- `src/helix/status/render.py`

Move status-specific helpers into that package:

- `inspect_optimize_status_workspace()` and related workspace scanning helpers move from `optimize/status.py` to `status/core.py`
- status rendering helpers move from `optimize/render.py` to `status/render.py`

Keep optimize-owned rendering separate:

- `render_batch_optimize_results()` stays under `src/helix/optimize/` because it belongs to batch optimize execution, not status reporting

Create a dedicated command entrypoint module:

- `src/helix/commands/status.py`

After the move:

- `src/helix/commands/optimize.py` should no longer own the status handler
- `src/helix/verification/core.py` should depend on `helix.status.core` for best-round selection instead of `helix.optimize.status`

## Behavior Preservation

Even after the rename and move:

- best-round ranking stays exactly the same
- baseline perf selection stays exactly the same
- text output and markdown output stay exactly the same
- single-workspace detection stays exactly the same
- malformed or incomplete verify-state payloads must continue to degrade gracefully instead of failing the command

## Tests And Docs

Update tests to assert:

- `status` parses and dispatches to `CommandKind.STATUS`
- `optimize-status` is absent from help output and no longer parses
- status helpers import from `helix.status.core`
- status rendering imports from `helix.status.render`
- verification imports the renamed status module path

Update user-facing docs and design docs so they use `status` terminology consistently when referring to the command.
