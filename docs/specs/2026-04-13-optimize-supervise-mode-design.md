# Optimize Supervise Mode Design

## Summary

> **Superseded note:** The current supervised optimize flow still uses worker and supervisor roles, but it no longer stages `.helix/roles/*` files and it no longer launches a dedicated `optimize-supervisor` skill. Supervisor behavior is prompt-driven.

- Add an explicit `--supervise {on,off}` option to `optimize` and `optimize-batch`.
- Default `--supervise` to `off` so ordinary optimize runs continue using the existing single-agent flow.
- Run the worker-plus-supervisor round gate only when `--supervise on` is selected.
- Keep `--interact` scoped to the worker role when supervision is enabled.
- Preserve the current optimize skill workflow and round-gate implementation, but make it opt-in instead of unconditional.

## Goals

- Restore a fast path where `optimize` behaves like one unconstrained code-agent run unless the user explicitly asks for supervision.
- Keep the round-gate workflow available for users who care about artifact completeness, workflow compliance, and next-round handoff quality.
- Make mode selection explicit so users can reason about optimize behavior from the command line alone.
- Preserve backward-compatible batch semantics by applying the same supervise mode independently per workspace.

## Non-Goals

- Do not delete the current round-gate implementation.
- Do not change optimize skill content based on supervise mode.
- Do not introduce an `auto` mode in the first version.
- Do not make `--supervise on` the default in this change.

## Problem

- The current optimize runtime always executes the worker-plus-supervisor round-gate loop.
- This is safer, but heavier than the original single-agent optimize behavior.
- Some users want the CLI to stay out of the way and let the code agent run the optimize workflow directly.
- The current implementation also makes it hard to reason about whether an optimize run will be supervised unless the user already knows the internal runtime structure.

## Approaches Considered

### Recommended: Explicit `--supervise on|off`

- Make supervision an explicit optimize mode choice.
- Default to `off` so existing "just let the code agent run" expectations are restored.
- Keep the supervised round-gate flow available behind `--supervise on`.

Why this is the best fit:

- It matches the user's stated preference that supervision should be opt-in.
- It makes optimize behavior obvious from the command line instead of from internal runtime knowledge.
- It gives the project a clean migration point if the default should change later.

### Alternative: Bare `--supervise` Boolean Flag

- Add a boolean flag that enables supervision when present.
- Leave the default unsupervised.

Why not choose this now:

- It is workable, but less extensible if the project later wants `auto`.
- Help text and config-style reuse are a little less explicit than `on|off`.

### Alternative: Keep Supervision As The Default

- Continue always running the worker-plus-supervisor loop.
- Possibly add `--no-supervise` to disable it.

Why not choose this now:

- It does not match the requested product behavior.
- It keeps the heavier orchestration path as the surprise default.

## User-Facing Design

### CLI Option

Add `--supervise` to both `optimize` and `optimize-batch`:

- `--supervise off`
  - Use the original single-agent optimize execution path.
- `--supervise on`
  - Use the explicit worker-round plus supervisor-gate orchestration path.

Default:

- `optimize`: `--supervise off`
- `optimize-batch`: `--supervise off`

The parser should expose `choices=["on", "off"]` instead of a boolean flag so the selected mode is explicit in help output and stored options.

### Examples

Single workspace:

```bash
uv run helix optimize --input operator.py
uv run helix optimize --input operator.py --supervise on
uv run helix optimize --input . --supervise on --min-rounds 10
```

Batch:

```bash
uv run helix optimize-batch --input operators_root
uv run helix optimize-batch --input operators_root --supervise on
```

## Behavior By Mode

### `--supervise off`

Use the original single-agent optimize flow:

1. Stage skills into the workspace.
2. Render the ordinary single-agent optimize guidance for the selected backend.
3. Launch one optimize agent request.
4. Apply the existing stall-recovery and minimum-round logic only if that older path already supports it.
5. Clean up staged files.

This mode is the closest match to the pre-round-gate optimize experience.

### `--supervise on`

Use the explicit round-gate flow:

1. Stage skills into the workspace.
2. Render shared role-neutral guidance plus worker and supervisor role briefs.
3. Launch a worker for one round.
4. Launch a supervisor audit pass.
5. Continue or stop according to the gate result.

This mode is the current artifact-enforcing behavior and remains the recommended path when workflow compliance matters.

## `--interact` Semantics

### `--supervise off`

- Preserve the original interactive optimize behavior.
- The optimize agent runs as one interactive session.

### `--supervise on`

- `worker` may run interactively.
- `supervisor` must always run non-interactively.
- If the session continues after a gate decision, the next worker invocation may again be interactive.

This keeps human interaction focused on optimization work rather than on the audit pass.

## `--min-rounds` Semantics

### `--supervise off`

- Preserve existing single-agent semantics.
- The CLI may still resume or continue as needed if the legacy flow already uses `min_rounds`.

### `--supervise on`

- Keep `min_rounds` enforced only at the orchestration layer.
- Do not mention `min_rounds` in the worker prompt.
- The supervisor/runtime loop remains responsible for deciding whether more rounds are required.

## Architecture

### Optimize Request Model

Extend optimize options and requests with an explicit supervision flag, for example:

```python
supervise: str  # "on" or "off"
```

The parser should validate only the supported values and provide a default of `off`.

This field belongs on optimize-specific option models rather than on unrelated command paths.

### Runtime Split

Refactor `run_optimize_request()` so it chooses between two runtime paths:

- `run_optimize_request_unsupervised(request, ...)`
- `run_optimize_request_supervised(request, ...)`

Both paths should share:

- skill staging
- optimize request construction
- output rendering
- cleanup and warning reporting

They should differ only in how the optimize agent is orchestrated after preparation.

### Guidance Split

When `--supervise off`:

- Do not render `.helix/roles/`
- Do not render `round-brief.md` or `supervisor-report.md`
- Do not launch the `optimize-supervisor` skill
- Do not write shared role-neutral orchestration guidance unless the unsupervised flow still needs the older single-agent guidance file

When `--supervise on`:

- Keep the existing shared-guidance plus role-brief layout
- Keep the current worker and supervisor role separation

## Batch Behavior

`optimize-batch` should pass the selected supervise mode through to each workspace request.

- `--supervise off`
  - every workspace runs the single-agent optimize path
- `--supervise on`
  - every workspace runs the worker-plus-supervisor path

This keeps batch behavior predictable and aligned with single-workspace optimize.

## Error Handling

- Unsupported `--supervise` values should fail in parser validation with a short actionable error.
- `--supervise on` should continue using the existing round-gate retry and gate failure behavior.
- `--supervise off` should preserve the current single-agent failure and recovery behavior.
- Switching between modes must not delete user-owned optimize artifacts from prior runs.

## Documentation

Update:

- `README.md`
  - explain `--supervise on|off`
  - state that the default is `off`
  - describe when supervised optimize is useful
- `AGENTS.md`
  - only if the project wants a durable rule about supervise mode defaults or expectations

## Testing

Add CLI and runtime coverage for:

- parser accepts `--supervise on|off` for `optimize`
- parser accepts `--supervise on|off` for `optimize-batch`
- default supervise mode is `off`
- unsupervised optimize chooses the single-agent runtime path
- supervised optimize chooses the round-gate runtime path
- `--interact` under supervised optimize keeps worker interactive and supervisor non-interactive
- batch optimize passes supervise mode through to each workspace request

## Recommendation

Ship `--supervise on|off` with default `off`.

This keeps the current round-gate investment available without forcing it on every optimize user. It also gives the project a clean migration point if supervised optimize later becomes robust enough to justify an `auto` mode or a new default.
