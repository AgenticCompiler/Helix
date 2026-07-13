# OpenCode Interactive Command Design

## Goal

Fix the OpenCode interactive backend command so `helix ... --agent opencode --interact` launches the installed `opencode` CLI successfully.

## Problem

The current interactive OpenCode runner builds commands like:

```text
opencode <workspace> --pure --thinking --prompt <prompt>
```

Local `opencode --help` shows that:

- top-level `opencode [project]` accepts `--prompt`
- top-level `opencode [project]` does not accept `--thinking`
- `--thinking` is available on `opencode run`

This means the interactive runner currently mixes top-level project mode with a flag that only exists on the `run` subcommand, so OpenCode exits before launching.

## User-Visible Behavior

- `--agent opencode --interact` should launch top-level OpenCode project mode with the workspace path and `--prompt`.
- Default interactive OpenCode runs should still include `--pure`.
- `--enable-agent-hooks` should still omit `--pure` so the staged project plugin can load.
- Non-interactive OpenCode runs should stay unchanged.

## Design

Update only the interactive branch in `src/helix/backends/opencode.py`:

- keep the top-level project positional workspace argument
- keep `--prompt <prompt>`
- remove `--thinking`
- preserve the existing `--pure` omission when hooks are enabled

The non-interactive branch remains:

```text
opencode run --dir <workspace> [--pure] --dangerously-skip-permissions --thinking <prompt>
```

## Testing

- Update OpenCode runner unit tests so interactive commands no longer expect `--thinking`.
- Add an assertion that the workspace positional argument is still preserved.
- Run focused OpenCode runner tests after the backend change.

## Scope

- Do not change optimize prompt construction.
- Do not change non-interactive OpenCode command assembly.
- Do not change hook staging behavior beyond preserving the existing `--pure` rule.
