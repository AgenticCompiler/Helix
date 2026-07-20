# OpenCode Default Agent

## User-visible behavior

Every Helix command that accepts `--agent` defaults to `opencode` when the
option is omitted. Users can continue to select any supported backend
explicitly with `--agent`.

This applies consistently to generation, conversion, log checking,
optimization, distillation, and reporting commands, including their batch
variants.

## Implementation

The CLI parser retains command-specific defaults where present and changes the
shared fallback for agent-backed command specifications from `codex` to
`opencode`. Parser tests cover every agent-backed command without an explicit
`--agent` value.

## Non-goals

This change does not alter backend choices, backend invocation behavior, or an
explicit user-selected agent.
