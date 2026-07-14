# Claude Plugin Builder Remove Agents Design

## Summary

Stop the Claude plugin builder from packaging the Helix optimize and convert agent definitions. The plugin continues to package skills and main-session hooks.

## Problem

The builder still emits `agents/helix-optimizer.md` and `agents/helix-convert.md`. These agent definitions are no longer part of the desired Claude plugin surface.

## Goals

- The built plugin contains no `agents/` directory.
- Remove agent-only rendering code and generated README instructions that invoke an agent.
- Keep bundled skills and existing main-session hook behavior unchanged.

## Non-Goals

- Do not remove packaged skills.
- Do not change session hook behavior.
- Do not change skill selection or staging behavior.

## Testing Strategy

- Assert builder assets contain no agent definition files.
- Assert a built plugin has no `agents/` directory and keeps its skills and hooks.
