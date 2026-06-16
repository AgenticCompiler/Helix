# Optimize Resume Mode Assertion Semantics

## Summary

- `--test-mode` and `--bench-mode` must follow the same resume-time semantics.
- On an existing resumable optimize session, explicit mode flags are assertions about the existing session contract, not overrides.
- If an explicit mode matches the recorded harness metadata, the command succeeds.
- If an explicit mode conflicts with the recorded harness metadata, the command fails with a short actionable error.
- Resume resolution must never silently rewrite or override existing harness metadata.

## Problem

The current optimize resume behavior has drifted across implementation, tests, and design documents:

- some places treat explicit mode flags as forbidden during resume
- some places treat them as ignored for resumed workspaces
- some places treat them as resumable-session assertions

This inconsistency makes it unclear what `--resume auto` and `--resume continue` are supposed to mean, especially when users pass explicit `--test-mode` or `--bench-mode`.

## Design Principles

### Principle 1: Test And Bench Modes Share One Contract

`--test-mode` and `--bench-mode` must use the same decision rules during resume resolution. The CLI should not special-case one while treating the other as an override, an ignored hint, or a forbidden input.

### Principle 2: Explicit Resume-Time Modes Are Assertions

When the workspace already contains a resumable optimize session, an explicit `--test-mode` or `--bench-mode` asserts what the user believes the existing session contract already is.

That means:

- matching explicit values are allowed
- conflicting explicit values are rejected
- omitted values reuse the existing recorded modes

### Principle 3: Existing Session Metadata Remains Authoritative

For resumable sessions, effective modes come from the existing generated harness metadata. Explicit CLI values may validate that metadata, but they do not mutate it and do not replace it.

### Principle 4: Conflicts Must Fail, Not Warn-And-Continue

When a user explicitly provides a mode that conflicts with the existing optimize session, the command should fail for that workspace instead of warning and silently continuing with another mode.

This keeps the command honest:

- the user asked for one contract
- the workspace already records another contract
- the CLI should surface that contradiction directly

## User-Visible Behavior

### `--resume fresh`

- The workspace must be classified as `no-session`.
- Explicit `--test-mode` and `--bench-mode` define the new optimize session.
- If either mode is omitted, the command uses the existing fresh-run defaults.

### `--resume auto` with `no-session`

- The command behaves like a fresh optimize start.
- Explicit `--test-mode` and `--bench-mode` define the new optimize session.
- If either mode is omitted, the command uses the existing fresh-run defaults.

### `--resume auto` with `resumable-session`

- The command resumes the existing optimize session.
- Effective modes come from the recorded test and benchmark harness metadata.
- If the user explicitly supplies a mode that matches recorded metadata, the command succeeds.
- If the user explicitly supplies a mode that conflicts with recorded metadata, the command fails for that workspace.

### `--resume continue`

- The workspace must be classified as `resumable-session`.
- Effective modes come from the recorded test and benchmark harness metadata.
- If the user explicitly supplies a mode that matches recorded metadata, the command succeeds.
- If the user explicitly supplies a mode that conflicts with recorded metadata, the command fails.

### Partial Optimize State

- If the workspace is `partial-session`, resume resolution fails before mode assertions are considered.
- The error should continue to explain the partial optimize-state problem rather than reporting a mode conflict first.

## Effective Mode Matrix

| Resume mode | Workspace state | Explicit mode omitted | Explicit mode matches metadata | Explicit mode conflicts with metadata |
| --- | --- | --- | --- | --- |
| `fresh` | `no-session` | use fresh default | use explicit value | use explicit value |
| `auto` | `no-session` | use fresh default | use explicit value | use explicit value |
| `auto` | `resumable-session` | reuse metadata | reuse metadata and succeed | fail |
| `continue` | `resumable-session` | reuse metadata | reuse metadata and succeed | fail |

## Conflict Errors

Mode conflicts should use direct errors that name both values. For example:

- `--test-mode standalone conflicts with existing harness test-mode differential`
- `--bench-mode msprof conflicts with existing harness bench-mode torch-npu-profiler`

These errors should be identical in meaning for `resume auto` and `resume continue`. Only workspace-state preconditions differ between those resume modes.

## Batch Behavior

`optimize-batch` should apply the same rules independently per workspace.

- fresh workspaces use explicit values or defaults
- resumable workspaces reuse metadata and validate explicit assertions
- partial workspaces fail before mode assertion logic

If one workspace has a mode conflict:

- that workspace fails
- unrelated workspaces continue using their own resolved modes
- the batch summary reports the workspace-level failure normally

This keeps batch behavior consistent with existing per-workspace resume resolution and avoids hidden batch-only mode semantics.

## Implementation Guidance

The implementation should separate two concerns:

1. workspace-state classification
2. effective-mode resolution plus explicit-assertion validation

Recommended structure:

- keep `resolve_optimize_resume()` responsible for deciding fresh vs resumable vs partial handling
- add one shared helper for mode resolution and conflict validation
- run both `test-mode` and `bench-mode` through that same helper

The helper contract should look conceptually like this:

- if no existing mode is present, use the requested value or default
- if an existing mode is present and no requested value is provided, reuse the existing value
- if an existing mode is present and the requested value matches it, reuse the existing value
- if an existing mode is present and the requested value conflicts, raise a mode-specific error

The implementation should not:

- ignore conflicting explicit mode flags
- treat `test-mode` and `bench-mode` differently
- rewrite `baseline/state.json`
- rewrite harness metadata during resume resolution

## Documentation Changes Required

After implementation, the repo should present one consistent contract across:

- `README.md`
- `docs/notes/2026-04-02-optimize-continue-mode.md`
- `docs/specs/2026-04-28-optimize-auto-bench-mode-design.md`
- any implementation plan that still says resumable `auto` rejects all explicit overrides

The unified documentation rule should be:

- explicit modes on resumable sessions are assertions
- matching assertions succeed
- conflicting assertions fail
- existing harness metadata remains authoritative

## Testing Requirements

Add or keep coverage for these cases:

- `resume continue` with matching `--test-mode`
- `resume continue` with conflicting `--test-mode`
- `resume continue` with matching `--bench-mode`
- `resume continue` with conflicting `--bench-mode`
- `resume auto` on a resumable session with matching `--test-mode`
- `resume auto` on a resumable session with conflicting `--test-mode`
- `resume auto` on a resumable session with matching `--bench-mode`
- `resume auto` on a resumable session with conflicting `--bench-mode`
- `resume auto` on a fresh workspace using explicit modes
- mixed batch roots where each workspace resolves its own state independently

The tests should assert both:

- command success or failure
- the effective request modes passed downstream

## Non-Goals

- Introducing force-override flags that mutate an existing optimize session's recorded modes
- Rewriting existing harness metadata when users pass a conflicting explicit mode
- Making batch mode resolution behave differently from single-workspace resolution
