# Optimize Auto Bench Mode Design

## Summary

`optimize` and `optimize-batch` currently reject `--resume auto --bench-mode ...` whenever a workspace resolves to an existing resumable optimize session. That makes batch reruns awkward because one explicit batch-wide benchmark preference can fail the whole command even when some workspaces are fresh and others are resumable. This change keeps continuation metadata authoritative while allowing `--resume auto` to accept an explicit `--bench-mode`.

## Goals

- Let `optimize` and `optimize-batch` accept `--resume auto --bench-mode ...`.
- Preserve existing continuation-session metadata as the authority when a workspace resumes.
- Keep mixed fresh-and-resume batch runs usable with one batch-wide `--bench-mode`.
- Limit the change to benchmark mode; do not broaden continuation overrides in general.

## Non-Goals

- Do not rewrite existing benchmark harness metadata or `baseline/state.json` during resume resolution.
- Do not add a new CLI flag for force-overriding existing optimize-session benchmark mode.

## CLI Contract

- `optimize --resume auto --bench-mode <mode>` is always valid; for resumable sessions the explicit mode must match existing harness metadata.
- `optimize-batch --resume auto --bench-mode <mode>` is valid per workspace; resumable workspaces validate the match independently.
- `optimize --resume continue --bench-mode <mode>` is valid when the explicit mode matches existing harness metadata; mismatches fail.
- `optimize-batch --resume continue --bench-mode <mode>` same per-workspace assertion semantics.

## Mode Resolution

### Fresh workspace

- If the workspace resolves to `no-session`, keep current behavior:
  - use the requested `--bench-mode` when provided
  - otherwise default to `torch-npu-profiler`

### Resumable workspace

- If the workspace resolves to `resumable-session`, continue the existing session.
- Reuse the benchmark mode recorded in the existing benchmark harness metadata.
- If an explicit `--bench-mode` is provided, validate it against the recorded metadata:
  - matching values succeed
  - conflicting values fail with a mode-specific error
- Applies equally to `--resume auto` and `--resume continue`.

### Partial workspace

- Keep current failure behavior for partial optimize state.
- Partial-session errors take precedence over mode-conflict errors.

## Batch Behavior

- `optimize-batch` resolves each workspace independently with the same assertion semantics.
- If one workspace has a mode conflict, that workspace fails while unrelated workspaces continue using their own resolved modes.
- The batch summary reports the workspace-level failure normally.

## Rationale

- Existing optimize sessions already have a benchmark harness and baseline metadata that define the session's benchmark contract.
- Explicit resume-time modes are assertions about what the user believes the existing session contract already is.
- Failing on conflict is honest: the user asked for one contract, the workspace records another.

## Documentation

- Update `README.md` so `optimize-batch` and `optimize` documentation explains that:
  - `--resume auto --bench-mode ...` and `--resume continue --bench-mode ...` are accepted
  - resumed workspaces validate explicit modes against existing harness metadata
  - matching assertions succeed, conflicting assertions fail

## Testing

- Add CLI coverage for `--resume auto` with resumable session: matching `--bench-mode` succeeds, conflicting fails.
- Add CLI coverage for `--resume continue`: matching `--bench-mode` succeeds, conflicting fails.
- Add batch coverage for mixed fresh/resumable roots with matching explicit modes.
- Cover `--test-mode` with identical assertion semantics.
