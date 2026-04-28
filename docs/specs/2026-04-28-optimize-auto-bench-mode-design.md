# Optimize Auto Bench Mode Design

## Summary

`optimize` and `optimize-batch` currently reject `--resume auto --bench-mode ...` whenever a workspace resolves to an existing resumable optimize session. That makes batch reruns awkward because one explicit batch-wide benchmark preference can fail the whole command even when some workspaces are fresh and others are resumable. This change keeps continuation metadata authoritative while allowing `--resume auto` to accept an explicit `--bench-mode`.

## Goals

- Let `optimize` and `optimize-batch` accept `--resume auto --bench-mode ...`.
- Preserve existing continuation-session metadata as the authority when a workspace resumes.
- Keep mixed fresh-and-resume batch runs usable with one batch-wide `--bench-mode`.
- Limit the change to benchmark mode; do not broaden continuation overrides in general.

## Non-Goals

- Do not allow `--resume continue --bench-mode ...`.
- Do not change `--test-mode` continuation validation.
- Do not rewrite existing benchmark harness metadata or `baseline/state.json` during resume resolution.
- Do not add a new CLI flag for force-overriding existing optimize-session benchmark mode.

## CLI Contract

- `optimize --resume auto --bench-mode <mode>` is valid.
- `optimize-batch --resume auto --bench-mode <mode>` is valid.
- `optimize --resume continue --bench-mode <mode>` remains invalid.
- `optimize-batch --resume continue --bench-mode <mode>` remains invalid.

## Mode Resolution

### Fresh workspace under `--resume auto`

- If the workspace resolves to `no-session`, keep current behavior:
  - use the requested `--bench-mode` when provided
  - otherwise default to `standalone`

### Resumable workspace under `--resume auto`

- If the workspace resolves to `resumable-session`, continue the existing session.
- Reuse the benchmark mode recorded in the existing benchmark harness metadata.
- Ignore the explicit `--bench-mode` value for that workspace instead of failing.
- The resolved request and prompt should use the reused recorded benchmark mode, not the ignored CLI override.

### Partial workspace under `--resume auto`

- Keep current failure behavior for partial optimize state.
- Accepting `--bench-mode` must not weaken the existing partial-session safety checks.

## Batch Behavior

- `optimize-batch` should continue resolving each workspace independently.
- In one batch run:
  - fresh workspaces may use the explicit `--bench-mode`
  - resumable workspaces may ignore that explicit value and keep their recorded mode
- A mixed batch must not fail only because a resumable workspace received `--bench-mode` under `--resume auto`.
- `--resume continue` keeps the current strict per-workspace validation in batch mode.

## Rationale

- Existing optimize sessions already have a benchmark harness and baseline metadata that define the session's benchmark contract.
- Fresh workspaces still benefit from an explicit batch-wide benchmark preference.
- Ignoring the explicit override on resumed workspaces is safer than silently mutating existing session metadata, and more usable than rejecting the whole command.

## Documentation

- Update `README.md` so `optimize-batch` and `optimize` documentation explains that:
  - `--resume auto --bench-mode ...` is accepted
  - resumed workspaces keep their existing benchmark mode
  - only fresh workspaces adopt the explicit override

## Testing

- Add CLI coverage showing `optimize --resume auto --bench-mode msprof` still succeeds for a resumable session and uses the recorded benchmark mode.
- Add batch coverage showing `optimize-batch --resume auto --bench-mode msprof` can process a mixed fresh/resumable root without resume-validation failure.
- Keep existing tests proving `--resume continue --bench-mode ...` is rejected.
