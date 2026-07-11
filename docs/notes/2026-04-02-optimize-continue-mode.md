# Optimize Resume Mode

## Summary

- Replace the old boolean continue flag with `optimize --resume {auto,continue,fresh}`.
- `auto` should resume a complete existing optimization session, start fresh when no optimize artifacts exist, and fail explicitly for partial optimize state.
- Continue-path optimize runs allow `--test-mode` and `--bench-mode` only when they match existing harness metadata; mismatches are rejected.

## User-Visible Behavior

- `uv run helix optimize --input <operator.py> --resume auto` is the default entrypoint.
- `uv run helix optimize --input <operator.py> --resume continue` requires an existing optimize session.
- `uv run helix optimize --input <operator.py> --resume fresh` requires a clean optimize workspace.
- A resumable optimize session requires:
  - `opt-note.md` in the operator workspace
  - at least one `opt-round-*` directory in the operator workspace
  - an existing generated test harness with readable `# test-mode: ...` metadata
  - an existing generated benchmark harness with readable `# bench-mode: ...` metadata
- `resume auto` starts fresh only when the workspace has no optimize artifacts at all.
- If the workspace contains partial optimize artifacts, `resume auto` fails immediately with a short actionable error instead of guessing.
- In continuation paths (`--resume auto` with an existing session and `--resume continue`), `--test-mode` and `--bench-mode` are allowed only when they match the existing session's modes; mismatches are rejected because the optimize session already has established validation artifacts and modes.

## Mode Resolution

- Fresh optimize runs keep the existing behavior:
  - default test mode: `differential`
  - default bench mode: `torch-npu-profiler`
- Continue-path optimize runs (`--resume auto` with an existing session and `--resume continue`) resolve modes from existing generated harness metadata:
  - test mode from the existing optimize test harness
  - bench mode from the existing optimize benchmark harness
- Explicit `--test-mode` and `--bench-mode` flags on these paths are treated as assertions: matching values succeed, conflicting values fail.
- If multiple plausible test harnesses exist for a continuation path, fail explicitly instead of guessing.

## Prompt Contract

- Fresh optimize prompts keep the existing long-running guidance.
- Continue-path optimize prompts must additionally say:
  - this is a continuation of an existing optimization session
  - do not restart from scratch
  - read `opt-note.md`, existing `opt-round-*`, and existing round logs before making changes

## Implementation Notes

- Add `--resume {auto,continue,fresh}` to both `optimize` and `optimize-batch`.
- Store the raw CLI choice as a string resume mode instead of a boolean continue flag.
- For optimize parsing, keep parser-level `--test-mode` and `--bench-mode` defaults unset so continuation paths can distinguish omitted flags from explicit overrides.
- Add small helper functions in the CLI layer to:
  - classify optimize workspaces as no-session, resumable-session, or partial-session
  - validate continuation prerequisites
  - resolve test and bench modes from metadata
- Keep this behavior in the CLI orchestration layer; do not move session-detection logic into the optimize skill itself.

## Documentation Updates

- Update `README.md` to document `--resume`.
- Update `AGENTS.md` to describe explicit resume-mode semantics and mode reuse.

## Verification

- Parser tests for `optimize --resume` and `optimize-batch --resume`.
- CLI tests for:
  - allowing `--resume continue --test-mode` when it matches existing metadata and rejecting mismatches
  - allowing `--resume continue --bench-mode` when it matches existing metadata and rejecting mismatches
  - resolving `resume auto` to fresh for no-session workspaces
  - rejecting partial optimize state in `resume auto`
  - rejecting existing optimize artifacts in `resume fresh`
  - resolving modes from existing metadata on continuation paths
- Prompt test for continuation wording.
- Full repo verification with ruff, pyright, and unittest.
