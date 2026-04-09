# Optimize Resume Mode Design

## Summary

- Replace the boolean `--continue` flag on `optimize` and `optimize-batch` with `--resume {auto,continue,fresh}`.
- Default `--resume` to `auto` so optimize workflows can reuse complete existing sessions without requiring an extra flag.
- Keep session detection and resume-mode validation in the CLI orchestration layer.
- Prefer explicit failures over silent fallbacks whenever the workspace contains partial optimize state.

## Goals

- Let users rerun `optimize` or `optimize-batch` in a directory with completed rounds without manually remembering `--continue`.
- Keep fresh and continuation workflows explicit and predictable.
- Reuse existing generated harness metadata when a run resumes.
- Protect existing optimize artifacts from accidental overwrite or ambiguous reuse.

## Non-Goals

- Do not add automatic cleanup or deletion of old optimize artifacts.
- Do not move optimize session detection into the optimize skill.
- Do not change optimize round artifact layout or `opt-note.md` semantics.
- Do not change supervisor-driven continuation that already happens after a successful run when `--min-rounds` is not yet satisfied.

## CLI Contract

- `optimize` accepts `--resume {auto,continue,fresh}`.
- `optimize-batch` accepts the same `--resume` option and applies it independently per workspace.
- `--resume` defaults to `auto`.
- The old `--continue` flag is removed instead of kept as a compatibility alias.
- User-facing help text, examples, and documentation are updated to use `--resume` terminology.

### Mode Semantics

- `auto`
  - If the workspace contains a complete resumable optimize session, continue it.
  - If the workspace contains no optimize session artifacts, start a fresh optimize run.
  - If the workspace contains partial optimize artifacts, fail with a short actionable error.
- `continue`
  - Require a complete resumable optimize session.
  - Reuse the modes recorded in existing harness metadata.
  - Fail immediately if the workspace is missing required optimize artifacts or metadata.
- `fresh`
  - Require a fresh optimize workspace.
  - Fail if the workspace already contains optimize session artifacts, including partial state.

## Session Detection

The CLI classifies each optimize workspace into one of three states.

### `no-session`

The workspace has no optimize session artifacts:

- no `opt-note.md`
- no `opt-round-*` directories
- no optimize test harness
- no optimize benchmark harness

Behavior by mode:

- `auto`: start a fresh optimize run
- `continue`: fail
- `fresh`: start a fresh optimize run

### `resumable-session`

The workspace contains a complete optimize session:

- `opt-note.md` exists
- at least one `opt-round-*` directory exists
- exactly one optimize test harness exists:
  - `test_<operator>.py`, or
  - `differential_test_<operator>.py`
- `bench_<operator>.py` exists
- the test harness contains a readable `# test-mode: ...` entry with a supported value
- the benchmark harness contains a readable `# bench-mode: ...` entry with a supported value

Behavior by mode:

- `auto`: continue the existing session
- `continue`: continue the existing session
- `fresh`: fail because optimize artifacts already exist

### `partial-session`

The workspace contains at least one optimize artifact, but does not meet the `resumable-session` requirements.

Examples:

- `opt-note.md` exists but no round directories exist
- round directories exist but `opt-note.md` is missing
- no benchmark harness exists
- both `test_<operator>.py` and `differential_test_<operator>.py` exist
- harness metadata is missing or invalid

Behavior by mode:

- `auto`: fail
- `continue`: fail
- `fresh`: fail

This failure behavior is deliberate. The CLI should not guess whether to resume or restart when a workspace already carries ambiguous optimize state.

## Mode Resolution And Validation

- Fresh optimize runs keep the current defaults:
  - `test-mode`: `differential`
  - `bench-mode`: `standalone`
- Continue-path optimize runs must resolve both modes from existing harness metadata.
- If the effective path is continuation, explicit `--test-mode` and `--bench-mode` overrides are rejected.
- `auto` must reject explicit `--test-mode` or `--bench-mode` when the workspace resolves to a continuation path, because the session already has established harnesses and mode metadata.
- `auto` may still accept explicit `--test-mode` and `--bench-mode` when the workspace resolves to a fresh run.
- `fresh` keeps accepting explicit `--test-mode` and `--bench-mode`.

## Prompt Contract

- Fresh optimize prompts keep the existing long-running optimize wording.
- Continue-path prompts must explicitly state:
  - this is a continuation of an existing optimize session
  - do not restart from scratch
  - read `opt-note.md`, existing `opt-round-*` directories, and existing round logs before making changes
- `auto` uses the same prompt contract as `continue` whenever session detection resolves to a continuation path.

## Batch Behavior

- `optimize-batch` evaluates each immediate child workspace independently.
- One workspace may resolve `auto` to fresh while another resolves `auto` to continue.
- A workspace that resolves to `partial-session` is recorded as a workspace-level failure and does not block unrelated workspaces.
- Batch summaries should make it clear whether a workspace ran in fresh mode, resumed an existing session, or failed during resume-mode validation.

## Error Handling

- Error messages should stay short and actionable.
- Report the concrete missing or conflicting artifact instead of generic state labels.
- Representative examples:
  - `resume continue requires existing opt-note.md: ...`
  - `resume auto found partial optimize state: missing bench_<operator>.py`
  - `resume fresh refused because optimize artifacts already exist in ...`

## Implementation Shape

- Replace the parser's `continue_optimize: bool` state with a string-valued resume mode.
- Add a small workspace classification helper in the optimize orchestration layer.
- Reuse the existing metadata readers for test and benchmark harnesses.
- Build optimize requests from the resolved execution path instead of from the raw CLI mode alone.
- Keep the optimize skill unchanged; this behavior belongs in the CLI wrapper.

## Testing

- Parser tests for `--resume` on `optimize` and `optimize-batch`
- CLI tests for:
  - `auto` selecting fresh in `no-session`
  - `auto` selecting continue in `resumable-session`
  - `auto` rejecting `partial-session`
  - `continue` rejecting missing session artifacts
  - `fresh` rejecting existing optimize artifacts
  - continue-path rejection of explicit `--test-mode`
  - continue-path rejection of explicit `--bench-mode`
- Prompt tests for continuation wording on both explicit `continue` and `auto`-resolved continuation
- Batch tests proving workspace-local resume decisions and isolated failures
- Full verification with `ruff`, `pyright`, and `unittest`
