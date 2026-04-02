# Optimize Continue Mode

## Summary

- Add an explicit `optimize --continue` mode for resuming an existing optimization session instead of starting a fresh search.
- In continue mode, the CLI must reject new `--test-mode` and `--bench-mode` overrides.
- Continue mode must validate that the workspace already contains optimization state and existing validation harnesses before launching the code agent.

## User-Visible Behavior

- `uv run triton-agent optimize --input <operator.py> --continue` resumes optimization from the existing workspace state.
- Continue mode requires:
  - `opt-note.md` in the operator workspace
  - at least one `opt-round-*` directory in the operator workspace
  - an existing generated test harness with readable `# test-mode: ...` metadata
  - an existing generated benchmark harness with readable `# bench-mode: ...` metadata
- If any of those are missing, the CLI fails immediately with a short actionable error.
- In continue mode, `--test-mode` and `--bench-mode` are rejected because the optimize session already has established validation artifacts and modes.

## Mode Resolution

- Fresh optimize runs keep the existing behavior:
  - default test mode: `differential`
  - default bench mode: `standalone`
- Continue optimize runs resolve modes from existing generated harness metadata:
  - test mode from the existing optimize test harness
  - bench mode from the existing optimize benchmark harness
- If multiple plausible test harnesses exist for continue mode, fail explicitly instead of guessing.

## Prompt Contract

- Fresh optimize prompts keep the existing long-running guidance.
- Continue optimize prompts must additionally say:
  - this is a continuation of an existing optimization session
  - do not restart from scratch
  - read `opt-note.md`, existing `opt-round-*`, and existing round logs before making changes

## Implementation Notes

- Add `--continue` only to the `optimize` subcommand.
- Store it under a non-keyword destination such as `continue_optimize`.
- For optimize parsing, stop using parser-level default values for `--test-mode` and `--bench-mode`; resolve defaults later in `main()` so continue mode can distinguish omitted flags from explicitly provided ones.
- Add small helper functions in the CLI layer to:
  - validate continue prerequisites
  - locate existing optimize harnesses
  - resolve test and bench modes from metadata
- Keep this behavior in the CLI orchestration layer; do not move session-detection logic into the optimize skill itself.

## Documentation Updates

- Update `README.md` to document `optimize --continue`.
- Update `AGENTS.md` to describe explicit continue-mode semantics and mode reuse.

## Verification

- Parser tests for `optimize --continue`.
- CLI tests for:
  - rejecting `--continue --test-mode`
  - rejecting `--continue --bench-mode`
  - rejecting missing `opt-note.md`
  - rejecting missing `opt-round-*`
  - resolving modes from existing metadata in continue mode
- Prompt test for continue wording.
- Full repo verification with ruff, pyright, and unittest.
