# Generation Overwrite Control

## Summary

Add explicit overwrite control for generated test and benchmark files.

User-visible behavior comes first:

- `gen-test` and `gen-bench` should refuse to overwrite an existing output file by default.
- Users can opt in to replacement with `--force-overwrite`.
- The overwrite flag should only affect generation commands and should not change `run-*` or `optimize` behavior.

## CLI Semantics

- Add `--force-overwrite` to:
  - `gen-test`
  - `gen-bench`
- If the resolved output path already exists and `--force-overwrite` is not set, the CLI should fail before launching the code agent.
- If `--force-overwrite` is set, the CLI should remove the existing output file before launching the code agent, then tell the agent that replacing that output is allowed.

## Implementation Notes

- Keep overwrite validation and pre-run cleanup in the CLI orchestration layer because they are local file safety rules.
- Pass the overwrite choice into the agent request and prompt construction so the backend prompt stays aligned with CLI behavior.
- Keep the default protective behavior explicit in tests and user-facing documentation.
