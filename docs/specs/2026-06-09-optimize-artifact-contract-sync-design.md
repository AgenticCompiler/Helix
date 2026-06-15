# Optimize Artifact Contract Sync Design

## Goal

Remove duplicated baseline schema from the round submit contract, make path-bearing state fields resolve relative to their owning state file first, and generate the optimize artifact reference from the submit-contract JSON sources so the docs stay in sync.

## User-Visible Semantics

- `skills/triton-npu-optimize-submit-baseline/references/contract.json` remains the source of truth for baseline state required fields and their descriptions.
- `skills/triton-npu-optimize-submit-round/references/contract.json` becomes the source of truth for round state fields and their descriptions instead of duplicating baseline fields.
- Any state field that represents a file or directory path must be written relative to the directory that contains that state JSON file.
- Baseline and round checkers first resolve declared paths relative to the state file directory.
- For compatibility with agent mistakes and older outputs, if a declared path does not exist there, the checker retries resolution relative to the operator workspace root before reporting a missing artifact.
- `skills/triton-npu-optimize/references/artifacts.md` keeps the surrounding workflow guidance, but its baseline-state and round-state sections are regenerated from the two contract JSON files by a dedicated script.

## Implementation Notes

- Keep the CLI thin: the behavior change lives in the submit skill scripts plus their shared contract readers.
- Store field descriptions in JSON structures that preserve required-field ordering and allow the doc generator to render markdown in JSON-like form.
- Cover the new resolution rules with tests that prove both the preferred state-relative lookup and the workspace-relative fallback path.
- Cover the generated artifacts reference with tests so manual drift is caught in CI.
