# Convert Iterative Repair Design

## Goals

- Update the convert workflow contract so agents reuse an existing operator test case when one is already present in the workspace.
- Keep new test generation as the fallback only when no reusable test exists or the user explicitly asks to regenerate it.
- Add a CLI-owned post-convert validation loop for `convert` so a failed converted-operator test can trigger one follow-up repair invocation with concrete failure context.

## User-Visible Semantics

- The `triton-npu-convert-pytorch-operator` skill must explicitly tell the agent:
  - if the operator workspace already contains a suitable test case, reuse it
  - do not generate a new test unless no suitable test exists or the user explicitly asks to regenerate
- A `convert` run still begins with one normal agent invocation that performs the conversion work.
- After that invocation returns successfully, the CLI must run one explicit verification pass against the converted operator:
  - if the reused/generated harness is standalone, rerun it once against the converted operator and require success
  - if the reused/generated harness is differential, rerun it against the converted operator and compare against the original operator via `--baseline-operator-file` semantics
- If that verification fails, `convert` should launch one repair follow-up invocation in the same workspace with the failure summary appended to the prompt, then rerun the same verification once.
- If the repair follow-up still fails verification, stop and return the failing status instead of looping indefinitely.

## Test Reuse Rule

- Prefer the default differential test path for the source operator: `differential_test_<input-stem>.py`.
- If that path does not exist, allow reuse of the default standalone test path: `test_<input-stem>.py`.
- If the default paths are absent, allow reuse of exactly one existing differential test file or exactly one existing standalone test file in the workspace.
- If multiple candidate reusable tests exist for the chosen mode and the default path is absent, fail explicitly with an actionable message instead of guessing.

## Implementation Boundaries

- Keep the thin CLI boundary: the new loop belongs in convert command/orchestration code, not in the convert skill scripts.
- Reuse the existing `helix.execution` and comparison helpers instead of shelling out through the CLI.
- Preserve current batch-convert behavior in this change; only single-workspace `convert` gains the extra CLI-owned repair loop.

## Non-Goals

- Do not add a new convert-specific session model.
- Do not add benchmark verification to `convert`.
- Do not change `convert-batch` to use the new repair loop in this change.
