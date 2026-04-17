# Optimize Batch Status File Design

## Summary

- Add an explicit root-level status file for `optimize-batch` so reruns can skip workspaces that were already completed.
- Keep this behavior separate from single-workspace `optimize --resume` session reuse.
- Use one visible file at `<batch-root>/optimize-batch-status.json`.

## Goals

- Let `optimize-batch` skip previously completed workspaces on rerun.
- Keep the completion signal explicit instead of inferring it from optimize artifacts.
- Keep the behavior local to `optimize-batch`.
- Preserve actionable behavior when the status file is missing or malformed.

## Non-Goals

- Do not change `optimize --resume` semantics.
- Do not teach `optimize-status` to infer or manage batch completion.
- Do not add per-workspace marker files.
- Do not add timestamps or history in the first version.

## Status File

Path:

- `<batch-root>/optimize-batch-status.json`

Shape:

```json
{
  "version": 1,
  "workspaces": {
    "matmul": {
      "status": "completed",
      "operator_file": "kernel.py"
    },
    "layernorm": {
      "status": "incomplete",
      "operator_file": "op.py"
    }
  }
}
```

Rules:

- Keys under `workspaces` are workspace paths relative to the batch root.
- Each entry stores:
  - `status`: `completed` or `incomplete`
  - `operator_file`: the operator filename selected for that workspace
- Unknown top-level keys are ignored.
- Unknown workspace fields are ignored.

## Skip Semantics

- `optimize-batch` reads the status file once before scheduling work.
- A workspace is skipped only when:
  - the status file parses successfully
  - the workspace has an entry
  - the entry status is `completed`
  - the recorded `operator_file` still matches the operator file selected for this run
- Any other case means the workspace is runnable.

Matching the operator filename avoids stale completion records surviving a workspace operator rename.

## Write Semantics

- When a workspace optimize run succeeds, write or update its entry to:
  - `status: "completed"`
  - `operator_file: <selected filename>`
- When a workspace optimize run fails, write or update its entry to:
  - `status: "incomplete"`
  - `operator_file: <selected filename>`
- Rewrite the whole JSON file atomically enough for local CLI use by writing complete JSON content on each update.

## Reset Semantics

- `optimize-batch --reset-optimize` already requests a fresh optimize run for each workspace.
- In batch mode, it should also delete `<batch-root>/optimize-batch-status.json` before discovery and scheduling.
- If the file does not exist, reset remains a no-op.

## Error Handling

- Missing status file: treat all workspaces as runnable.
- Malformed JSON: treat all workspaces as runnable and continue.
- Invalid entry shape for one workspace: ignore that entry and continue.
- Status-file issues must not fail the whole batch command.

## User-Visible Output

- Add a third batch result state: `SKIP`.
- A skipped workspace should render a short message such as `already completed`.
- Batch summary should report:
  - succeeded
  - failed
  - skipped

Example:

```text
[SKIP] matmul: already completed
[OK] layernorm: optimized kernel.py
Summary: 1 succeeded, 0 failed, 1 skipped
```

## Implementation Notes

- Keep status-file logic in `src/triton_agent/optimize/batch.py` or an optimize-batch-local helper module.
- Do not route this through skills or prompt logic.
- Keep the result rendering update narrow to batch optimize output.

## Verification

- Unit tests for reading and validating the status file.
- Batch runtime tests for:
  - skipping completed workspaces
  - writing completed after success
  - writing incomplete after failure
  - ignoring malformed status files
  - clearing the file on `--reset-optimize`
- CLI tests for updated summary text.
