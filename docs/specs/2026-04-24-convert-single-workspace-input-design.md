# Convert Single-Workspace Input Design

## Summary

- Make single-command `convert` accept one operator workspace directory, not only a direct operator file path.
- When `--input` points to a workspace directory, resolve the single operator file inside that workspace using the same candidate rules as `convert-batch`.
- Use the workspace directory itself as `workdir` so staged skills and backend cwd are placed under that workspace.

## Problem

`convert-batch` already treats a single workspace directory as a valid root and stages skills under that workspace.

Single-command `convert` currently does not. It always sets:

- `workdir = input_path.parent`
- `input_path = args.input` directly

That means `convert --input ./A` stages skills under `./`, not `./A`, and passes the directory path itself through request construction instead of resolving the operator file inside the workspace.

## Goals

- Keep `convert` aligned with batch workspace semantics.
- Ensure staged backend skill directories live under the requested operator workspace.
- Preserve existing file-input behavior for `convert --input a.py`.

## Non-Goals

- Do not change batch candidate-selection rules.
- Do not broaden convert to multi-workspace traversal; that remains `convert-batch`.

## User-Facing Behavior

For file input:

- `convert --input a.py` keeps using `a.py` as the source operator file.
- `workdir` remains `a.py`'s parent directory.

For workspace-directory input:

- `convert --input ./A` resolves the single candidate operator file inside `./A`.
- `workdir` becomes `./A`.
- staged backend skill paths are created under `./A/.<backend>/skills/...`.

If the workspace directory has zero or multiple candidate operator files, `convert` should fail with the same operator-resolution error shape already used by `convert-batch`.

## Validation

Add command tests that verify:

- single-workspace directory input resolves the operator file inside the workspace
- the request `workdir` equals the workspace directory
- the default output path stays inside the workspace
