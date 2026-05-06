# Single-Operator `--input` Workdir Design

## Summary

Unify the single-operator command semantics for `--input` so directory inputs behave like workspace roots instead of being treated as plain paths whose parent becomes the agent working directory.

## User-Facing Behavior

- For single-operator commands, `--input <file>` keeps the existing behavior:
  - resolve the operator from that file path
  - set `workdir` to the file's parent directory
- For single-operator commands, `--input <dir>` should:
  - resolve exactly one candidate operator file from that directory using the command's existing workspace rules
  - set `workdir` to that directory itself

## In Scope

- `gen-test`
- `gen-bench`
- `gen-eval`
- `optimize`

## Out Of Scope

- Batch commands such as `gen-eval-batch` and `optimize-batch`
- Commands that already require a workspace directory, such as `verify`
- Changes to remote workspace creation semantics beyond the local `workdir` selected for the request

## Design

Add a shared helper for single-operator `--input` resolution:

- if the input path is a file, return `(operator_path=input_path, workdir=input_path.parent)`
- if the input path is a directory, resolve the operator file inside that directory and return `(operator_path=resolved_file, workdir=input_path)`

`gen-test`, `gen-bench`, and `gen-eval` should reuse the same directory-to-operator resolution behavior so they match `convert` and `optimize` workspace usage. `optimize` should keep its existing candidate-file rules, but stop collapsing directory inputs to the resolved file's parent.

## Validation

- Add command-handler tests proving directory inputs keep the workspace directory as `workdir`
- Cover both generation and optimize entrypoints
- Update README wording for generation commands if the current text still implies file-only input
