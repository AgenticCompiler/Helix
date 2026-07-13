# Batch Operator Filter Design

## Summary

- Add a shared `--operator-filter <glob>` option to `optimize-batch`, `gen-eval-batch`, and `convert-batch`.
- Apply the filter only after each command's existing generated-artifact exclusions have already reduced the candidate `.py` files for one workspace.
- Match only the candidate file basename with shell-style glob syntax and keep the existing `0 / 1 / many` candidate resolution contract.

## Goals

- Let users disambiguate multi-file workspaces without renaming files or changing the built-in exclusion rules.
- Keep the filter behavior aligned across `optimize-batch`, `gen-eval-batch`, and `convert-batch`.
- Keep single-workspace and root-fallback batch semantics unchanged apart from the additional candidate filtering.
- Keep the CLI thin by treating this as batch workspace selection behavior, not workflow execution behavior.

## Non-Goals

- Do not add this option to single-workspace commands such as `optimize`, `gen-eval`, or `convert`.
- Do not change the existing exclusion rules for `test_*`, `bench_*`, `opt_*`, `triton_*`, or `__init__.py`.
- Do not introduce regex matching, recursive path matching, or path-based selector semantics.
- Do not widen this change into a generic batch workspace query language.

## User-Facing Behavior

### Supported commands

The following commands accept the new option:

- `helix optimize-batch`
- `helix gen-eval-batch`
- `helix convert-batch`

The new flag shape is:

```text
--operator-filter <glob>
```

### Matching semantics

- The value uses shell-style glob matching.
- Matching is against the candidate file basename only.
- Matching is case-sensitive.
- The batch scanner still looks only at direct files under one workspace directory.
- Directory separators have no meaning because paths are not matched.

Supported glob syntax:

- `*` matches any number of characters
- `?` matches one character
- `[abc]` matches one character from a set
- `[!abc]` matches one character not in a set

Examples:

- `kernel.py`
- `triton_*.py`
- `*_fp16.py`
- `kernel?.py`

### Resolution order

For each workspace, operator selection should happen in this order:

1. Enumerate direct child files in the workspace.
2. Apply the existing command-specific candidate rules:
   - `.py` suffix required
   - excluded names removed
   - excluded prefixes removed
3. If `--operator-filter` is present, keep only candidate basenames that match the user glob.
4. Reuse the existing final resolution contract:
   - zero candidates is an error
   - one candidate is selected
   - more than one candidate is an error

This means the new option narrows the already valid candidate set. It does not re-include files that the built-in exclusions removed earlier.

### Examples

Workspace contents:

```text
kernel.py
triton_kernel.py
test_kernel.py
bench_kernel.py
```

Behavior:

- `gen-eval-batch` without `--operator-filter` still sees `kernel.py` and `triton_kernel.py` as the remaining candidates and fails because there are two.
- `convert-batch --operator-filter 'triton_*.py'` still fails in the same workspace because `convert-batch` excludes `triton_*` before user filtering, so the filter cannot bring `triton_kernel.py` back.
- `optimize-batch --operator-filter 'kernel.py'` selects `kernel.py`.

## Error Handling

- If the user filter removes every remaining candidate, the command should fail for that workspace with an explicit message that mentions the filter value.
- If multiple candidates still remain after applying the filter, the command should keep the existing multi-candidate failure shape while also making it clear that the user filter was applied.
- If the option is omitted, error text and behavior should remain unchanged.

Recommended failure wording:

- `found no candidate operator file after applying --operator-filter 'triton_*.py'`
- `found multiple candidate operator files after applying --operator-filter 'foo*.py': foo.py, foo_v2.py`

## Design

### CLI surface

- Add `--operator-filter` only to the parser definitions for `CommandKind.OPTIMIZE_BATCH`, `CommandKind.GEN_EVAL_BATCH`, and `CommandKind.CONVERT_BATCH`.
- Keep the option outside the shared single-workspace command options because it affects batch workspace selection only.

### Shared helper changes

- Extend `resolve_batch_operator_file(...)` in `src/helix/batch_utils.py` with an optional basename glob filter argument.
- Keep the existing command-specific `is_operator_candidate(...)` callback unchanged.
- Apply the user filter after the built-in candidate filtering and before the final `0 / 1 / many` check.

### Command-specific wiring

- `resolve_batch_optimize_operator_file(...)`, `resolve_batch_gen_eval_operator_file(...)`, and `resolve_batch_convert_operator_file(...)` should accept and pass through the optional filter.
- `run_optimize_batch(...)`, `run_gen_eval_batch(...)`, and `run_convert_batch(...)` should pass the parsed CLI value into the workspace-discovery resolver.
- The batch request options models do not need a new field when the value is only used during batch operator discovery.

### Batch workspace discovery

- `discover_batch_workspaces(...)` should continue to behave the same for:
  - child workspace scanning
  - hidden directory skipping
  - single-workspace root fallback
- The only change is that operator resolution inside each workspace may now reject or disambiguate candidates based on the user filter.

## Testing

- Add CLI parser tests that confirm:
  - `optimize-batch`, `gen-eval-batch`, and `convert-batch` accept `--operator-filter`
  - the parsed value is forwarded to their batch handlers
  - unrelated commands do not gain the flag
- Add batch helper tests that confirm:
  - a filter can reduce multiple valid candidates to one selected file
  - a filter can remove all valid candidates and produce the new no-match error
  - a filter can still leave multiple valid candidates and produce the multi-candidate error
- Add command-level regression tests for the three batch commands so behavior without `--operator-filter` stays unchanged.

## Risks And Mitigations

- Risk: users may expect the filter to match paths or bypass built-in exclusions.
  - Mitigation: keep help text and error messages explicit that matching is basename-only and applies after the existing exclusions.
- Risk: placing this value in workflow option models could blur command ownership.
  - Mitigation: keep the filter in batch selection helpers and handler wiring only.
- Risk: command-specific exclusion differences may surprise users.
  - Mitigation: preserve each command's current exclusion semantics and document that the user filter narrows the post-exclusion candidate set.

## Verification

- Run focused parser and batch-runtime tests for:
  - `tests.test_cli`
  - `tests.test_generation_batch`
  - `tests.test_convert_commands`
  - `tests.test_optimize_runtime`
- Run repository verification before claiming completion:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/`
