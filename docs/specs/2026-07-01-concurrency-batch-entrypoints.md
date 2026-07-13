# Concurrency Batch Entrypoints

## User Semantics

Existing `*-batch` commands remain supported. The matching single-workspace commands also accept `--concurrency`; when that option is present, the command runs the existing batch implementation instead of the single-workspace implementation.

This lets users choose either spelling:

- `helix optimize-batch -i operators --concurrency 4`
- `helix optimize -i operators --concurrency 4`

The batch switch is based on the presence of `--concurrency`, not on its numeric value. `--concurrency 1` on a single command still means batch mode with one worker. Omitting `--concurrency` keeps the current single-workspace behavior.

## Command Mapping

- `gen-eval --concurrency ...` runs the `gen-eval-batch` handler.
- `convert --concurrency ...` runs the `convert-batch` handler.
- `optimize --concurrency ...` runs the `optimize-batch` handler.
- `log-check --concurrency ...` runs the `log-check-batch` handler.
- `report --concurrency ...` runs the `report-batch` handler.
- `verify --concurrency ...` runs the `verify-batch` handler and prints a warning that verify batch ignores concurrency.

Batch command spellings keep their current parser defaults and behavior.

## Implementation Notes

The parser should distinguish omitted concurrency from an explicit `--concurrency 1`. Single commands therefore use `default=None` for their newly added concurrency option. Existing batch commands keep their current default of `1`.

Batch-only options that are useful through the merged single-command entrypoint move onto the single command as well:

- `--operator-filter` on `gen-eval`, `convert`, and `optimize`.
- `--post-optimize-command` on `optimize`.
- `--summary-file` on `log-check`.

The handlers remain the ownership boundary for choosing single versus batch behavior. Each handler checks whether `args.concurrency` is `None`; if it is not, it delegates to the existing batch handler path.

## Verification

Tests should cover:

- Single commands omit `--concurrency` by default.
- Single commands parse explicit concurrency values.
- Single commands dispatch to batch handlers when concurrency is explicit.
- `verify --concurrency` dispatches to `verify-batch` and warns that concurrency is ignored.
- Existing `*-batch` commands continue parsing as before.
