# Gen Eval Batch Design

## Summary

- Add a new `gen-eval-batch` subcommand that scans one root directory for operator workspaces and runs one `gen-eval` workflow per workspace.
- Reuse the existing batch wrapper shape from `optimize-batch`: detect one candidate operator file per immediate child directory, run workspaces concurrently, and print a compact summary at the end.
- Keep `gen-eval-batch` as wrapper orchestration only; each workspace still uses the same `gen-eval` workflow skill, prompt contract, remote propagation, and restricted skill staging.

## Goals

- Let a user point at a directory of operator workspaces and have the CLI repair operators, generate harnesses, and validate them in parallel.
- Keep batch behavior predictable and familiar for users who already understand `optimize-batch`.
- Preserve the single-workspace `gen-eval` semantics for each workspace, including direct edits to the original operator file and remote-aware validation.

## Non-Goals

- Do not add a separate batch-specific skill.
- Do not teach the CLI to orchestrate test generation and benchmark generation independently of `gen-eval`.
- Do not add `--interact` or `--output` to the batch command.
- Do not change the single-workspace `gen-eval` prompt contract.

## CLI Contract

- Add `gen-eval-batch` as a new agent-backed batch command.
- `gen-eval-batch` accepts:
  - `--input/-i`
  - `--agent`
  - `--remote`
  - `--remote-workdir`
  - `--test-mode {standalone,differential}`
  - `--bench-mode {standalone,msprof}`
  - `--max-concurrency <N>`
  - `--show-output`
  - `--verbose`
- `gen-eval-batch` does not support:
  - `--output`
  - `--interact`
- `--test-mode` defaults to `differential`.
- `--bench-mode` defaults to `standalone`.
- `--max-concurrency` must be at least 1 and should default to a small explicit value such as `2`.

## Workspace Discovery

- Treat each immediate child directory under the input root as one operator workspace candidate.
- In each workspace, select the only remaining `.py` file after excluding generated or non-entrypoint files:
  - `test_*.py`
  - `differential_test_*.py`
  - `bench_*.py`
  - `opt_*.py`
  - `__init__.py`
- If a workspace has zero candidates, record a workspace failure and continue.
- If a workspace has multiple candidates, record a workspace failure and continue.

## User-Visible Semantics

- `gen-eval-batch` runs one logical `gen-eval` request per detected workspace.
- Each workspace may repair its original operator file directly.
- Batch execution continues after workspace-local failures and returns a non-zero exit code if any workspace failed.
- `--show-output` should stream live output with `[workspace-name] ` prefixes so users can attribute interleaved agent output.

## Result Rendering

- Print one compact line per workspace:
  - success: `[OK] <workspace>: generated eval artifacts for <operator>.py`
  - failure: `[FAIL] <workspace>: <message>`
- Print a final summary line:
  - `Summary: <N> succeeded, <M> failed`
- Failure summaries should prefer the last non-blank stderr line, then the last non-blank stdout line, then a return-code fallback such as `gen-eval exited with return code 7`.

## Implementation Shape

- Add `CommandKind.GEN_EVAL_BATCH`.
- Add a generation-batch command handler that validates the input root path and dispatches to a batch runtime helper.
- Implement the batch runtime in a dedicated module that:
  - discovers workspaces
  - resolves candidate operator files
  - builds one `gen-eval` request per workspace
  - runs requests concurrently
  - renders the batch summary
- Extend the parser alias normalization to accept `gen_eval_batch`.
- Reuse the existing `GenerationOptions`, `build_generation_request`, and `run_generation_request` logic instead of inventing a second single-workspace execution path.

## Error Handling

- If the batch root does not exist, fail with the usual short parser error.
- If the batch root is not a directory, fail with the usual short parser error.
- If there are no immediate child directories, print `No operator workspaces found under <root>` to stderr and return `1`.
- If `--max-concurrency` is less than 1, fail with a short actionable parser error.

## Testing

- Parser coverage for `gen-eval-batch`, its defaults, aliases, and supported flags.
- Batch helper coverage for candidate-file filtering and failure summarization.
- CLI batch coverage for:
  - operator auto-detection
  - workspace-selection failures
  - concurrency limit behavior
  - prefixed streaming output
  - invalid concurrency rejection
- Full verification with `ruff`, `pyright`, and `unittest`.
