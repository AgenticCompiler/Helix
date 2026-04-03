# Batch Optimize Subcommand

## User-visible semantics

- Add a new `optimize-batch` subcommand for scanning one root directory that contains multiple operator workspaces.
- Treat each immediate child directory under the input root as one operator workspace candidate.
- In each workspace, auto-detect the operator input file by selecting the only remaining `.py` file after excluding generated artifacts and common non-entrypoint files.
- Exclude `test_*.py`, `differential_test_*.py`, `bench_*.py`, `opt_*.py`, and `__init__.py` from candidate selection.
- If a workspace has no remaining candidate file or more than one remaining candidate file, record that workspace as a failure with a short actionable message and continue with the rest of the batch.
- Run one logical `optimize` workflow per detected operator workspace and reuse the existing optimize prompt, backend selection, skill staging, optimize guidance, and supervision behavior.
- Support bounded parallelism through `--max-concurrency <N>`.
- Default to finishing the entire batch even when some workspaces fail, then return a non-zero exit code if any workspace failed.
- Print a compact per-workspace summary plus final totals.

## CLI contract

- `optimize-batch` accepts `--input/-i` as the batch root directory.
- `optimize-batch` accepts `--agent`, `--remote`, `--remote-workdir`, `--test-mode`, `--bench-mode`, `--min-rounds`, `--continue`, `--no-agent-session`, and `--verbose`.
- `optimize-batch` does not support `--output`, `--interact`, or `--show-output` because those flags do not compose cleanly across multiple concurrent optimize runs.
- `--max-concurrency` must be at least 1 and defaults to a small explicit value.
- `--continue` keeps the existing single-workspace validation rules for each workspace independently.

## Implementation notes

- Keep candidate discovery and concurrent batch orchestration in the CLI layer because this is wrapper-specific behavior.
- Reuse the existing single-workspace optimize execution path instead of inventing a separate backend flow.
- Keep workspace failures isolated so one bad directory does not block unrelated optimize runs.
