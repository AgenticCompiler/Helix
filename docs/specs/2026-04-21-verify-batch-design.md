# `verify-batch` Design

## Goal

Add a batch verification command for optimize workspaces while keeping `optimize-status` read-only.

The new flow should let users run verification across a root directory of operator workspaces, reuse the latest verification result by default, optionally force a rerun, and expose a compact verification signal in `optimize-status`.

## User-Facing Changes

### New command

Add a new CLI command:

```bash
uv run triton-agent verify-batch -i operators_root
uv run triton-agent verify-batch -i operators_root --force-verify
```

Behavior:

1. Accept a root directory that contains operator workspaces.
2. Scan each child directory and decide whether it is verifiable.
3. For each verifiable workspace:
   - reuse the latest verify result when one already exists
   - rerun verification when `--force-verify` is supplied
4. Skip non-verifiable workspaces without aborting the whole batch run.
5. Return a non-zero exit code when any workspace verification run fails.

### Remote execution

`verify-batch` should accept the same remote execution flags as `verify`:

- `--remote`
- `--remote-workdir`
- `--keep-remote-workdir`
- `--verbose`

These flags apply uniformly to every workspace in the batch run. A single invocation does not support per-workspace remote overrides.

### `optimize-status` display

Keep `optimize-status` read-only. It must not run verification.

Add verification columns to markdown output:

- show `Verified` only when the latest verification result is complete and successful
- show `Verified Geomean speedup` and `Verified Total speedup` from the latest verified speedup summary
- leave verified speedup cells blank otherwise

Text output may include the latest verify state path for diagnostics, but markdown should stay compact.

## Reuse And Rerun Semantics

`verify-batch` should use this policy:

- default: reuse the latest verify result if one exists for the workspace
- `--force-verify`: always rerun verification and generate a fresh verify directory

This keeps the batch command automation-friendly without forcing a heavy rerun on every invocation.

## Verifiable Workspace Rules

Do not create a separate set of batch-only validation rules.

Reuse the existing `prepare_verify_target()` contract from `src/triton_agent/verification/core.py`. A workspace is verifiable only when the existing single-workspace verify path can prepare successfully, which means the workspace has enough data to run the same verification flow:

- baseline metadata and artifacts
- a numeric best round
- the best round operator artifact
- test and benchmark harnesses

When preparation raises an error, treat the workspace as non-verifiable and continue scanning the rest of the root.

## Latest Verify Result

For each workspace, read verification artifacts from:

```text
opt-verify/verify-*/verify-state.json
```

The latest result should be selected by verify directory name ordering, not filesystem modification time. The directory naming scheme already encodes creation time and keeps selection deterministic:

- `verify-YYYYMMDD-HHMMSS`
- `verify-YYYYMMDD-HHMMSS-2`
- `verify-YYYYMMDD-HHMMSS-3`

## Verified Semantics

The `Verified` marker in `optimize-status` should only be shown when the latest verify result is a full, successful verification run.

That means the latest `verify-state.json` must contain:

- `verify-result.test.status == "passed"`
- `verify-result.rerun_baseline_bench.status == "passed"`
- `verify-result.rerun_best_bench.status == "passed"`
- `verify-result.compare_perf.status == "passed"`

This intentionally excludes partial results such as test-only or bench-only runs.

## Data Model Changes

Extend `OptimizeStatusWorkspace` with read-only verification metadata so rendering can stay simple:

- `latest_verify_state: Path | None`
- `verified: bool`

These fields are derived from the latest verify result on disk. `optimize-status` should not need to understand all verify details beyond:

- where the latest state file is
- whether that latest result qualifies as `Verified`

## Module Boundaries

Keep execution behavior separate from status reporting.

### CLI and command entrypoints

- `src/triton_agent/cli.py`
  - register `verify-batch`
  - add `--force-verify`
- `src/triton_agent/commands/verification.py`
  - add `handle_verify_batch`

### Batch verification logic

Add a feature-local module:

- `src/triton_agent/verification/batch.py`

Responsibilities:

- scan child workspaces under a root
- resolve latest verify state for each workspace
- decide reuse vs rerun
- invoke single-workspace verify when needed
- collect per-workspace outcomes
- compute the batch command exit code

### Verify state discovery

Add small helpers for:

- finding the latest `verify-state.json`
- parsing whether the latest result qualifies as `Verified`

This logic may live in `optimize/status.py` or a narrow new helper module if that keeps `status.py` focused.

## Failure Handling

### Batch command

Do not abort the entire batch run when one workspace fails.

For each workspace:

- non-verifiable workspace: skip it
- verification rerun failed: record the failure and continue
- latest verify exists but is incomplete or failed: reuse it when not forcing, but do not mark the workspace as `Verified`

The batch command should return:

- `0` when every verification action that ran succeeded
- non-zero when any rerun verification fails

### Status rendering

`optimize-status` must never fail just because the latest verify result is malformed or incomplete. In that case:

- keep `latest_verify_state` when the file exists
- set `verified = False`

## Rendering

### Markdown

Add verification columns with compact values:

- `Verified`
- `-`
- verified speedup values formatted as `Nx`

Do not include verify paths or detailed consistency information in markdown output.

### Text

Text output may include the latest verify state path when present. This keeps diagnostics available without widening the markdown table.

## Testing

Add or update tests for:

1. CLI parser
   - `verify-batch` maps to a command kind
   - `--force-verify` parses correctly
   - batch verify accepts the same remote flags as single-workspace verify

2. Batch verify behavior
   - reuse latest verify result by default
   - rerun when `--force-verify` is present
   - propagate shared remote options to every rerun verification
   - skip non-verifiable workspaces
   - continue after one workspace fails
   - return non-zero when any rerun verification fails

3. Latest verify discovery
   - choose the newest verify directory by name
   - mark `verified = True` only for full successful runs
   - keep `verified = False` for partial or failed runs

4. `optimize-status` rendering
   - markdown includes `Verified`
   - successful latest verify shows `Verified`
   - missing, partial, or failed latest verify shows `-`
