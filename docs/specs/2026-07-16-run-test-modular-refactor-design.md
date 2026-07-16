# Run-Test Modular Refactor Design

## Summary

Refactor the `ascend-npu-run-eval` skill's run-test implementation into
single-purpose script modules. The refactor separates command orchestration,
test contracts, local execution, and remote execution while preserving the
three public run-test commands, their output and error behavior, and a clearly
named `run_test_api.py` bridge consumed by `helix.eval.runners`.

## Current State And Problem

`skills/common/ascend-npu-run-eval/scripts/test_runner.py` currently owns all
of the following:

- local subprocess worker argument parsing and result-file protocol;
- standalone and declarative differential test execution;
- test metadata parsing, dynamic module loading, case validation, and payload
  serialization;
- local Torch/NPU initialization and warning filtering;
- remote workspace setup, file transfer, generated remote Python programs,
  archive copying, and remote result comparison.

At the same time, `scripts/cli.py` owns the parser plus the complete
`run-test-baseline`, `run-test-convert`, and `run-test-optimize` workflows,
including mode/reference validation, case-id handling, comparison, cleanup,
and optimize round timing. These unrelated responsibilities make either file
difficult to read, test, and evolve safely.

## Goals And Non-Goals

### Goals

- Give each run-test concern one clear owner.
- Keep all existing command names, options, defaults, result text, errors, and
  exit codes unchanged.
- Expose the skill-to-Helix bridge through `run_test_api.py`, rather than a
  generically named runner module.
- Keep local and remote differential semantics identical to current behavior.
- Move tests with the owning behavior so unit tests do not depend on a large
  aggregate module.

### Non-Goals

- Do not redesign run-test contracts, result payload formats, generated test
  hooks, comparison policy, or remote transport.
- Do not change skill staging configuration. The skill directory is staged as
  a whole, so new sibling scripts are included automatically.
- Do not refactor run-bench, profile-bench, or unrelated CLI commands beyond
  extracting helpers that are already shared with run-test.
- Do not create a package-level abstraction intended for reuse outside this
  skill.

## Target Module Boundaries

All modules remain flat siblings in
`skills/common/ascend-npu-run-eval/scripts/` so they work when the staged skill
directory is placed on `sys.path`.

| Module | Responsibility |
| --- | --- |
| `test_contract.py` | Test metadata parsing, dynamic module loading, Torch/NPU bootstrap helpers, differential case normalization and selection, `compute-kind` interpretation, and payload serialization/deserialization. |
| `run_local_api.py` | Parent-process API for local worker invocation, result-file recovery, single-case payload execution, and local warning filtering. |
| `run_remote_api.py` | Remote workspace lifecycle, transfers, remote worker invocation, archive copying, serialized payload extraction, and remote differential comparison. |
| `run_test_local_worker.py` | Fixed local executable: loads test/operator hooks and runs standalone or differential work in the isolated local process. |
| `run_test_result.py` | Shared differential archive naming and warning filtering for local and remote APIs. |
| `run_test_remote_worker.py` | Fixed remote executable: loads test/operator hooks and runs standalone or differential work after it has been copied to the remote workspace. |
| `run_test_command.py` | The `run-test-*` command flow: arguments, path/mode/reference validation, local/remote selection, case-id flow, rendering, comparison, cleanup, and timing event calls. |
| `run_test_api.py` | Stable API boundary from `helix` into the skill. It re-exports the local and remote API functions. |
| `cli.py` | Top-level parser and dispatch for all commands. It delegates the three run-test subcommands to `run_test_command.py`; it retains non-run-test command handling. |

The existing optimize timing and PT-result cleanup helpers are shared by
run-test and run-bench. Move them from `cli.py` to one small neutral helper
module, for example `execution_lifecycle.py`, rather than duplicating them in
the command modules. `cli.py` and `run_test_command.py` import that helper.

## Interfaces And Compatibility

### Command-line interface

The following commands and all current options remain unchanged:

- `run-test-baseline`
- `run-test-convert`
- `run-test-optimize`

This includes `--test-file`, `--operator-file`, reference aliases,
`--case-id`, remote options, `--keep-remote-workdir`, `--verbose`,
`--test-mode`, and `--accuracy-mode`. Parser errors, output ordering, archive
messages, cleanup behavior, and return codes are regression-compatible.

`cli.py` continues to build these parsers. `run_test_command.py` supplies the
argument-registration helper and dispatch handler so parser ownership remains
centralized without keeping workflow implementation in the top-level CLI.

### Python bridge interface

`run_test_api.py` is the only bridge module that `helix` imports. It exports
these public functions with the existing signatures and return values:

- `parse_test_metadata`
- `run_local_test`
- `run_remote_test`
- `run_local_test_case_payload`
- `run_remote_test_case_payload`
- `run_remote_differential_comparison`

`run_test_api.py` does not provide an executable worker entrypoint. The local
API invokes `run_test_local_worker.py` with the `local-test-worker` or
`local-test-payload-worker` subcommand. The facade delegates each public
function to the owning APIs. Therefore
`src/helix/eval/runners.py` changes only the loaded skill-module name from
`test_runner` to `run_test_api`; its own public wrappers, command handlers, and
verification flows retain their existing interfaces.

`test_runner.py` is removed as part of the completed refactor. During an
incremental implementation it may temporarily re-export `run_test_api.py`, but
it is not a supported long-term API and must not receive new behavior.

### Fixed remote worker

Remote execution must not construct the standalone or differential Python
program as a large `python3 -c` string. Instead, `run_remote_api.py` copies
the versioned `run_test_remote_worker.py` script and its explicit runtime
dependencies into the remote workspace, then invokes it with normal arguments:

```text
python3 run_test_remote_worker.py \
  --test-file <remote-test> \
  --operator-file <remote-operator> \
  --test-mode <standalone|differential> \
  [--case-id <id>] [--no-archive] [--emit-serialized-payload]
```

The worker imports contract/runtime modules only from the copied workspace, not
from the local skill path or an assumed remote installation. It writes a normal
archive for whole differential runs and emits the existing delimited serialized
payload for single-case execution. This preserves remote output and result
transport semantics while making the remote implementation reviewable,
type-checkable, and directly testable as source code.

## Runtime Data Flow

### Ordinary local execution

```text
run-test-* CLI arguments
  -> run_test_command validates paths, mode, and reference policy
  -> run_local_api starts run_test_local_worker.py
  -> test_contract loads test/operator hooks and validates differential cases
  -> run_local_api returns result plus optional archive
  -> run_test_command renders, compares, cleans up, records timing, and returns an exit code
```

### Ordinary remote execution

```text
run-test-* CLI arguments
  -> run_test_command validates remote/reference constraints
  -> run_remote_api creates workspace and transfers runtime/test/operator files
  -> run_test_remote_worker.py loads hooks and executes the requested test mode
  -> run_remote_api filters output and optionally copies archive back
  -> run_test_command renders, compares, records timing, and returns an exit code
```

### Differential and single-case behavior

- A differential whole-test run executes every normalized case (or its selected
  `--case-id`), writes the existing `<operator>_result.pt` archive when
  requested, and compares it against a resolved reference archive if present.
- A `--case-id` run obtains or derives only the matching reference payload, then
  obtains the candidate payload without creating an archive. The command
  compares payload objects and preserves the current missing-payload failure.
- Remote differential validation with a reference operator continues to execute
  both reference and candidate entirely on the remote host and compares their
  archives there; it does not transfer PT archives locally.

## Ownership Rules

- `test_contract.py` must not create subprocesses, transfer files, or decide
  command-level reference policy.
- `run_local_api.py` owns local worker transport and result recovery; it must
  not execute user test hooks or parse worker arguments.
- `run_test_local_worker.py` owns local standalone/differential execution and
  the worker-only argument parser; it must not perform reference comparisons.
- `run_remote_api.py` owns remote transport and result recovery, not test
  implementation source generation.
- `run_test_remote_worker.py` owns remote standalone/differential execution;
  its inputs are explicit files and arguments, never interpolated source code.
- `run_test_command.py` owns user-facing validation, result rendering,
  reference resolution, comparison, cleanup, and return-code selection.
- The lifecycle helper owns optimize timing and PT cleanup because run-bench
  uses the same behavior. It must contain no run-test-specific dispatch.
- `run_test_api.py` is the stable Helix-facing bridge. New implementation code
  belongs in an owning module, not the API layer.

## Behavior Preservation Matrix

| Scenario | Required preserved behavior |
| --- | --- |
| Standalone local | Runs the standalone hook in the isolated worker; never produces a differential archive. |
| Differential local | Runs normalized cases, preserves case selection, and produces the existing archive for whole-test success. |
| Differential with `--case-id` | Executes/compares only the selected payload and does not create a candidate archive. |
| Reference result | Reuses an existing archive or selected archived case; does not re-run the reference operator. |
| Reference operator | Produces the reference archive/payload through the same local or remote mechanism before candidate comparison. |
| Remote standalone | Transfers required files, streams output, and honors remote workspace cleanup/retention. |
| Remote differential | Performs remote-only reference/candidate archive comparison when required; preserves `--ref-operator-file` constraint. |
| `run-test-optimize` archive | Applies the existing `HELIX_OPTIMIZE_DELETE_PT_FILES=run-test` cleanup policy only after the normal result flow. |
| Active optimize round | Appends matching start/end timing events with current command, paths, and final return code. |
| `--keep-remote-workdir` | Retains the remote workspace and prints its path using existing conditions. |

## Test Organization

Split behavior-focused tests while preserving the full regression surface:

- `tests/test_test_contract.py`: metadata, module loading, differential case
  validation/selection, compute flag, and payload serialization.
- `tests/test_run_local_api.py`: worker invocation, result recovery, warning
  filtering, and single-case payload handling.
- `tests/test_run_test_local_worker.py`: worker protocol, standalone/differential
  local execution, NPU bootstrap, result filtering, and single-case payloads.
- `tests/test_run_remote_api.py`: workspace cleanup, transfer inputs,
  fixed worker invocation, archive copy, payload extraction, and remote
  comparison.
- `tests/test_run_test_remote_worker.py`: direct standalone/differential remote
  worker behavior, arguments, archive generation, and serialized payload
  output.
- `tests/test_run_test_api.py`: Helix-facing API exports, delegation, and local
  worker executable compatibility.
- `tests/test_run_test_command.py`: parser registration and command
  orchestration for references, modes, case IDs, cleanup, timing, output, and
  return codes.

Move the run-test-specific cases out of `tests/test_skill_command_script.py`.
That file retains coverage for compare-perf, run-bench, profile commands, and
other script CLI behavior. Tests must patch the owning module rather than the
compatibility facade where practical; bridge tests continue to verify that the
facade exports and delegates the stable public API.

## Migration And Rollback

1. Add the pure contract module and move contract tests.
2. Extract the local worker and parent API, then verify both direct bridge
   calls and `python run_test_local_worker.py` worker invocations.
3. Add the fixed remote worker, change remote transport to copy and invoke it,
   then verify transfer, cleanup, archive, payload, and remote-only comparison
   paths with mocks and direct worker tests.
4. Extract `run_test_api.py`, update `helix.eval.runners` to load it, and
   retain a temporary `test_runner.py` re-export only during migration if a
   staged compatibility check requires it.
5. Extract run-test command orchestration and shared lifecycle helpers; leave
   `cli.py` as parser/dispatcher and verify all three subcommands.
6. Remove the temporary `test_runner.py` compatibility module and obsolete
   dynamic remote-source builders only after focused and full regression suites
   pass.

Each stage is independently reversible by restoring the old delegation or
remote invocation without changing command behavior. Do not combine this
structural change with output, contract, staging, or transport-protocol changes.

## Verification

Run focused tests after each migration stage, then run the complete relevant
suite:

```bash
uv run python -m pytest -q --tb=short --no-header -p no:warnings \
  tests/test_test_contract.py \
  tests/test_run_local_api.py \
  tests/test_run_test_local_worker.py \
  tests/test_run_remote_api.py \
  tests/test_run_test_remote_worker.py \
  tests/test_run_test_api.py \
  tests/test_run_test_command.py \
  tests/test_execution_commands.py \
  tests/test_run_eval_mcp_server.py \
  tests/test_run_eval_mcp_server_tool_metadata.py

bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/test_contract.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/run_local_api.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/run_test_local_worker.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/run_remote_api.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/run_test_remote_worker.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/run_test_api.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/run_test_command.py
bash scripts/run-skill-script-pyright.sh \
  skills/common/ascend-npu-run-eval/scripts/cli.py

uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

The exact new test filenames are created by the implementation. During the
first migration step, run the existing `tests/test_test_runner.py` and
`tests/test_skill_command_script.py` alongside the new focused files until the
test move is complete.
