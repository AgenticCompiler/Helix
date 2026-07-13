# Run-Simulator Subcommand Design

## Context

The CLI already has a thin execution layer for local `run-test`, `run-bench`, and benchmark profiling. The current benchmark runtime has been unified under `skills/triton-npu-run-eval/scripts/bench_runtime.py`, which now owns:

- benchmark module loading
- benchmark case construction
- benchmark case selection
- single-case execution through `run-one`

The new `run-simulator` subcommand should reuse that unified runtime instead of adding a new benchmark contract or duplicating case selection logic in `src/`.

This command is a local execution helper only. It does not need to appear in skill-facing command surfaces, does not need MCP exposure, and does not need remote execution support.

## User-Visible Semantics

`run-simulator` executes one selected benchmark case under `msprof op simulator` and streams the simulator process output directly to the terminal.

The command accepts:

- `--bench-file <path>`
- `--operator-file <path>`
- `--case-id <id>` optional
- `--kernel-name <name>` optional

The command does not accept:

- `--bench-mode`
- `--remote`
- `--remote-workdir`
- `--keep-remote-workdir`
- `--output`

The command ignores `# bench-mode:` metadata completely. It uses the unified benchmark runtime for case execution regardless of the benchmark file's default mode marker.

The command does not generate:

- perf artifacts
- profile directories
- comparison summaries

It returns the simulator process exit code unchanged.

## Case Selection

`run-simulator` reuses the existing benchmark case selection contract from `bench_runtime.py`.

- If the benchmark resolves exactly one case, omitting `--case-id` is allowed and selects that case automatically.
- If the benchmark resolves multiple cases, omitting `--case-id` fails with an actionable error that lists the available case ids.
- If `--case-id` is provided but does not exist, the command fails with an actionable error that lists the available case ids.

This keeps the new command aligned with the current `run-one` runtime semantics instead of adding a simulator-only case selector.

## Kernel Selection

`run-simulator` resolves kernels from the existing benchmark kernel resolution helper in `bench_contract.py`.

- If `--kernel-name` is provided, it must exactly match one resolved kernel name.
- If `--kernel-name` is omitted and exactly one kernel resolves, that kernel is selected automatically.
- If `--kernel-name` is omitted and multiple kernels resolve, the command fails with an actionable error that lists the resolved kernel names.
- If benchmark metadata and operator inspection resolve no kernels, the existing kernel resolution error is propagated.

Kernel resolution should continue using the stable union of:

- benchmark metadata kernels from `# kernels:` or `# kernel:`
- `@triton.jit` kernels discovered from the runtime operator file

This preserves the current source-of-truth behavior for benchmark execution and profiler parsing.

## Execution Shape

The target benchmark case command should match the existing unified runtime entrypoint:

```bash
python3 bench_runtime.py run-one --bench-file <bench-file> --operator-file <operator-file> --case-id <case-id>
```

`run-simulator` wraps that case command with:

```bash
msprof op simulator --soc-version=<soc-version> --kernel-name <kernel-name> <case-command>
```

For local execution:

- the working directory is `bench_file.parent`
- `--bench-file` should be passed as `bench_file.name`
- `--operator-file` should be passed as the relative path from `bench_file.parent` to `operator_file`
- `soc-version` should be read from `HELIX_SIMULATOR_SOC_VERSION`, defaulting to `Ascend950PR_9599`
- `TRITON_ALWAYS_COMPILE=1` should be set for the child process
- output should be streamed live through the existing process runner
- the benchmark timeout environment path should be reused instead of introducing a simulator-specific timeout

The command should not parse profiler output, inspect simulator artifacts, or infer any success state beyond the child process result payload.

## Code Structure

### CLI and command routing

Add a new command kind:

- `CommandKind.RUN_SIMULATOR = "run-simulator"`

Register a new execution subcommand in `src/helix/cli.py` with:

- help group `Execution`
- dedicated `input_mode="run-simulator"`
- parser arguments for `--bench-file`, `--operator-file`, optional `--case-id`, and optional `--kernel-name`

The command should not reuse the existing `run-bench` input mode because it has a narrower argument surface and intentionally omits bench mode, output, and remote execution options.

Add a new handler in `src/helix/commands/execution.py`:

- validate and resolve the bench and operator paths
- invoke the simulator runtime wrapper
- return the simulator process exit code

Unlike `run-bench`, the handler should not print `Perf file:` hints or post-process output.

### Runtime wrapper in `src/`

Add a dedicated wrapper in `src/helix/execution.py` that loads a simulator helper module through the existing skill loader bridge.

This wrapper should follow the existing pattern used by:

- `run_local_test`
- `run_local_bench`

The wrapper should normalize the returned result payload into `AgentResult` so the CLI command handler can stay consistent with the rest of the execution layer.

### Skill-side helper

Add a new internal helper module:

- `skills/triton-npu-run-eval/scripts/simulator_runner.py`

This helper owns simulator-specific orchestration:

- load the unified bench runtime module
- resolve the selected case using `load_bench_cases(...)` and `select_bench_case(...)`
- resolve kernels using `resolve_bench_kernel_resolution(...)`
- validate or infer `--kernel-name`
- build the `bench_runtime.py run-one` command
- wrap it in `msprof op simulator`
- execute it through `run_streaming_process(...)`

This helper should be internal only. It should not be wired into:

- `skills/triton-npu-run-eval/scripts/run-command.py`
- `skills/triton-npu-run-eval/SKILL.md`
- MCP tool metadata

That keeps the command as a CLI-only execution surface.

## Error Handling

The command should fail explicitly with short actionable messages for:

- missing bench file
- missing operator file
- unknown benchmark case id
- missing `--case-id` when multiple cases exist
- invalid `--kernel-name`
- omitted `--kernel-name` when multiple kernels resolve
- kernel resolution failure from metadata plus operator inspection

The command should not silently fall back to:

- an arbitrary benchmark case
- an arbitrary kernel from a multi-kernel resolution
- benchmark metadata mode selection

## Testing

Add focused tests for:

- CLI parser coverage for `run-simulator`
- handler coverage in `tests/test_execution_commands.py`
- simulator helper behavior in a dedicated test module such as `tests/test_simulator_runner.py`

Test scenarios should include:

- explicit `--case-id` and explicit `--kernel-name`
- implicit single-case selection
- missing `--case-id` with multiple cases
- implicit single-kernel selection
- missing `--kernel-name` with multiple resolved kernels
- invalid `--kernel-name`
- correct simulator command construction
- relative operator path handling when the operator file is outside the bench file directory but still locally addressable
- direct propagation of the child process return code

Because the new helper lives under `skills/*/scripts/`, completion should also include the file-scoped strict Pyright check required by repository policy.

## Files Expected To Change

- `docs/specs/2026-06-11-run-simulator-subcommand-design.md`
- `src/helix/models.py`
- `src/helix/cli.py`
- `src/helix/commands/execution.py`
- `src/helix/execution.py`
- `skills/triton-npu-run-eval/scripts/simulator_runner.py`
- `tests/test_cli.py`
- `tests/test_execution_commands.py`
- `tests/test_simulator_runner.py`

## Out Of Scope

This change does not:

- add remote simulator execution
- add simulator support to `run-command.py`
- add simulator support to MCP tools
- generate perf artifacts
- generate profile directories
- change benchmark file generation contracts
- change `run-bench` or `profile-bench` semantics
