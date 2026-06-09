# Unified Bench Case Contract Design

## Summary

Unify generated `torch-npu-profiler` and `msprof` benchmark files so they share the same import-only Python module structure and differ only in runner-owned profiling behavior. Keep `# bench-mode:` metadata as the default execution-strategy hint, but stop using it to select between two different generated file shapes. Remove the old `msprof` self-executing benchmark CLI contract, switch case selection to stable `case_id` values, and clean up stale standalone-only runtime naming and helper paths that are no longer justified once both modes consume the same benchmark contract.

## Goals

- Make generated `torch-npu-profiler` and `msprof` benchmark files structurally identical.
- Keep `# bench-mode: torch-npu-profiler|msprof` as the benchmark's default execution-mode hint.
- Move benchmark case selection to stable string `case_id` values across `run-bench`, `profile-bench`, and IR capture.
- Make both profiling modes consume the same runtime helper and the same benchmark case declarations.
- Remove obsolete code paths, naming, docs, and tests that only exist to support the old `msprof` benchmark mini-CLI.

## Non-Goals

- Do not preserve backward compatibility for old `msprof` benchmark files that implement `main()`, `--num-bench`, or `--bench <N>`.
- Do not add a new metadata line such as `# bench-contract: ...`.
- Do not change the top-level `run-bench` or `profile-bench` command names.
- Do not redesign perf artifact format, kernel aggregation rules, or total-op fallback semantics beyond replacing numeric case labels with real `case_id` labels.
- Do not infer benchmark cases from module globals, function naming conventions, or profiler output.

## Decision

### Unified benchmark module contract

- Generated benchmark files remain named `bench_<op>.py` and live beside the operator file.
- Generated benchmark files remain import-only Python modules.
- Generated benchmark files must keep this metadata header near the top of the file:
  - `# bench-mode: torch-npu-profiler|msprof`
  - `# api-name: <resolved_entrypoint>`
  - `# api-kind: <triton-wrapper|torch-function|torch-module>`
  - `# kernels: <resolved_kernel_names>`
- `# bench-mode:` continues to declare the benchmark's default execution strategy when the user does not pass an explicit mode override.
- `# bench-mode:` no longer selects between two different generated file structures.
- No new metadata key is added for contract versioning in this change.

Generated benchmark files must export exactly these three hooks:

- `build_operator_api(operator_module)`
- `build_bench_cases()`
- `build_bench_case_fn(operator_api, case)`

### `build_operator_api(operator_module)`

- `build_operator_api(operator_module)` is required.
- It receives the already-imported runtime operator module loaded from `--operator-file`.
- It must return the final callable object that benchmark execution should invoke.
- The runner must treat the returned object as already prepared for execution.
- The runner must not add extra `.npu()`, `.eval()`, wrapper, or constructor logic after this hook returns.
- This hook remains the benchmark's place to express operator-specific construction logic such as:
  - `torch.nn.Module` instantiation
  - constructor arguments
  - wrapping a module method into a callable
  - device placement
  - evaluation-mode selection

### `build_bench_cases()`

- `build_bench_cases()` is required.
- It returns the full benchmark case list in stable execution order.
- Each case must be a mapping with:
  - `id`: required, non-empty string, unique within the benchmark file
  - optional shared execution hints such as `warmup`, `repeats`, and `seed`
  - additional operator-specific shape, dtype, layout, and attribute fields as needed by the benchmark
- `build_bench_cases()` must be a cheap declaration step:
  - do not allocate NPU tensors
  - do not execute the operator
  - do not depend on single-use side effects
- Case declarations should be deterministic and repeatable across local, remote, parallel, and IR-capture workflows.
- Case coverage policy remains shared across both modes:
  - total case count must be `<= 20`
  - when the operator's shape space is broad enough, prefer `8-20` representative cases
  - cover small, medium, and large representative shapes unless the valid input space is genuinely narrow

### `build_bench_case_fn(operator_api, case)`

- `build_bench_case_fn(operator_api, case)` is required.
- It receives the final callable returned by `build_operator_api(...)` plus one declared case mapping.
- It must return a zero-argument callable that performs the measured operator execution for that case.
- This hook is where benchmark-specific input construction belongs:
  - tensor allocation
  - deterministic random seeding
  - attribute binding
  - closure creation
- Setup should happen outside the returned callable whenever practical so the measured callable contains only the benchmarked operator body.
- If randomized inputs are used, seeding must remain explicit and deterministic for the selected case.

### Unified `bench-mode` semantics

- `torch-npu-profiler` and `msprof` now mean "which profiling strategy the runner should apply to the same benchmark contract."
- They no longer mean "which benchmark-file structure the generator should emit."
- `run-bench`, `profile-bench`, and IR capture continue to read `# bench-mode:` by default when the user does not pass an explicit mode override.
- Explicit CLI mode overrides remain supported.

### Unified case-selection semantics

- `run-bench` loads the benchmark module, resolves all cases from `build_bench_cases()`, and runs every declared case in order.
- `profile-bench` selects one case by `--case-id`.
- IR capture selects one case by `--case-id`.
- The numeric `--bench <N>` case selector is removed from benchmark profiling and IR capture.
- If `--case-id` is omitted:
  - when exactly one case exists, select it automatically
  - when multiple cases exist, fail explicitly and list the available case ids
- Case labels in perf artifacts and failure messages must use the declared `case_id`, not synthetic `case-1`, `case-2`, or other positional aliases.

## Runtime And Runner Design

### Common runtime helper

- Replace `standalone_bench_runtime.py` with a mode-neutral `bench_runtime.py`.
- Rename standalone-specific runtime types and helper entrypoints to neutral `bench_*` names.
- The shared runtime helper owns:
  - loading benchmark metadata
  - importing the benchmark module
  - importing the runtime operator module
  - calling `build_operator_api(...)`
  - calling `build_bench_cases()`
  - validating case declarations
  - selecting a case by `case_id`
  - calling `build_bench_case_fn(operator_api, case)`

The runtime helper may keep an internal helper CLI for repository-owned callers only:

- `list-cases`
- `run-one`
- `profile-one`

The benchmark file itself must not define its own CLI entrypoint.

### `run-bench`

- Public `run-bench --bench-file ... --operator-file ...` entry shape stays unchanged.
- `run-bench` no longer executes `python bench_<op>.py --num-bench`.
- `run-bench` no longer executes `python bench_<op>.py --bench <N>`.
- For both benchmark modes, `run-bench` first resolves the case list through the shared runtime helper.
- For `torch-npu-profiler`, the runner profiles each selected case with centralized `torch_npu.profiler` logic, as it already does conceptually today.
- For `msprof`, the runner wraps the shared runtime helper case execution:
  - `msprof python3 bench_runtime.py run-one --bench-file ... --operator-file ... --case-id ...`
- Local, remote, serial, and parallel `run-bench` flows all consume the same case ids and the same benchmark case declarations.

### `profile-bench`

- Public `profile-bench --bench-file ... --operator-file ...` entry shape stays unchanged except for case selection.
- `profile-bench` keeps optional `--bench-mode` override support.
- `profile-bench` removes `--bench`.
- `profile-bench` keeps `--case-id`.
- `profile-bench` uses `--case-id` for both `torch-npu-profiler` and `msprof`.
- The runtime helper resolves the case id before profiling begins.
- For `torch-npu-profiler`, the helper profiles exactly one selected case with `torch_npu.profiler`.
- For `msprof`, the helper profiles exactly one selected case by wrapping the shared runtime helper `run-one` path in `msprof`.

### IR capture

- `skills/triton-npu-analyze-ir/scripts/capture_ir.py` removes `--bench`.
- IR capture adds or keeps only `--case-id` as the benchmark case selector.
- Both modes build execution commands around the shared runtime helper:
  - `python3 bench_runtime.py run-one --bench-file ... --operator-file ... --case-id ...`
  - `msprof python3 bench_runtime.py run-one --bench-file ... --operator-file ... --case-id ...`
- IR capture no longer needs separate "standalone helper vs benchmark-file CLI" command rendering logic.

## Obsolete Code Cleanup

This change must remove old paths instead of preserving dual behavior.

### Remove obsolete generated-benchmark expectations

- Delete the `msprof` benchmark contract requirement that generated benchmark files define:
  - `main()`
  - `argparse`
  - `--num-bench`
  - `--bench <N>`
- Delete docs and tests that still describe or require benchmark-file self-execution for `msprof`.
- Delete docs and tests that still describe `standalone` benchmark files as a different structural contract from `msprof`.

### Remove obsolete runtime naming

- Rename standalone-only runtime helpers, support-path helpers, and protocol names to neutral benchmark-runtime names once both modes share the same runtime contract.
- Remove duplicate code that only exists because `msprof` currently discovers cases through benchmark-file CLI execution while `standalone` discovers cases through imported hooks.

### Remove obsolete CLI and MCP surface

- Remove `--bench` from:
  - `skills/triton-npu-run-eval/scripts/run-command.py profile-bench`
  - `skills/triton-npu-analyze-ir/scripts/capture_ir.py`
  - MCP tool schemas and docs that expose numeric benchmark selection
- Remove helper code whose only job is parsing benchmark-file `--num-bench` output.

### Remove obsolete tests and fixtures

- Delete or rewrite fixtures that create fake `msprof` benchmark files containing only metadata plus a script-style CLI body.
- Rewrite tests so both modes use the same import-only benchmark module fixtures and the same `case_id`-based selection model.

## Error Handling

- Fail explicitly before execution when:
  - the benchmark module is missing `build_operator_api`
  - the benchmark module is missing `build_bench_cases`
  - the benchmark module is missing `build_bench_case_fn`
  - any required hook is not callable
  - `build_bench_cases()` returns no cases
  - any case is missing `id`
  - any case id is empty or duplicated
  - `build_bench_case_fn(operator_api, case)` does not return a callable
  - combined kernel resolution from metadata plus runtime operator discovery fails
- If the user provides an unknown `--case-id`, fail explicitly and list available case ids.
- If the user omits `--case-id` and more than one case exists, fail explicitly instead of silently choosing the first case.
- During `run-bench`, case-level failures remain best-effort:
  - one failed case must not stop later cases from running
  - the perf file must still be written when at least one case was attempted
  - the overall command return code must be non-zero if any case fails
- For `msprof`, when resolved kernels do not match profiler rows:
  - keep existing `NA` plus error-comment semantics
  - use the real `case_id` as the label
- Missing `# kernels:` remains an explicit contract failure. The runtime must not guess around it.

## Migration Boundary

- This change is intentionally not backward compatible with old `msprof` benchmark files.
- Repository code should switch directly to the unified import-only benchmark contract.
- Existing old-style `msprof` benchmark files must be regenerated before use.
- The repository should not keep a compatibility adapter for old benchmark-file CLI behavior.

## Scope Of Required Follow-Up Changes

- Update `skills/triton-npu-gen-bench/SKILL.md`.
- Rewrite:
  - `skills/triton-npu-gen-bench/references/bench-standalone-spec.md`
  - `skills/triton-npu-gen-bench/references/bench-msprof-spec.md`
- Update:
  - `skills/triton-npu-run-eval/SKILL.md`
  - `skills/triton-npu-run-eval/references/run-bench.md`
  - `skills/triton-npu-run-eval/references/profile-bench.md`
  - `skills/triton-npu-run-eval-mcp/references/run-bench.md`
  - `skills/triton-npu-run-eval-mcp/references/profile-bench.md`
  - `skills/triton-npu-profile-operator/SKILL.md`
- Replace `standalone_bench_runtime.py` with `bench_runtime.py` and update support-path staging everywhere that copies runtime helper files.
- Update `bench_runner.py`, `bench_runner_standalone.py`, `bench_runner_msprof.py`, `profile_runner.py`, `bench_runner_deps.py`, and `capture_ir.py` around the unified runtime helper and `case_id` case selection.
- Update CLI and MCP parsing layers that currently expose numeric benchmark selectors.

## Verification

- Add contract tests that verify:
  - standalone and msprof benchmark specs now describe the same import-only benchmark structure
  - the benchmark skill no longer requires `msprof` benchmark-file CLI handling
  - `# bench-mode:` remains required metadata
- Add runtime tests that verify:
  - benchmark modules load through the shared runtime helper
  - missing required hooks fail explicitly
  - duplicate or empty case ids fail explicitly
  - `build_bench_case_fn(...)` must return a callable
  - case selection by `case_id` works and error messages list available ids
- Add runner tests that verify:
  - `run-bench` in both modes resolves cases through the shared runtime helper
  - `msprof` wraps `bench_runtime.py run-one --case-id ...` instead of calling benchmark-file CLI arguments
  - local and remote `profile-bench` use `--case-id`
  - parallel local and remote execution still preserves stable case ordering by declared `case_id`
- Add IR capture tests that verify:
  - `capture_ir.py` accepts `--case-id`
  - both modes render runtime-helper execution commands
  - numeric `--bench` is no longer accepted
- Add CLI and MCP tests that verify:
  - `profile-bench` no longer accepts numeric benchmark selectors
  - tool metadata exposes `case_id` and no longer exposes numeric `bench`
- Run repository verification relevant to touched code, including strict file-scoped `pyright` for modified `skills/*/scripts/` files.
