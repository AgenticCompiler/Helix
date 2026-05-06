# Standalone Bench Profiler Design

## Summary

Redesign `standalone` benchmark execution so the benchmark file becomes an import-only Python module with explicit hooks, while `bench_runner.py` centrally owns profiling, kernel aggregation, and perf artifact generation. Keep the public `run-bench --bench-file ... --operator-file ...` entry shape, keep `profile-bench --bench-mode standalone` available, and make standalone perf artifacts speak the same language as `msprof` perf artifacts.

## Goals

- Keep `standalone` benchmark invocation anchored on `--bench-file` plus `--operator-file`.
- Replace standalone self-timing benchmark scripts with runner-owned `torch_npu.profiler` execution.
- Define a stable standalone benchmark-module contract that supports custom operator construction and custom case setup without pushing operator-specific parsing into the runner.
- Persist standalone perf data in the same comment-augmented `latency-*` format already used by `msprof`, including raw per-op payloads for total-op fallback.
- Preserve `profile-bench --bench-mode standalone` as a supported workflow.

## Non-Goals

- Do not keep or extend the old direct-execution standalone contract where `bench_<op>.py` is run as a script and prints `latency-*` lines itself.
- Do not add a `perf_counter()` fallback path for standalone profiling.
- Do not add a new top-level benchmark artifact format beyond the existing perf txt plus existing profiler-directory flow.
- Do not change `msprof` benchmark semantics except where shared helper reuse naturally reduces duplication.
- Do not make the CLI understand operator-specific input schemas or benchmark-case construction details.

## Decision

### Standalone benchmark module contract

- `bench_<op>.py` remains the benchmark file name and continues to live beside the operator file.
- The benchmark file must keep the metadata header near the top of the file:
  - `# bench-mode: standalone`
  - `# api-name: <name>`
  - `# api-kind: <triton-wrapper|torch-function|torch-module>`
  - `# kernels: <kernel-a>, <kernel-b>, ...`
- The metadata header remains the benchmark's declared kernel source of truth, but runtime execution still unions metadata kernels with kernels discovered from the runtime operator file, using the existing stable-order `metadata + operator` behavior already used by `msprof`.
- A standalone benchmark file is no longer required to be directly executable.
- A standalone benchmark file no longer needs `main()`, `argparse`, `--operator-file`, or stdout timing output.
- A standalone benchmark file must export these two required hooks:
  - `build_operator_api(operator_module)`
  - `build_standalone_bench_cases(operator_api)`

### `build_operator_api(operator_module)`

- `build_operator_api(operator_module)` is required.
- It receives the already-imported runtime operator module loaded from `--operator-file`.
- It must return the final callable object that benchmark cases should invoke.
- The runner must treat the returned object as already prepared for execution.
- The runner must not apply extra `.npu()`, `.cuda()`, `.eval()`, wrapper, or constructor logic after `build_operator_api(...)` returns.
- This hook is where standalone benchmark files express operator-specific construction logic such as:
  - `torch.nn.Module` instantiation
  - constructor arguments
  - device placement
  - evaluation mode
  - wrapping a module method into a callable

### `build_standalone_bench_cases(operator_api)`

- `build_standalone_bench_cases(operator_api)` is required.
- It receives the final callable returned by `build_operator_api(...)`.
- It must return the full case list in stable execution order.
- Each returned case must be a mapping with:
  - `id`: required, non-empty string, unique within the benchmark file
  - `fn`: required, zero-argument callable that performs the measured operator execution
  - `warmup`: optional positive integer
  - `repeats`: optional positive integer
- `warmup` and `repeats` default to the runner's existing standalone defaults when omitted.
- Case setup must happen outside `fn` whenever practical.
  - tensor allocation
  - static parameter binding
  - deterministic case-specific setup
- The measured `fn` should contain only the operator execution body so profiling excludes one-time setup work.
- The runner must not scan module globals or function names to discover cases. It only uses the list returned by `build_standalone_bench_cases(...)`.
- Standalone benchmark cases should follow the same representative-coverage policy as `msprof`:
  - the total case count must be `<= 20`
  - when the operator's shape space is broad enough, prefer `8-20 representative cases`
  - cover small, medium, and large representative shapes unless the valid input space is genuinely narrow

### `run-bench --bench-mode standalone`

- Local and remote standalone `run-bench` stop executing `python bench_<op>.py ...`.
- Instead, the runner:
  1. loads the benchmark module from `--bench-file`
  2. loads the runtime operator module from `--operator-file`
  3. calls `build_operator_api(operator_module)`
  4. calls `build_standalone_bench_cases(operator_api)`
  5. profiles every returned case with centralized `torch_npu.profiler` logic
- The standalone runner uses profiler-derived latency only.
- `perf_counter()` fallback is removed.
- Standalone case execution should be best-effort across the full case list.
  - one failed case must not stop later cases from running
  - the perf file must still be written when at least one case was attempted
  - the overall command return code must be non-zero if any case fails
- Local and remote standalone execution should reuse the same centralized case-profiling helper as much as possible so the two paths stay aligned.

### `profile-bench --bench-mode standalone`

- `profile-bench --bench-mode standalone` remains supported.
- Add a standalone-only selector flag:
  - `--case-id <id>`
- For standalone mode, profiling should target exactly one case.
- `--case-id` is required for standalone profile execution.
- The runner resolves `--case-id` against the case ids returned by `build_standalone_bench_cases(operator_api)`.
- If the id is valid, the runner profiles only that one case and returns the generated profiler directory path.
- `--bench` is not the standalone case selector anymore.
- `--bench` remains the `msprof` case selector.
- `--case-id` is invalid for `msprof` profiling and should fail explicitly to prevent mixed semantics.

## Perf Artifact Contract

### Standalone perf txt

- Standalone perf artifacts continue to be written beside the runtime operator file using the existing `<operator-stem>_perf.txt` path shape.
- The standalone perf txt adopts the same normalized structure already used by `msprof`:
  - `latency-<case-id>: <value|NA>`
  - `# raw-op-statistic-<case-id>: {"ops":[...]}`
  - `# resolved-kernels-case-<case-id>: kernel_a,kernel_b,...`
  - `# kernel-source-case-<case-id>: metadata|operator|metadata+operator`
  - `# latency-error-case-<case-id>: <message>` when needed
- `latency-<case-id>` represents the sum of the resolved Triton kernels' average device-side latency for that case.
- Standalone perf values should use the same microsecond-scale interpretation as `msprof` perf values so the two modes share one comparison vocabulary.

### Raw-op payload normalization

- The standalone runner reads profiler output from `operator_details.csv`.
- It must normalize profiler rows into the same raw-op payload shape already used by `msprof`:

```json
{"ops":[{"op_type":"KernelA","avg_time_us":12.34},{"op_type":"KernelB","avg_time_us":56.78}]}
```

- The normalized payload must include every parsed operator row for the case, not only matched Triton kernels.
- This keeps total-op fallback available to `compare-perf` exactly as it is for `msprof`.

### Latency aggregation rules

- The standalone runner should compute one average device-side time per profiler row name from `operator_details.csv`.
- When profiler output includes a usable per-row count column, only rows that represent the active repeated region should contribute to the averaged result.
- When the count column is missing, the runner should group rows by name and divide summed device self duration by the configured repeat count.
- After row normalization, `latency-<case-id>` is the sum of every normalized row whose operator name exactly matches one of the resolved Triton kernels for that case.
- If some resolved kernels are present and others are missing, sum only the matched subset and still write the full raw-op payload.
- If no resolved kernel matches any normalized profiler row:
  - write `latency-<case-id>: NA`
  - write the raw-op payload
  - write `# latency-error-case-<case-id>: no resolved kernels matched operator_details csv`

## Error Handling

- Fail explicitly before case execution when:
  - the benchmark module is missing `build_operator_api`
  - the benchmark module is missing `build_standalone_bench_cases`
  - either required hook is not callable
  - combined kernel resolution from metadata plus runtime operator discovery fails
  - the returned case list is empty
  - any case is missing `id`
  - any case is missing `fn`
  - any case id is empty or duplicated
  - any `fn` is not callable
  - any `warmup` or `repeats` value is present but is not a positive integer
- If setup fails before any case is available or selected, return failure without writing a perf file.
  - exception while constructing the operator API
  - exception while building the case list
  - invalid standalone `--case-id` selection
- During `run-bench --bench-mode standalone`, case-level failures should be recorded per case and later cases should still run.
- A case-level failure includes:
  - exception while invoking a case `fn`
  - missing profiler output
  - malformed or empty `operator_details.csv`
  - profiler CSV missing the required timing columns
- For a case-level failure after case iteration starts:
  - write `latency-<case-id>: NA`
  - write `# latency-error-case-<case-id>: ...`
  - do not write `# raw-op-statistic-<case-id>` unless a trustworthy normalized payload was produced
- For standalone `profile-bench`:
  - missing `--case-id` is an explicit error
  - unknown `--case-id` is an explicit error that lists available ids
  - `--case-id` with `--bench-mode msprof` is an explicit error
- Temporary profiling directories used by standalone `run-bench` should be cleaned up after parsing.
- Standalone `profile-bench` should preserve and return the selected case's profiler directory, including remote copy-back behavior.

## Scope Of Required Follow-Up Changes

- Update `skills/triton-npu-gen-bench/references/bench-standalone-spec.md` to describe the new import-only standalone hook contract instead of direct executable benchmark scripts.
- Update `skills/triton-npu-gen-bench/SKILL.md` so standalone benchmark generation targets the new two-hook module contract.
- Update `skills/triton-npu-run-eval/SKILL.md` and `README.md` so standalone benchmark and profile examples stop describing direct benchmark-script execution semantics.
- Update `skills/triton-npu-run-eval/scripts/bench_runner.py` to import and execute standalone benchmark hooks.
- Update `skills/triton-npu-run-eval/scripts/profile_runner.py` and `skills/triton-npu-run-eval/scripts/run-command.py` to support standalone `--case-id`.
- Update generation-contract tests and runtime tests that currently assume standalone benchmark files remain directly executable.

## Verification

- Add standalone benchmark runner tests that verify:
  - the runner imports benchmark and operator modules instead of executing the benchmark file as a script
  - missing required hooks fail explicitly
  - duplicate and empty case ids fail explicitly
  - invalid `warmup` and `repeats` fail explicitly
  - one failed standalone case does not stop later cases
  - the perf file is still written for mixed-success case sets
  - successful cases emit `latency-*`, raw-op payload, resolved kernels, and kernel source comments
  - no-match cases emit `NA` plus raw-op payload and the missing-match error comment
- Add standalone profiling tests that verify:
  - `profile-bench --bench-mode standalone --case-id <id>` profiles exactly one case
  - missing or unknown `--case-id` fails explicitly
  - standalone profile execution returns or copies back the profiler directory for the selected case
- Add `compare-perf` coverage to verify standalone perf files participate in the same total-op fallback path as `msprof` perf files.
- Update generation-contract tests so the standalone benchmark spec no longer requires:
  - `main()`
  - direct `--operator-file` parsing inside `bench_<op>.py`
  - `print(f"latency-...")` inside the benchmark file
- Run strict file-scoped `pyright` for `skills/triton-npu-run-eval/scripts/bench_runner.py` and any modified skill scripts under `skills/*/scripts/`.
