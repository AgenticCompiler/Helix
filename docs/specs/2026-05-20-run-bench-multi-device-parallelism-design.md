# Run-Bench Multi-Device Parallelism Design

## Summary

Add an opt-in `run-bench --npu-devices ...` mode that schedules benchmark cases across multiple Ascend NPU devices in parallel for both local and remote execution. Keep the existing public `run-bench --bench-file ... --operator-file ...` entrypoint, preserve current serial behavior when the option is omitted, support both `msprof` and `standalone` benchmark modes, and keep perf artifacts stable by aggregating per-case results back into one ordered perf file.

## Goals

- Let `run-bench` execute distinct benchmark cases concurrently on different NPU devices when the user explicitly provides a device list.
- Support the same user-facing option for:
  - local `run-bench`
  - remote `run-bench --remote user@host[:port]`
  - `msprof` benchmark mode
  - `standalone` benchmark mode
- Reuse as much existing benchmark execution logic as practical instead of rewriting the benchmark runners from scratch.
- Keep benchmark artifacts deterministic:
  - stable case ordering in the final perf file
  - best-effort execution when some cases fail
  - unchanged success and failure semantics outside the new opt-in mode
- Keep skill-side helper scripts self-contained without introducing imports from `src/helix`.

## Non-Goals

- Do not change current `run-bench` behavior when `--npu-devices` is not provided.
- Do not add automatic device discovery.
- Do not add a second concurrency knob such as `--max-concurrency` for benchmark cases in this change.
- Do not support scheduling one `run-bench` invocation across multiple remote hosts.
- Do not change the public perf artifact format beyond the existing per-case records and existing comments.
- Do not move benchmark orchestration logic out of the run-eval skill into the top-level CLI.

## Current Problem

Today `run-bench` executes cases serially in both supported benchmark modes:

- local `msprof` iterates `--bench 1..N` in a loop
- remote `msprof` does the same over SSH
- local `standalone` iterates the declared standalone case list in one process
- remote `standalone` runs the same centralized standalone helper remotely in one process

This means a benchmark suite with many cases cannot use a multi-NPU machine efficiently. It also means users cannot explicitly reserve a subset of devices for a benchmark run the way batch commands can already reserve one device per workspace.

## User-Facing Control

Introduce one new optional `run-bench` flag:

- `--npu-devices`

Examples:

```bash
python3 ./scripts/run-command.py run-bench \
  --bench-file bench_abs.py \
  --operator-file opt_abs.py \
  --npu-devices 0,1,2,3

python3 ./scripts/run-command.py run-bench \
  --bench-file bench_abs.py \
  --operator-file opt_abs.py \
  --bench-mode msprof \
  --remote user@host:2222 \
  --remote-workdir /tmp/helix \
  --npu-devices 0-3
```

Semantics:

- Omitted: preserve current serial execution exactly.
- Provided: enable case-level multi-device scheduling.
- The value is parsed as a comma-separated list of device identifiers.
- Whitespace around entries is ignored.
- Numeric ascending ranges such as `0-3` expand to `0,1,2,3`.
- Empty entries are invalid.
- Duplicate devices are invalid.
- Device identifiers remain opaque strings after parsing so the contract can support plain numeric Ascend device ids without over-constraining future runtime needs.

Case concurrency is derived automatically:

- effective concurrency = `min(case_count, len(parsed_devices))`
- there is no separate benchmark case concurrency flag in this design

## Remote Scope

When `--remote` is present, this design assumes:

- one remote host per `run-bench` invocation
- that remote host may expose multiple NPU devices
- `--npu-devices` describes the allowed device pool on that one remote host

This design does not cover multi-host fanout such as splitting one benchmark run across multiple `user@host` targets.

## Design Summary

Add a benchmark-local case scheduler that:

1. Parses `--npu-devices` when provided.
2. Creates a bounded device lease pool.
3. Enumerates benchmark cases in stable order.
4. Executes each case in an isolated local or remote case workspace.
5. Injects `ASCEND_RT_VISIBLE_DEVICES=<device>` into the one worker process running that case.
6. Collects per-case structured results.
7. Writes one final perf file in original case order.

The scheduling layer should live inside the run-eval skill scripts because benchmark case execution is already owned there.

## Affinity Layering

### Why not import `src/helix/npu_affinity.py`

The repository rule for `skills/*/scripts/` requires those scripts to remain self-contained and not import `helix`.

As a result, the new benchmark affinity helper cannot import the existing batch affinity module from `src/helix/npu_affinity.py`.

### Skill-local affinity helper

Add a small helper under:

- `skills/triton-npu-run-eval/scripts/npu_affinity.py`

That helper should mirror the existing batch-affinity behavior closely, but remain independent. It should own:

- parsing `--npu-devices`
- validating empty entries, duplicate devices, and malformed ranges
- leasing devices from a bounded pool
- converting one assigned device into:
  - `{"ASCEND_RT_VISIBLE_DEVICES": device}`

This is a semantic reuse design, not shared-code reuse.

## Device Lease Model

### Required behavior

- A benchmark case acquires a device only when its worker starts execution.
- The case holds that device for the duration of that one case.
- The device is released in a `finally` path.
- No two concurrently running case workers may hold the same device lease.

### Why case-level leases instead of one device for the full benchmark run

The goal of this feature is to let different cases run concurrently on different devices. A benchmark-wide lease would pin the whole run to one device and would not solve the problem.

## Environment Contract

For an assigned device `<D>`, inject:

- `ASCEND_RT_VISIBLE_DEVICES=<D>`

No additional diagnostic environment variable is required for this feature.

## Execution Isolation

### Requirement

When `--npu-devices` is enabled, each case must run in an isolated workspace instead of sharing `bench_file.parent` or one shared remote workspace directory.

### Why isolated workspaces are required

Benchmark case execution can create or mutate scratch files such as:

- `extra-info/`
- profiler output directories
- `msprof` output directories
- operator-specific temporary files created by the benchmark or runtime

Running multiple cases concurrently in one shared working directory would introduce races between:

- case-local `extra-info/` cleanup
- per-case profiler output production
- any benchmark-generated scratch content

### Local isolated case workspaces

In local multi-device mode:

- create one preserved or temporary run root for the benchmark invocation
- create one `case-<label>/` subdirectory per running case
- copy the minimal required input set into that case directory
- run the case from that directory

The copied input set should include:

- the benchmark file
- the operator file
- the optional sibling benchmark JSON file when present
- any standalone runtime support scripts required by the worker mode

### Remote isolated case workspaces

In remote multi-device mode:

- keep the existing one remote root workspace for the overall invocation
- create one `case-<label>/` subdirectory per case under that root
- copy or stage each case's required files into its own subdirectory
- run the remote case command from that case subdirectory

If `--keep-remote-workdir` is enabled, preserve the remote root workspace and its case subdirectories for debugging.

## `msprof` Benchmark Mode

### Current shape

`msprof` already has a natural case boundary:

1. query `python bench_x.py --num-bench`
2. run `msprof ... python bench_x.py --bench <N>` for each case
3. parse one per-case output directory

### Required refactor

Extract the current single-case `msprof` execution path into a helper that:

- receives one case index
- receives the resolved kernel metadata
- receives one case workspace and one output directory
- receives optional local or remote execution hooks
- returns one structured `PerfCaseRecord` plus execution output metadata

### Serial mode

When `--npu-devices` is omitted:

- keep the current serial behavior
- keep using the same single-case helper in a simple loop

### Parallel mode

When `--npu-devices` is provided:

- submit one worker per case
- each worker acquires a device lease
- each worker receives an isolated case workspace
- each worker injects `ASCEND_RT_VISIBLE_DEVICES=<device>`
- each worker runs its one `msprof` case and returns a structured case result

This keeps the case execution logic shared across serial and parallel modes while only changing the outer scheduling layer.

## `standalone` Benchmark Mode

### Current shape

The standalone runtime currently:

- loads benchmark and operator modules
- builds the standalone case list
- profiles each case in one process
- parses profiler output and writes one ordered perf artifact

### Required refactor

Refactor the standalone runtime around an explicit single-case execution unit.

The standalone implementation should separate:

- case discovery and ordering
- execution of one concrete `StandaloneBenchCase`
- aggregation of final perf artifacts

### Single-case helper

Extract a helper that, for one already-resolved standalone case:

- creates the case-local profiler output directory
- executes and profiles that case
- parses profiler output
- produces the structured information needed for a `PerfCaseRecord`

### Worker entrypoint

Add a single-case worker entrypoint that can:

- load the benchmark module and operator module from file paths in one isolated workspace
- rebuild the standalone case list
- resolve one `case_id`
- run only that one case
- emit a structured result payload that the parent runner can aggregate

### Serial mode

When `--npu-devices` is omitted:

- keep the current serial standalone behavior
- reuse the same single-case helper inside the existing per-case loop

### Parallel mode

When `--npu-devices` is provided:

- execute each standalone case through its own worker process
- each worker gets one isolated case workspace
- each worker gets exactly one device assignment through `ASCEND_RT_VISIBLE_DEVICES`
- the parent runner aggregates returned per-case results into the final perf file

This preserves existing standalone semantics while allowing device-isolated case parallelism.

## Worker Model

### Why workers must be process-isolated

The benchmark feature needs to bind different concurrently running cases to different NPU devices. That binding is effectively process-scoped through environment injection and runtime process state.

A thread-only model inside one Python process is not sufficient because:

- all threads would share one process environment
- device-specific runtime state would not be isolated cleanly

### Worker shape

Each parallel case worker should be one independently launched process:

- local worker for local execution
- remote command for remote execution

The parent process remains responsible for:

- device lease scheduling
- stable result ordering
- final perf file writing
- overall return code semantics

## Output And Failure Semantics

### Perf file ordering

The final perf file must remain ordered by the benchmark's original case order, not by worker completion order.

### Mixed success and failure

Once case enumeration succeeds:

- one failed case must not stop later cases from running
- successful cases must still contribute normal perf records
- failed cases must still contribute `latency-error-*` comments the same way current best-effort runners do

### Final return code

- return success only when every attempted case succeeds and no case stalls
- return failure when any case fails or stalls

### Captured output

The runner should continue aggregating stdout and stderr from all case workers into the final `ResultPayload`.

Verbose logging may include additional transient diagnostics such as:

- case id
- case index
- assigned device
- local or remote case workspace path

These diagnostics should remain additive and should not require changes to the perf artifact format.

## Remote Command Semantics

Remote execution should continue using the existing SSH command wrappers in `run_runtime.py`.

For one remote case with assigned device `<D>`, the conceptual execution shape remains:

```bash
cd <remote_case_workspace> && ASCEND_RT_VISIBLE_DEVICES=<D> <case-command>
```

This applies to both:

- remote `msprof` case execution
- remote standalone single-case workers

## File Responsibilities

### `skills/triton-npu-run-eval/scripts/run-command.py`

- add the `--npu-devices` CLI option to `run-bench`
- pass the parsed raw option through to benchmark runners

### `skills/triton-npu-run-eval/scripts/npu_affinity.py`

- new skill-local device parsing and lease helper

### `skills/triton-npu-run-eval/scripts/bench_runner.py`

- parse and validate the optional device list for benchmark execution
- add the outer case scheduler
- isolate per-case local and remote workspaces
- refactor `msprof` execution around a shared single-case helper
- orchestrate remote and local worker collection

### `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`

- refactor standalone case execution around a reusable single-case helper
- add a single-case worker entrypoint usable from parallel local and remote execution

### `skills/triton-npu-run-eval/references/run-bench.md`

- document `--npu-devices`
- explain that it enables case-level multi-device scheduling for both benchmark modes
- document remote interpretation as the device pool on the one target host

### `README.md`

- document the new `run-bench --npu-devices` behavior at the user workflow level

## Testing Strategy

Add or update tests for:

1. Parsing valid `--npu-devices` lists including numeric ranges.
2. Rejecting empty entries, duplicate devices, descending ranges, and malformed ranges.
3. Preserving current serial behavior when `--npu-devices` is omitted.
4. Local `msprof` parallel execution:
   - distinct case workspaces
   - distinct device env injection
   - stable final perf ordering
5. Remote `msprof` parallel execution:
   - remote case workspace isolation
   - SSH env prefix includes `ASCEND_RT_VISIBLE_DEVICES`
   - stable final perf ordering
6. Local standalone parallel execution:
   - reuse of the same single-case execution logic as serial mode
   - case-id based worker selection
   - distinct case workspaces
7. Remote standalone parallel execution:
   - one remote case workspace per case
   - env injection on remote commands
   - stable final perf ordering
8. Lease release after both case success and case failure.
9. Preservation of best-effort behavior when one case fails and later cases still run.
10. Preservation of `--keep-remote-workdir` semantics for the overall remote workspace tree.

## Risks And Mitigations

### Risk: skill-local affinity logic drifts from batch affinity behavior

Mitigation:

- intentionally mirror the same parsing rules and test coverage patterns already used by `src/helix/npu_affinity.py`
- keep the helper narrowly scoped and benchmark-specific

### Risk: isolated case workspaces change benchmark-relative path behavior

Mitigation:

- copy the benchmark file, operator file, optional sibling benchmark JSON, and standalone support files together into the case workspace
- preserve the relative file layout expected by current benchmark helpers

### Risk: remote staging overhead reduces gains for tiny cases

Mitigation:

- keep the feature opt-in behind `--npu-devices`
- preserve serial behavior as the default

### Risk: standalone parallel workers duplicate module import and case construction overhead

Mitigation:

- accept that overhead as the cost of true device isolation
- keep the reused single-case helper focused so the standalone logic remains maintainable

## Verification

- Run `uv run --group dev ruff check`
- Run `uv run pyright`
- Run `uv run python -m unittest discover -s tests -v`
- Run the additional strict file-scoped skill-script checks required for modified files under `skills/*/scripts/`, including:
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py`
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/npu_affinity.py`
