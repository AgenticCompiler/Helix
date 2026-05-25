# Run-Bench Multi-Device Parallelism Implementation Plan

**Goal:** Add an opt-in `run-bench --npu-devices ...` mode that executes benchmark cases concurrently across multiple Ascend NPU devices for both local and remote execution, while preserving current serial behavior when the option is omitted.

**Architecture:** Keep the feature inside the run-eval skill runtime. Add a skill-local affinity helper instead of importing `src/triton_agent`, use process-isolated case workers so device assignment remains real and enforceable, isolate each parallel case in its own local or remote case workspace, and aggregate all per-case results back into the existing ordered perf artifact format. Reuse the same single-case execution logic in both serial and parallel paths, especially for standalone benchmarks.

**Tech Stack:** Python 3, `argparse`, `unittest`, `concurrent.futures`, existing run-eval helper scripts, SSH-based remote execution helpers

---

## Task 1: Add Skill-Local Device Parsing And Lease Helpers

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/npu_affinity.py`
- Modify: `tests/test_bench_runner.py`

- [ ] Add focused tests for parsing `--npu-devices` values, including whitespace trimming, numeric range expansion, empty-entry rejection, duplicate rejection, descending-range rejection, and malformed-range rejection.
- [ ] Implement a skill-local parser and lease pool that mirrors the existing batch-affinity semantics without importing `triton_agent`.
- [ ] Keep the environment builder minimal and return only `{"ASCEND_RT_VISIBLE_DEVICES": device}`.
- [ ] Run the focused affinity-related bench tests until they pass.

## Task 2: Thread `--npu-devices` Through `run-bench`

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `tests/test_bench_runner.py`

- [ ] Add failing CLI-level coverage that `run-bench` accepts `--npu-devices` and forwards the raw value to the benchmark runner.
- [ ] Extend the `run-bench` parser with the new option while keeping all existing arguments and behavior stable.
- [ ] Update the local and remote benchmark runner call signatures to accept the optional raw device-list argument.
- [ ] Confirm that omitting `--npu-devices` still uses the existing serial path.
- [ ] Run the focused `run-command` and bench runner tests until they pass.

## Task 3: Refactor `msprof` Into Reusable Single-Case Execution Helpers

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `tests/test_bench_runner.py`

- [ ] Add failing coverage that isolates the current `msprof` single-case execution responsibilities from the outer per-case loop.
- [ ] Extract one local single-case `msprof` helper that receives one case index, one output directory, and resolved kernel metadata, then returns the structured result needed to build a `PerfCaseRecord`.
- [ ] Extract the corresponding remote single-case `msprof` helper with the same responsibilities using the existing SSH runtime wrappers.
- [ ] Rewire the current serial `msprof` path so it uses the extracted helper without changing existing output semantics.
- [ ] Run the focused `msprof` tests until they pass.

## Task 4: Add Local And Remote Parallel `msprof` Scheduling

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `tests/test_bench_runner.py`

- [ ] Add failing coverage that local parallel `msprof` execution creates isolated case workspaces, injects per-case `ASCEND_RT_VISIBLE_DEVICES`, and still writes the final perf file in original case order.
- [ ] Add failing coverage that remote parallel `msprof` execution creates one remote case workspace per case and injects `ASCEND_RT_VISIBLE_DEVICES` into the SSH command prefix.
- [ ] Add the outer local parallel case scheduler that acquires a device lease per case, launches one isolated worker process per case, and aggregates case results.
- [ ] Add the outer remote parallel case scheduler with the same semantics over the existing remote execution helpers.
- [ ] Preserve best-effort behavior when one `msprof` case fails and later cases still run.
- [ ] Run the focused local and remote `msprof` tests until they pass.

## Task 5: Refactor Standalone Execution Around A Reusable Single-Case Helper

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] Add failing coverage that serial standalone execution reuses an explicit single-case execution unit instead of hardwiring all work into one monolithic loop.
- [ ] Extract one helper that executes one already-resolved `StandaloneBenchCase`, profiles it, parses the profiler output, and returns the structured data needed for a `PerfCaseRecord`.
- [ ] Keep existing serial standalone behavior intact by switching the current loop to the new single-case helper.
- [ ] Add a case-id based worker entrypoint that can load the benchmark and operator modules from disk, rebuild the case list, resolve one case, and run only that case.
- [ ] Run the standalone runtime tests until they pass.

## Task 6: Add Local And Remote Parallel Standalone Scheduling

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] Add failing coverage that local standalone parallel execution launches isolated per-case worker processes and preserves final perf ordering.
- [ ] Add failing coverage that remote standalone parallel execution creates one remote case workspace per case, injects `ASCEND_RT_VISIBLE_DEVICES`, and aggregates the returned case results into one perf artifact.
- [ ] Implement local standalone parallel scheduling using the new case-id worker entrypoint.
- [ ] Implement remote standalone parallel scheduling using the same worker entrypoint and the existing remote execution wrappers.
- [ ] Preserve current mixed-success behavior: one standalone case failure must not stop later cases, and the final return code must still fail if any case failed.
- [ ] Run the focused standalone bench tests until they pass.

## Task 7: Add Parallel Case Workspace Isolation Helpers

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `tests/test_bench_runner.py`

- [ ] Add failing coverage that parallel mode does not reuse `bench_file.parent` directly and does not allow `extra-info/` or other scratch outputs to collide between concurrent cases.
- [ ] Add local case-workspace staging helpers that copy the benchmark file, operator file, optional sibling benchmark JSON, and standalone support files into one per-case subdirectory.
- [ ] Add remote case-workspace staging helpers that create one case directory under the shared remote root and stage the same minimal input set there.
- [ ] Keep `--keep-remote-workdir` semantics intact by preserving the full remote root tree when requested.
- [ ] Run the focused staging and cleanup tests until they pass.

## Task 8: Update User Docs And Skill Docs

**Files:**
- Modify: `skills/triton-npu-run-eval/references/run-bench.md`
- Modify: `README.md`

- [ ] Document `--npu-devices` as an opt-in case-level multi-device feature for `run-bench`.
- [ ] Document that the option works for both `msprof` and `standalone`.
- [ ] Document that `--remote` continues to target one remote host and that `--npu-devices` describes the device pool on that one host.
- [ ] Keep the docs focused on user-visible behavior rather than internal worker details.

## Task 9: Final Verification

**Files:**
- Modify: none

- [ ] Run focused unit suites for bench runner and standalone runtime behavior.
- [ ] Run the repository verification commands:
  - `uv run --group dev ruff check`
  - `uv run pyright`
  - `uv run python -m unittest discover -s tests -v`
- [ ] Run the required strict file-scoped skill-script checks for all modified files under `skills/*/scripts/`, including:
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/bench_runner.py`
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
  - `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/npu_affinity.py`
