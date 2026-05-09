# Bench Runner And Standalone Runtime Refactor Implementation Plan

**Goal:** Split shared benchmark/perf helpers out of the eval runtime scripts, remove the unused standalone script entrypoint, and route standalone profiling through `profile-bench` as the owning orchestrator.

**Architecture:** Create small shared modules for benchmark-contract parsing and perf artifact handling. Keep `bench_runner.py` and `standalone_bench_runtime.py` as thin mode-specific layers that import those helpers. Update `profile_runner.py` to call standalone profiling functions directly so remote and local standalone profiling use the same module API.

**Tech Stack:** Python 3, `unittest`, existing repo helper scripts

---

### Task 1: Extract Shared Benchmark Contract Helpers

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/bench_contract.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] Add failing coverage for shared metadata/kernel-resolution behavior through both runtimes.
- [ ] Extract `parse_bench_metadata`, kernel-name parsing, operator-kernel discovery, stable union, and kernel-source description into the new helper.
- [ ] Repoint both runtimes to the shared helper and keep the existing error messages stable.
- [ ] Run the focused tests until they pass.

### Task 2: Extract Shared Perf Artifact Helpers

**Files:**
- Create: `skills/triton-npu-run-eval/scripts/perf_artifacts.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_standalone_bench_runtime.py`

- [ ] Add failing coverage for perf rendering/comparison through the new helper module.
- [ ] Move perf line rendering, perf file writing, perf path helpers, and perf comparison parsing into the new module.
- [ ] Keep the emitted `latency-*` and comment lines unchanged.
- [ ] Run the focused perf tests until they pass.

### Task 3: Remove The Standalone Script Entrypoint

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/standalone_bench_runtime.py`
- Modify: `skills/triton-npu-run-eval/scripts/profile_runner.py`
- Modify: `tests/test_profile_runner.py`

- [ ] Add failing coverage that `profile-bench` no longer shells out to `python3 standalone_bench_runtime.py ...`.
- [ ] Remove `main()` from the standalone runtime module.
- [ ] Make local and remote standalone profiling call the module functions directly.
- [ ] Run the profile tests until they pass.

### Task 4: Final Verification

**Files:**
- Modify: none
- Test: `tests/test_bench_runner.py`, `tests/test_profile_runner.py`, `tests/test_standalone_bench_runtime.py`

- [ ] Run the focused unit suites for bench, profile, and standalone runtime behavior.
- [ ] Run the repository's standard Python verification commands for the touched scripts if any helper scripts changed.
