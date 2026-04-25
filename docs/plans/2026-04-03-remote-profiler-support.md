# Remote Profiler Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a remote-aware profiling execution path so `triton-npu-profile-operator` can collect and summarize `msprof` output through the same remote execution model already used by validation and optimize workflows.

**Architecture:** Keep profiler orchestration in `skills/triton-npu-run-eval/scripts/` instead of teaching the profiler skill to manage SSH details itself. Add a new `profile-bench` helper entrypoint that resolves benchmark metadata, branches by `bench-mode`, reuses the existing remote workspace runtime helpers, copies the relevant profiler artifacts back to the local workspace, and then lets `triton-npu-profile-operator` summarize those local artifacts. Update the profiler skill so it documents the mode-specific benchmark contract and prefers the new helper over ad hoc `msprof <command>` invocations.

**Tech Stack:** Python 3.11, `argparse`, existing triton-npu-run-eval skill scripts, `unittest`, Markdown docs and skill guides

---

### Task 1: Add failing command and runtime tests for profiler execution

**Files:**
- Modify: `tests/test_bench_runner.py`
- Modify: `tests/test_remote_execution.py`
- Modify: `tests/test_generation_contracts.py`
- Modify: `tests/test_ascend_npu_operator_profiler.py`

- [ ] **Step 1: Add failing tests for new `profile-bench` parser support and `bench-mode` metadata resolution**
- [ ] **Step 2: Add failing local runner tests for `standalone` profiling command construction**
- [ ] **Step 3: Add failing local runner tests for `msprof` profiling command construction, required `# kernel:` metadata, and selected case handling**
- [ ] **Step 4: Add failing remote runner tests for remote workspace creation, artifact copy-back, quoted filenames, and keep-or-clean workspace behavior**
- [ ] **Step 5: Add failing contract tests that require `triton-npu-profile-operator` to document different argument rules for `standalone` and `msprof` benchmark modes**
- [ ] **Step 6: Run the targeted unittest cases and confirm the new profiler behavior is not implemented yet**

### Task 2: Implement reusable profiling execution helpers

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `skills/triton-npu-run-eval/scripts/bench_runner.py`
- Modify: `skills/triton-npu-run-eval/scripts/run_runtime.py`
- Create: `skills/triton-npu-run-eval/scripts/profile_runner.py`

- [ ] **Step 1: Add a `profile-bench` subcommand with explicit profiler-facing flags such as `--bench-file`, `--operator-file`, optional `--bench`, optional `--target-op`, `--remote`, `--remote-workdir`, `--keep-remote-workdir`, and `--verbose`**
- [ ] **Step 2: Reuse existing benchmark metadata parsing so omitted `--bench-mode` still resolves from the benchmark header**
- [ ] **Step 3: Implement local `standalone` profiling by wrapping `python3 <bench> --operator-file <operator>` with `msprof` and returning the generated profile directory**
- [ ] **Step 4: Implement local `msprof` profiling by validating `# kernel:`, selecting one benchmark case, and running the case-specific profiler command without conflating it with perf-only `run-bench` behavior**
- [ ] **Step 5: Implement remote profiling with existing workspace helpers for SSH, `scp`, remote command execution, artifact copy-back, and optional retained remote workspaces**
- [ ] **Step 6: Add a focused helper for locating the produced `PROF_*` directory and copying the minimum required profiler output back to the local workspace**
- [ ] **Step 7: Make command failures short and actionable when the benchmark mode, required metadata, selected benchmark case, or profiler output is invalid**
- [ ] **Step 8: Run targeted unittests for the new helper module and make them pass**

### Task 3: Update profiler and validation skill contracts

**Files:**
- Modify: `skills/triton-npu-profile-operator/SKILL.md`
- Modify: `skills/triton-npu-run-eval/SKILL.md`
- Modify: `skills/triton-npu-optimize/SKILL.md`

- [ ] **Step 1: Rewrite the profiler skill to prefer `../triton-npu-run-eval/scripts/run-command.py profile-bench` as the default execution path**
- [ ] **Step 2: Document that `standalone` benchmarks profile `python3 bench_<op>.py --operator-file <operator-file>` and must not receive `--bench` or `--num-bench`**
- [ ] **Step 3: Document that `msprof` benchmarks first query `--num-bench`, then profile one selected `--bench <N>` case, and require resolvable `# kernel:` metadata**
- [ ] **Step 4: Document that remote-aware optimize or validation workflows must pass the same `--remote` and `--remote-workdir` settings through profiler execution**
- [ ] **Step 5: Keep direct `msprof <command>` guidance only as a local fallback for cases where the unified helper is not the right tool**
- [ ] **Step 6: Update the triton-npu-run-eval and optimize skill text so the new profiling helper is discoverable from the main execution path**

### Task 4: Update user-facing documentation and design notes

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/notes/2026-04-02-ascend-npu-operator-profiler-skill.md`
- Create: `docs/notes/2026-04-03-remote-profiler-support.md`

- [ ] **Step 1: Write a short design doc that explains why profiler remote support belongs in the unified triton-npu-run-eval runtime instead of inside the profiler skill**
- [ ] **Step 2: Add README examples for local and remote `profile-bench` usage and note the `standalone` versus `msprof` argument differences**
- [ ] **Step 3: Update `AGENTS.md` so the durable project rules describe profiler remote support as part of the unified validation execution layer**
- [ ] **Step 4: Update the earlier profiler-skill design doc so it no longer presents direct `msprof <command>` as the only default workflow**

### Task 5: Verify the full change

**Files:**
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`
- Modify: `skills/triton-npu-run-eval/scripts/profile_runner.py`
- Modify: `skills/triton-npu-profile-operator/SKILL.md`
- Modify: `README.md`

- [ ] **Step 1: Run the targeted profiler and remote-execution unittests**
- [ ] **Step 2: Run `uv run python -m unittest discover -s tests -v`**
- [ ] **Step 3: Run `uv run --group dev ruff check`**
- [ ] **Step 4: Run `uv run pyright`**
- [ ] **Step 5: Fix any regressions and re-run verification until clean**
