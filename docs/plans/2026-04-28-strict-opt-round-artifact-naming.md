# Strict Opt Round Artifact Naming Implementation Plan

**Goal:** Enforce `opt_<original>.py` and `opt_<original>_perf.txt` as the only valid optimize round operator/perf artifact names, and make `run-bench` print an explicit saved-path message at the end.

**Architecture:** Tighten the optimize round contract around the original workspace operator name, remove permissive round artifact inference, and keep `baseline/perf.txt` unchanged. Update both CLI entrypoints so `run-bench` clearly reports where the perf file was saved.

**Tech Stack:** Python, unittest, repository docs

---

### Task 1: Add failing tests for run-bench saved-path messaging

**Files:**
- Modify: `tests/test_execution_commands.py`
- Modify: `tests/test_cli.py`

- [ ] Add a unit test in `tests/test_execution_commands.py` that expects `handle_run_bench(...)` to print both `Perf file: ...` and `Saved perf file to: ...` when a perf path is returned.
- [ ] Add a CLI-level test in `tests/test_cli.py` that expects `main(["run-bench", ...])` to include the same saved-path message.
- [ ] Run the two targeted tests and confirm they fail before implementation.

### Task 2: Add failing tests for strict round operator/perf naming

**Files:**
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_status.py`
- Modify: `tests/test_verify.py`
- Modify: `tests/test_optimize_runtime.py`

- [ ] Update round-fixture helpers to create `opt_<original>.py` and `opt_<original>_perf.txt` instead of `kernel.py` and `perf.txt`.
- [ ] Add or update tests so `check-round` fails when the round-local operator is not `opt_<original>.py`.
- [ ] Add or update tests so `check-round` fails when `round-state.json["perf_artifact"]` is not `opt_<original>_perf.txt`.
- [ ] Update status tests to require round comparisons from `opt_<original>_perf.txt` only.
- [ ] Update verify tests to expect `prepare_verify_target(...)` to select and copy `opt_<original>.py`.
- [ ] Run the targeted tests and confirm they fail before implementation.

### Task 3: Implement strict round artifact resolution

**Files:**
- Modify: `skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`
- Modify: `src/triton_agent/status/core.py`
- Modify: `src/triton_agent/verification/core.py`

- [ ] Replace permissive round operator discovery with a helper that resolves the original workspace operator and requires `opt_<original>.py` inside `opt-round-N/`.
- [ ] Replace permissive round perf discovery with a helper that requires `opt_<original>_perf.txt`.
- [ ] Make `check-round` reject any `round-state.json["perf_artifact"]` that does not match the expected filename.
- [ ] Keep baseline handling unchanged.

### Task 4: Implement run-bench saved-path messaging

**Files:**
- Modify: `src/triton_agent/commands/execution.py`
- Modify: `skills/triton-npu-run-eval/scripts/run-command.py`

- [ ] After `run-bench` succeeds and returns a perf path, print `Saved perf file to: <path>` after the existing perf-path line.
- [ ] Keep behavior unchanged when no perf path is returned.

### Task 5: Update docs and workflow prompts

**Files:**
- Modify: `skills/triton/triton-npu-optimize/SKILL.md`
- Modify: `skills/triton/triton-npu-optimize/references/artifacts.md`
- Modify: `src/triton_agent/optimize/prompts.py`

- [ ] Update optimize workflow text to require `opt_<original>.py` in each round.
- [ ] Update artifact docs to require `opt_<original>_perf.txt` instead of `perf.txt`.
- [ ] Add prompt guidance that round perf artifacts must be the generated `opt_<original>_perf.txt`.

### Task 6: Verify and close out

**Files:**
- Modify only as needed from earlier tasks

- [ ] Run the targeted unittest commands covering execution, CLI, optimize-check, status, verify, and optimize runtime.
- [ ] Run `bash scripts/run-skill-script-pyright.sh skills/triton-npu-optimize-submit-round/scripts/optimize_submit_round_contract.py`.
- [ ] Run `bash scripts/run-skill-script-pyright.sh skills/triton-npu-run-eval/scripts/run-command.py`.
- [ ] If any test or pyright check fails, fix and rerun before completion.
