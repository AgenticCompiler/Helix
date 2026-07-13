# Optimize Progress Loop From Agent Exit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make checked/supervised optimize continue or stop based on post-exit session progress instead of worker exit classification, while allowing explicitly failed rounds to count toward `--min-rounds`.

**Architecture:** Keep the accepted-round checker strict, but add a separate terminal-round progress path for the optimize controller and workflow-state closure. Replace optimize-only stall/transient/fatal orchestration with one post-exit progress check, add explicit `passed` / `failed` / `not_run` contract enums, and let workflow state close rejected terminal rounds so the next round can start cleanly.

**Tech Stack:** Python `unittest`, existing optimize runtime/controller modules, skill-side optimize-state scripts, JSON state contracts, repository trace logging

---

## File Structure

- `src/helix/optimize/execution.py`
  Owns checked/supervised multi-invocation control flow. This is where the optimize-only recovery loop must be replaced with post-exit progress accounting and controller trace events.
- `src/helix/optimize/orchestration.py`
  Owns initial batch scheduling and currently derives the next round from accepted-round count. It must align with attempted-round semantics.
- `src/helix/optimize/checks.py`
  Owns the bridge into skill-side round helpers. It may need new exported helpers for attempted versus accepted round accounting.
- `src/helix/optimize/contract.py`
  Loads optimize contract JSON and exposes shared contract lines to prompts/runtime.
- `skills/common/ascend-npu-optimize-state/references/round-contract.json`
  Source of truth for round-state required fields and new machine-readable status enums.
- `skills/common/ascend-npu-optimize-state/references/baseline-contract.json`
  Source of truth for baseline-state required fields and matching status enums.
- `skills/common/ascend-npu-optimize-state/scripts/shared/models.py`
  Owns the shared baseline/round state dataclasses and should stop treating these statuses as unconstrained strings.
- `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
  Owns accepted-round validation, completed-round iteration, and speedup helpers. It needs explicit enum validation plus terminal-round counting helpers without weakening accepted-round checks.
- `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
  Owns baseline validation and must enforce the same explicit enum contract.
- `skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`
  Owns temporary workflow-state lifecycle and currently only supports `active` / `passed` round states.
- `skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_round.py`
  Owns skill-side round submission behavior and must close rejected terminal rounds while still returning non-zero.
- `skills/triton/triton-npu-optimize/script/update-artifacts.py`
  Regenerates `skills/triton/triton-npu-optimize/references/artifacts.md` after contract updates.
- `tests/test_optimize_runtime.py`
  Verifies optimize controller loop behavior, batching, and post-exit decisions.
- `tests/test_optimize_checks.py`
  Verifies accepted-round versus terminal-round helpers and enum validation.
- `tests/test_optimize_workflow_state.py`
  Verifies workflow-state lifecycle, including closing rejected rounds.
- `tests/test_skill_command_script.py`
  Verifies `submit-round` script behavior for accepted, rejected, and unresolved rounds.
- `tests/test_optimize_round_contract.py`
  Verifies the shared round contract surface that prompt/runtime code consumes.

### Task 1: Lock The New Contracts With Failing Tests

**Files:**
- Modify: `tests/test_optimize_round_contract.py`
- Modify: `tests/test_optimize_checks.py`
- Modify: `tests/test_optimize_workflow_state.py`
- Modify: `tests/test_skill_command_script.py`

- [ ] **Step 1: Add a round-contract test that expects machine-readable status enums**

Extend `tests/test_optimize_round_contract.py` with assertions that `round-contract.json` and `baseline-contract.json` expose the shared enum values `passed`, `failed`, and `not_run`.

- [ ] **Step 2: Add a round-check test that distinguishes accepted and terminal rejected rounds**

Add focused tests in `tests/test_optimize_checks.py` that build:

```python
round_state = {
    "correctness_status": "failed",
    "benchmark_status": "not_run",
}
```

and assert:

- accepted-round helpers do not count it
- new terminal-round helpers do count it
- out-of-range values such as `"maybe"` are rejected as invalid and do not count

- [ ] **Step 3: Add a workflow-state test that a rejected round can close as failed**

Add a test in `tests/test_optimize_workflow_state.py` that bootstraps state, starts a round, calls the new close/fail path, and asserts:

- `phase` returns to `awaiting_round_start`
- `current_round` becomes `null`
- the round entry status becomes `failed`

- [ ] **Step 4: Add a submit-round script test for rejected terminal rounds**

Add a script-level test in `tests/test_skill_command_script.py` that writes a valid `round-state.json` with `correctness_status="failed"` and `benchmark_status="not_run"` and asserts:

- `submit-round` returns non-zero
- the JSON payload remains `status == "fail"`
- workflow state still closes the active round instead of leaving it `active`

- [ ] **Step 5: Run the focused tests and verify they fail for the missing behavior**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_round_contract \
  tests.test_optimize_checks \
  tests.test_optimize_workflow_state \
  tests.test_skill_command_script \
  -v
```

Expected: FAIL because enum metadata, terminal-round helpers, and rejected-round closure do not exist yet.

### Task 2: Implement Contract And Workflow-State Support

**Files:**
- Modify: `skills/common/ascend-npu-optimize-state/references/round-contract.json`
- Modify: `skills/common/ascend-npu-optimize-state/references/baseline-contract.json`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/shared/models.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/round/check.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/baseline/check.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py`
- Modify: `skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_round.py`
- Modify: `skills/triton/triton-npu-optimize/script/update-artifacts.py`

- [ ] **Step 1: Add shared enum arrays to both contract JSON files**

Add a machine-readable field such as:

```json
"status_enums": {
  "correctness_status": ["passed", "failed", "not_run"],
  "benchmark_status": ["passed", "failed", "not_run"]
}
```

to both contract JSON files.

- [ ] **Step 2: Re-run the focused contract tests and confirm they still fail on runtime behavior**

Run only the contract-focused tests so we know the surface is now present while the rest of behavior is still red.

- [ ] **Step 3: Add shared enum validation and terminal-round helpers**

Update `scripts/shared/models.py` and `scripts/round/check.py` so they provide:

- shared allowed-value constants
- explicit validation for baseline and round states
- an accepted-round helper that stays strict
- a terminal-round helper/counting function for controller progress

- [ ] **Step 4: Extend workflow-state round status lifecycle**

Update `state_machine.py` so rounds can move from `active` to `passed` or `failed`, and validation accepts `failed` as a terminal stored status.

- [ ] **Step 5: Let `submit-round` close rejected terminal rounds without reporting success**

Update `submit_round.py` so:

- accepted rounds behave exactly as before
- terminal rejected rounds return the existing fail payload but close workflow state as failed
- unresolved rounds do not close workflow state

- [ ] **Step 6: Regenerate the skill artifact documentation**

Run:

```bash
python3 skills/triton/triton-npu-optimize/script/update-artifacts.py
```

Expected: `skills/triton/triton-npu-optimize/references/artifacts.md` reflects the new enum wording.

- [ ] **Step 7: Re-run the focused contract/workflow tests and verify they pass**

Run:

```bash
uv run python -m unittest \
  tests.test_optimize_round_contract \
  tests.test_optimize_checks \
  tests.test_optimize_workflow_state \
  tests.test_skill_command_script \
  -v
```

Expected: PASS.

### Task 3: Replace Optimize-Only Recovery Branching With Post-Exit Progress Control

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/helix/optimize/execution.py`
- Modify: `src/helix/optimize/orchestration.py`
- Modify: `src/helix/optimize/checks.py`

- [ ] **Step 1: Add failing runtime tests for post-exit continuation behavior**

Extend `tests/test_optimize_runtime.py` with focused cases that assert:

- non-zero worker exit plus one new rejected terminal round continues
- non-zero worker exit plus `min_speedup` already met stops successfully
- repeated exits with no new terminal rounds stop after the no-progress limit
- a rejected terminal round advances the next `batch_start`

- [ ] **Step 2: Run just the new runtime tests and verify they fail**

Run the exact new `unittest` selectors and confirm the current controller still stops on fatal/transient/stall branches instead of post-exit progress.

- [ ] **Step 3: Add controller-visible attempted-round helpers**

Update `src/helix/optimize/checks.py` and related loader bridges so runtime code can ask for:

- attempted/terminal round directories or counts
- accepted/completed round directories or counts
- best accepted-round speedup

- [ ] **Step 4: Rewrite `run_round_loop()` around post-exit progress**

In `src/helix/optimize/execution.py`, replace optimize-only `_run_worker_with_recovery()` flow with:

- pre-launch progress snapshot
- one worker launch
- raw worker-result trace event
- post-exit progress snapshot
- no-progress counter update
- continue/stop decision based only on min speedup, min rounds, and no-progress limit

- [ ] **Step 5: Recompute batch bounds from attempted-round progress**

Update both initial and next-batch scheduling so later invocations start from `attempted_round_count + 1`.

- [ ] **Step 6: Delete obsolete optimize-only recovery branches**

Remove the optimize-only use of `classify_worker_failure`, `RecoveryBudget`, and recovery-note prompting from this controller path, while leaving backend retry infrastructure intact for non-optimize flows.

- [ ] **Step 7: Re-run the focused runtime tests and verify they pass**

Run the exact runtime selectors added in Step 1.

### Task 4: Add Controller Trace Coverage And Final Verification

**Files:**
- Modify: `tests/test_optimize_runtime.py`
- Modify: `src/helix/optimize/execution.py`

- [ ] **Step 1: Add a failing test for the new optimize controller trace events**

Extend `tests/test_optimize_runtime.py` so a checked run with a trace path asserts the controller emits:

- `optimize_loop_start`
- `optimize_iteration_start`
- `optimize_worker_result`
- `optimize_progress_check`
- `optimize_iteration_decision`
- `optimize_loop_stop`

- [ ] **Step 2: Run the trace-focused test and verify it fails**

Use an exact `unittest` selector.

- [ ] **Step 3: Add the minimal trace writes in `execution.py`**

Append structured JSONL controller events to the existing per-launch trace stream with attempted/accepted counts and decision reasons.

- [ ] **Step 4: Re-run the trace-focused test and verify it passes**

Use the same exact selector from Step 2.

- [ ] **Step 5: Run the repository verification stack**

Run:

```bash
UV_PROJECT_ENVIRONMENT=/Users/cdj/Projects/helix/.venv uv run --group dev ruff check
uv run pyright
uv run python -m pytest -q --tb=short --no-header -p no:warnings tests/
```

If a skill-side Python helper changed, also run:

```bash
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/round/check.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/baseline/check.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/state_machine.py
bash scripts/run-skill-script-pyright.sh skills/common/ascend-npu-optimize-state/scripts/state_manage/submit_round.py
```

Expected: all checks pass, or any residual failures are understood and documented before finishing.
